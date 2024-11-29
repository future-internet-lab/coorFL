import os
import pika
import pickle
import argparse
import sys
import yaml

import torch
import requests
import random
import numpy as np

import src.Model
import src.Log
from src.Selection import client_selection_algorithm
from src.Cluster import clustering_algorithm

from requests.auth import HTTPBasicAuth

parser = argparse.ArgumentParser(description="Split learning framework with controller.")

# parser.add_argument('--topo', type=int, nargs='+', required=True, help='List of client topo, example: --topo 2 3')

args = parser.parse_args()

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

total_clients = config["server"]["clients"]
model_name = config["server"]["model"]
data_name = config["server"]["data-name"]
address = config["rabbit"]["address"]
username = config["rabbit"]["username"]
password = config["rabbit"]["password"]

num_round = config["server"]["num-round"]
save_parameters = config["server"]["parameters"]["save"]
load_parameters = config["server"]["parameters"]["load"]
validation = config["server"]["validation"]

# Algorithm
data_mode = config["server"]["data-mode"]
client_selection_mode = config["server"]["client-selection"]
client_cluster_mode = config["server"]["client-cluster"]

# Clients
batch_size = config["learning"]["batch-size"]
lr = config["learning"]["learning-rate"]
momentum = config["learning"]["momentum"]

log_path = config["log_path"]

num_labels = 10


class Server:
    def __init__(self):
        credentials = pika.PlainCredentials(username, password)
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(address, 5672, '/', credentials))
        self.channel = self.connection.channel()
        self.num_round = num_round
        self.round = self.num_round

        self.channel.queue_declare(queue='rpc_queue')

        self.total_clients = total_clients
        self.current_clients = 0
        self.updated_clients = 0
        self.responses = {}  # Save response
        self.list_clients = []
        self.all_model_parameters = []
        self.avg_state_dict = None
        self.round_result = True

        if data_mode == "even":
            # self.label_counts = [[250 for _ in range(num_labels)] for _ in range(total_clients)]
            self.label_counts = [[148 for _ in range(num_labels)], [148 for _ in range(num_labels)],
                                 [273 for _ in range(num_labels)], [4430 for _ in range(num_labels)]]
        else:
            self.label_counts = [[random.randint(0, 500) for _ in range(num_labels)] for _ in range(total_clients)]
        # self.speeds = [325, 788, 857, 915, 727, 270, 340, 219, 725, 228, 677, 259, 945, 433, 222, 979, 339, 864, 858, 621, 242, 790, 807, 368, 259, 776, 218, 845, 294, 340, 731, 595, 799, 524, 779, 581, 456, 574, 754, 771]
        self.speeds = [25, 20, 77, 33, 74, 25, 77, 54, 39, 88, 36, 76, 34, 37, 84, 85, 80, 28, 44, 20, 87, 57, 86, 43,
                       90, 58, 23, 41, 35, 41, 21, 60, 92, 81, 37, 30, 85, 79, 84, 22]
        self.selected_client = []

        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(queue='rpc_queue', on_message_callback=self.on_request)
        self.logger = src.Log.Logger(f"{log_path}/app.log")
        self.logger.log_info("### Application start ###\n")
        src.Log.print_with_color(f"Server is waiting for {self.total_clients} clients.", "green")

    def on_request(self, ch, method, props, body):
        message = pickle.loads(body)
        routing_key = props.reply_to
        action = message["action"]
        client_id = message["client_id"]
        self.responses[routing_key] = message

        if action == "REGISTER":
            if str(client_id) not in self.list_clients:
                self.list_clients.append(str(client_id))
                src.Log.print_with_color(f"[<<<] Received message from client: {message}", "blue")

            # If consumed all clients - Register for first time
            if len(self.list_clients) == self.total_clients:
                src.Log.print_with_color("All clients are connected. Sending notifications.", "green")
                self.client_selection()
                src.Log.print_with_color(f"Start training round {self.num_round - self.round + 1}", "yellow")
                self.notify_clients()
        elif action == "UPDATE":
            data_message = message["message"]
            result = message["result"]
            src.Log.print_with_color(f"[<<<] Received message from client: {data_message}", "blue")
            self.updated_clients += 1
            # Save client's model parameters
            if not result:
                self.round_result = False

            if save_parameters and self.round_result:
                model_state_dict = message["parameters"]
                self.all_model_parameters.append(model_state_dict)

            # If consumed all client's parameters
            if self.updated_clients == len(self.selected_client):
                self.updated_clients = 0
                src.Log.print_with_color("Collected all parameters.", "yellow")
                if save_parameters and self.round_result:
                    self.avg_all_parameters()
                    self.all_model_parameters = []
                # Test
                if save_parameters and validation and self.round_result:
                    src.Model.test(model_name, data_name, self.logger)
                # Start a new training round
                if not self.round_result:
                    src.Log.print_with_color(f"Training failed!", "yellow")
                else:
                    self.round -= 1
                self.round_result = True

                if self.round > 0:
                    src.Log.print_with_color(f"Start training round {self.num_round - self.round + 1}", "yellow")
                    self.client_selection()
                    if save_parameters:
                        self.notify_clients()
                    else:
                        self.notify_clients(register=False)
                else:
                    self.notify_clients(start=False)
                    sys.exit()

        # Ack the message
        ch.basic_ack(delivery_tag=method.delivery_tag)

    def notify_clients(self, start=True, register=True):
        # Send message to clients when consumed all clients
        if start:
            filepath = f'{model_name}.pth'
            # Read parameters file
            state_dict = None
            if load_parameters and register:
                if os.path.exists(filepath):
                    state_dict = torch.load(filepath, weights_only=False)
            for i in self.selected_client:
                client_id = self.list_clients[i]
                # Request clients to start training
                src.Log.print_with_color(f"[>>>] Sent start training request to client {client_id}", "red")
                response = {"action": "START",
                            "message": "Server accept the connection!",
                            "model_name": model_name,
                            "data_name": data_name,
                            "parameters": state_dict,
                            "label_counts": self.label_counts[i],
                            "batch_size": batch_size,
                            "lr": lr,
                            "momentum": momentum}
                self.send_to_response(client_id, pickle.dumps(response))
            if data_mode != "even":
                self.label_counts = [[random.randint(0, 1000) for _ in range(num_labels)] for _ in range(total_clients)]
        else:
            for client_id in self.list_clients:
                # Request clients to stop process
                src.Log.print_with_color(f"[>>>] Sent stop training request to client {client_id}", "red")
                response = {"action": "STOP",
                            "message": "Stop training!",
                            "parameters": None}
                self.send_to_response(client_id, pickle.dumps(response))

    def start(self):
        self.channel.start_consuming()

    def send_to_response(self, client_id, message):
        reply_channel = self.connection.channel()
        reply_queue_name = f'reply_{client_id}'
        reply_channel.queue_declare(reply_queue_name, durable=False)

        src.Log.print_with_color(f"[>>>] Sent notification to client {client_id}", "red")
        reply_channel.basic_publish(
            exchange='',
            routing_key=reply_queue_name,
            body=message
        )

    def client_selection(self):
        local_speeds = self.speeds[:len(self.list_clients)]
        num_datas = [np.sum(self.label_counts[i]) for i in range(len(self.list_clients))]
        total_training_time = np.array(num_datas) / np.array(local_speeds)

        if client_selection_mode:
            if client_cluster_mode:
                num_cluster, labels = clustering_algorithm(self.label_counts)
                self.selected_client = []
                for i in range(num_cluster):
                    cluster_client = [index for index, label in enumerate(labels) if label == i]
                    self.selected_client += client_selection_algorithm(cluster_client, local_speeds, num_datas)
            else:
                self.selected_client = client_selection_algorithm([i for i in range(len(self.list_clients))],
                                                                  local_speeds, num_datas)
        else:
            self.selected_client = [i for i in range(len(self.list_clients))]

        # From client selected, calculate and log training time
        training_time = np.max([total_training_time[i] for i in self.selected_client])
        self.logger.log_info(f"Active with {len(self.selected_client)} client: {self.selected_client}")
        self.logger.log_info(f"Total training time round = {training_time}")

    def avg_all_parameters(self):
        # Average all client parameters
        state_dicts = self.all_model_parameters
        num_models = len(state_dicts)

        if num_models == 0:
            return

        self.avg_state_dict = state_dicts[0]

        for key in state_dicts[0].keys():
            for i in range(1, num_models):
                self.avg_state_dict[key] += state_dicts[i][key]

            if self.avg_state_dict[key].dtype != torch.long:
                self.avg_state_dict[key] /= num_models
            else:
                self.avg_state_dict[key] //= num_models

        # Save to files
        torch.save(self.avg_state_dict, f'{model_name}.pth')


def delete_old_queues():
    url = f'http://{address}:15672/api/queues'
    response = requests.get(url, auth=HTTPBasicAuth(username, password))

    if response.status_code == 200:
        queues = response.json()

        credentials = pika.PlainCredentials(username, password)
        connection = pika.BlockingConnection(pika.ConnectionParameters(address, 5672, '/', credentials))
        http_channel = connection.channel()

        for queue in queues:
            queue_name = queue['name']
            if queue_name.startswith("reply") or queue_name.startswith("intermediate_queue") or queue_name.startswith(
                    "gradient_queue"):
                try:
                    http_channel.queue_delete(queue=queue_name)
                    src.Log.print_with_color(f"Queue '{queue_name}' deleted.", "green")
                except Exception as e:
                    src.Log.print_with_color(f"Failed to delete queue '{queue_name}': {e}", "yellow")
            else:
                try:
                    http_channel.queue_purge(queue=queue_name)
                    src.Log.print_with_color(f"Queue '{queue_name}' purged.", "green")
                except Exception as e:
                    src.Log.print_with_color(f"Failed to purge queue '{queue_name}': {e}", "yellow")

        connection.close()
        return True
    else:
        src.Log.print_with_color(
            f"Failed to fetch queues from RabbitMQ Management API. Status code: {response.status_code}", "yellow")
        return False


if __name__ == "__main__":
    delete_old_queues()
    server = Server()
    server.start()
    src.Log.print_with_color("Ok, ready!", "green")
