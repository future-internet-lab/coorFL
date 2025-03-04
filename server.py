import os
import pika
import pickle
import argparse
import sys
import yaml
import signal

import torch
import requests
import random
import numpy as np

import src.Validation
import src.Log
from src.Selection import client_selection_speed_base, client_selection_random
from src.Cluster import clustering_algorithm
from src.Utils import DomainDataset, generate_random_array
from src.Notify import send_mail

from requests.auth import HTTPBasicAuth

parser = argparse.ArgumentParser(description="Federated learning framework with controller.")

parser.add_argument('--device', type=str, required=False, help='Device of client')

args = parser.parse_args()

device = None

if args.device is None:
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using device: {torch.cuda.get_device_name(device)}")
    else:
        device = "cpu"
        print(f"Using device: CPU")
else:
    device = args.device
    print(f"Using device: {device}")

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
random_seed = config["server"]["random-seed"]
accuracy_drop = config["server"]["accuracy-drop"]

data_distribution = config["server"]["data-distribution"]
data_range = data_distribution["num-data-range"]
non_iid_rate = data_distribution["non-iid-rate"]
refresh_each_round = data_distribution["refresh-each-round"]

stop_when_false = config["server"]["stop-when-false"]
email_config = config["server"]["send-mail"]

# Algorithm
data_mode = config["server"]["data-mode"]
client_selection_config = config["server"]["client-selection"]
client_cluster_config = config["server"]["client-cluster"]

# Clients
batch_size = config["learning"]["batch-size"]
lr = config["learning"]["learning-rate"]
momentum = config["learning"]["momentum"]
clip_grad_norm = config["learning"]["clip-grad-norm"]


log_path = config["log_path"]

if data_name == "CIFAR10" or data_name == "MNIST":
    num_labels = 10
elif data_name == "DOMAIN":
    num_labels = 21
elif data_name == "DOMAIN2":
    num_labels = 2
else:
    num_labels = 0


if random_seed:
    random.seed(random_seed)


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
        self.last_accuracy = 0.0
        self.responses = {}  # Save response
        self.list_clients = []
        self.all_model_parameters = []
        self.avg_state_dict = None
        self.round_result = True
        self.label_counts = None
        self.non_iid_label = None
        if not refresh_each_round:
            if data_name == "DOMAIN":
                self.non_iid_label = [np.insert(src.Utils.non_iid_rate(num_labels - 1, non_iid_rate), 0, 1) for _ in range(self.total_clients)]
            else:
                self.non_iid_label = [src.Utils.non_iid_rate(num_labels, non_iid_rate) for _ in range(self.total_clients)]

        # self.speeds = [325, 788, 857, 915, 727, 270, 340, 219, 725, 228, 677, 259, 945, 433, 222, 979, 339, 864, 858, 621, 242, 790, 807, 368, 259, 776, 218, 845, 294, 340, 731, 595, 799, 524, 779, 581, 456, 574, 754, 771]
        self.speeds = [25, 20, 77, 33, 74, 25, 77, 54, 39, 88, 36, 76, 34, 37, 84, 85, 80, 28, 44, 20, 87, 57, 86, 43,
                       90, 58, 23, 41, 35, 41, 21, 60, 92, 81, 37, 30, 85, 79, 84, 22]
        self.selected_client = []

        self.logger = src.Log.Logger(f"{log_path}/app.log")
        self.validation = src.Validation.Validation(model_name, data_name, self.logger)

        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(queue='rpc_queue', on_message_callback=self.on_request)

        self.logger.log_info("### Application start ###\n")
        src.Log.print_with_color(f"Server is waiting for {self.total_clients} clients.", "green")

    def start(self):
        self.channel.start_consuming()

    def data_distribution(self):
        if data_name == "DOMAIN":
            if data_mode == "even":
                self.label_counts = np.array([[25000 // total_clients] + [1250 // total_clients for _ in range(num_labels-1)]
                                     for _ in range(total_clients)])
            else:
                if refresh_each_round:
                    self.non_iid_label = [np.insert(src.Utils.non_iid_rate(num_labels-1, non_iid_rate), 0, 1) for _ in range(self.total_clients)]
                self.label_counts = [np.array(
                                        [random.randint(data_range[0]*(num_labels-1), data_range[1]*(num_labels-1))] +
                                        [random.randint(data_range[0] // non_iid_rate, data_range[1] // non_iid_rate) for _ in range(num_labels-1)])
                                     * self.non_iid_label[i] for i in range(total_clients)]

        else:
            if data_mode == "even":
                self.label_counts = np.array([[5000 // total_clients for _ in range(num_labels)] for _ in range(total_clients)])
            else:
                if refresh_each_round:
                    self.non_iid_label = [src.Utils.non_iid_rate(num_labels, non_iid_rate) for _ in range(self.total_clients)]
                self.label_counts = [np.array([random.randint(data_range[0]//non_iid_rate, data_range[1]//non_iid_rate)
                                              for _ in range(num_labels)]) *
                                     self.non_iid_label[i] for i in range(total_clients)]

    def send_to_response(self, client_id, message):
        """
        Response message to clients
        :param client_id: client ID
        :param message: message
        :return:
        """
        reply_channel = self.channel
        reply_queue_name = f'reply_{client_id}'
        reply_channel.queue_declare(reply_queue_name, durable=False)

        src.Log.print_with_color(f"[>>>] Sent notification to client {client_id}", "red")
        reply_channel.basic_publish(
            exchange='',
            routing_key=reply_queue_name,
            body=message
        )

    def on_request(self, ch, method, props, body):
        """
        Handler request from clients
        :param ch: channel
        :param method:
        :param props:
        :param body: message body
        :return:
        """
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
                self.data_distribution()
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
                client_size = message["size"]
                self.all_model_parameters.append({'client_id': client_id, 'weight': model_state_dict,
                                                  'size': client_size})

            # If consumed all client's parameters
            if self.updated_clients == len(self.selected_client):
                self.process_consumer()

        # Ack the message
        ch.basic_ack(delivery_tag=method.delivery_tag)

    def process_consumer(self):
        """
        After collect all training clients, start validation and make decision for the next training round
        :return:
        """
        self.updated_clients = 0
        src.Log.print_with_color("Collected all parameters.", "yellow")
        # TODO: detect model poisoning with self.all_model_parameters at here
        if save_parameters and self.round_result:
            self.avg_all_parameters()

            self.all_model_parameters = []
        # Server validation
        accuracy = 0.0
        if save_parameters and validation and self.round_result:
            self.round_result, accuracy = self.validation.test(self.avg_state_dict, device)

        if not self.round_result:
            src.Log.print_with_color(f"Training failed!", "yellow")
            send_mail(email_config, f"Quá trình training bị lỗi tại round {self.num_round - self.round + 1}")
            if stop_when_false:
                # Stop training
                self.notify_clients(start=False)
                delete_old_queues()
                sys.exit()
        elif self.last_accuracy - accuracy > accuracy_drop:
            src.Log.print_with_color(f"Accuracy drop!", "yellow")
        else:
            self.last_accuracy = accuracy
            # Save to files
            torch.save(self.avg_state_dict, f'{model_name}.pth')
            self.round -= 1
        self.round_result = True

        if self.round > 0:
            # Start a new training round
            src.Log.print_with_color(f"Start training round {self.num_round - self.round + 1}", "yellow")
            self.data_distribution()
            self.client_selection()
            self.notify_clients()
        else:
            # Stop training
            send_mail(email_config, f"Đã hoàn thành quá trình training")
            self.notify_clients(start=False)
            delete_old_queues()
            sys.exit()

    def notify_clients(self, start=True):
        """
        Control message to clients
        :param start: If True (default), request clients to start. Else if False, stop training
        :return:
        """
        # Send message to clients when consumed all clients
        if start:
            filepath = f'{model_name}.pth'
            # Read parameters file
            state_dict = None
            if load_parameters:
                if os.path.exists(filepath):
                    state_dict = torch.load(filepath, weights_only=True)

            count_labels = np.zeros(num_labels)
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
                            "momentum": momentum,
                            "clip_grad_norm": clip_grad_norm}
                count_labels += self.label_counts[i]
                self.send_to_response(client_id, pickle.dumps(response))

            self.logger.log_info(f"All training labels count = {count_labels.tolist()}")
        else:
            for client_id in self.list_clients:
                # Request clients to stop process
                src.Log.print_with_color(f"[>>>] Sent stop training request to client {client_id}", "red")
                response = {"action": "STOP",
                            "message": "Stop training!",
                            "parameters": None}
                self.send_to_response(client_id, pickle.dumps(response))

    def client_selection(self):
        """
        Select the specific clients
        :return: The list contain index of active clients: `self.selected_client`.
        E.g. `self.selected_client = [2,3,5]` means client 2, 3 and 5 will train this current round
        """
        local_speeds = self.speeds[:len(self.list_clients)]
        num_datas = [np.sum(self.label_counts[i]) for i in range(len(self.list_clients))]
        total_training_time = np.array(num_datas) / np.array(local_speeds)

        if client_selection_config['enable']:
            if client_cluster_config['enable']:
                num_cluster, labels, _ = clustering_algorithm(self.label_counts, client_cluster_config)
                self.logger.log_info(f"Num cluster = {num_cluster}, labels = {labels}")
                self.selected_client = []
                for i in range(num_cluster):
                    cluster_client = [index for index, label in enumerate(labels) if label == i]
                    if client_selection_config['mode'] == 'speed':
                        self.selected_client += client_selection_speed_base(cluster_client, local_speeds, num_datas)
                    elif client_selection_config['mode'] == 'random':
                        self.selected_client += client_selection_random(cluster_client)
            else:
                if client_selection_config['mode'] == 'speed':
                    self.selected_client = client_selection_speed_base([i for i in range(len(self.list_clients))],
                                                                       local_speeds, num_datas)
                elif client_selection_config['mode'] == 'random':
                    self.selected_client += client_selection_random([i for i in range(len(self.list_clients))])
        else:
            self.selected_client = [i for i in range(len(self.list_clients))]

        # From client selected, calculate and log training time
        training_time = np.max([total_training_time[i] for i in self.selected_client])
        self.logger.log_info(f"Active with {len(self.selected_client)} client: {self.selected_client}")
        self.logger.log_info(f"Total training time round = {training_time}")

    def avg_all_parameters(self):
        """
        Consuming all client's weight from `self.all_model_parameters` - a list contain all client's weight
        :return: Global weight on `self.avg_state_dict`
        """
        # Average all client parameters
        num_models = len(self.all_model_parameters)

        if num_models == 0:
            return

        self.avg_state_dict = self.all_model_parameters[0]['weight']
        all_client_sizes = [item['size'] for item in self.all_model_parameters]

        for key in self.avg_state_dict.keys():
            if self.avg_state_dict[key].dtype != torch.long:
                self.avg_state_dict[key] = sum(self.all_model_parameters[i]['weight'][key] * all_client_sizes[i]
                                               for i in range(num_models)) / sum(all_client_sizes)
            else:
                self.avg_state_dict[key] = sum(self.all_model_parameters[i]['weight'][key] * all_client_sizes[i]
                                               for i in range(num_models)) // sum(all_client_sizes)


def signal_handler(sig, frame):
    print("\nCatch stop signal Ctrl+C. Stop the program.")
    delete_old_queues()
    sys.exit(0)


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
    signal.signal(signal.SIGINT, signal_handler)
    server = Server()
    server.start()
    src.Log.print_with_color("Ok, ready!", "green")
