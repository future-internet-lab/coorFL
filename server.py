import os
import pika
import pickle
import argparse
import sys
import torch.utils
import torchvision
import yaml
import signal
import collections
import torch
import requests
import random
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import src.Utils
import src.Validation
import src.Log
from src.Selection import client_selection_speed_base, client_selection_random
from src.Cluster import clustering_algorithm
from src.Utils import DomainDataset, generate_random_array
from src.Notify import send_mail
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
import torchvision.transforms as transforms
from requests.auth import HTTPBasicAuth
from src.Model import *

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

#interference
epoch_round_cluster = config['server']['interference-each-client']['epoch-round-cluster']
sample_foward_propagation= config['server']['interference-each-client']['sample-foward-propagation']
data_for_cluster = config['server']['data-for-cluster']
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
        self.model_foward = None
        self.interference_output = None
        self.cluster = True
        self.count_cluster = False
        self.all_model_parameters_temp = []
        self.neural_last_layer = []
        self.all_model_paremeters_all_client = []
        self.labels = None
        self.num_clusters =None
        self.train_set = None
        self.dga_label = []
        self.label_to_indices_list = []
        self.all_features = []
        if not refresh_each_round:
            if data_name == "DOMAIN":
                self.non_iid_label = [np.insert(src.Utils.non_iid_rate(num_labels - 1, non_iid_rate), 0, 1) for _ in range(self.total_clients)]
            elif data_name == "DOMAIN2":
                self.dga_label = src.Utils.dga_label(self.total_clients)
            else:
                
                self.non_iid_label = [src.Utils.non_iid_rate(num_labels, non_iid_rate) for _ in range(self.total_clients)]

        # self.speeds = [325, 788, 857, 915, 727, 270, 340, 219, 725, 228, 677, 259, 945, 433, 222, 979, 339, 864, 858, 621, 242, 790, 807, 368, 259, 776, 218, 845, 294, 340, 731, 595, 799, 524, 779, 581, 456, 574, 754, 771]
        #self.speeds = [25, 20, 77, 33, 74, 25, 77, 54, 39, 88, 36, 76, 34, 37, 84, 85, 80, 28, 44, 20, 87, 57, 86, 43,
        #             90, 58, 23, 41, 35, 41, 21, 60, 92, 81, 37, 30, 85, 79, 84, 22]
        if model_name == "PositionalEncodingTransformer":
            self.speeds = [random.randrange(224, 1792) for _ in range(total_clients)]
        elif model_name == "CNNClassifier":
            self.speeds = [random.randrange(702, 5616) for _ in range(total_clients)]
        elif model_name == "BiLSTMClassifier":
            self.speeds = [random.randrange(1170, 9360) for _ in range(total_clients)]
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
        elif data_name == "DOMAIN2": 
            if data_mode == "even":
                if refresh_each_round:
                  self.dga_label = src.Utils.dga_label(self.total_clients)
                self.label_counts = np.array([[15000// total_clients] + [15000 // total_clients for _ in range(num_labels-1)]
                                     for _ in range(total_clients)])
                self.logger.log_info(f'DGA_Label: {self.dga_label}')
                print(f"DGA label: {self.dga_label} ")
            else:
                if refresh_each_round:
                  self.dga_label = src.Utils.dga_label(self.total_clients)
                self.label_counts = np.random.randint(250, 750, size=(total_clients, num_labels))
                self.logger.log_info(f'DGA_Label: {self.dga_label}')
                print(f"DGA label: {self.dga_label} ")
            if data_for_cluster == "VAE":
                for i in range(total_clients):
                    client_id = self.list_clients[i]
                    src.Log.print_with_color(f"[>>>] Sent DGA label {self.dga_label[i]} and label count {self.label_counts[i]} to client {client_id} ", "red")
                    response = {"action": "INFOR",
                                "message": "DGA label and label count",
                                "parameters": None,
                                "model_name": model_name,
                                "data_name": data_name,
                                "dga_label":self.dga_label[i],
                                "label_counts":self.label_counts[i],
                                "data_cluster":data_for_cluster,
                            }
                    self.send_to_response(client_id, pickle.dumps(response))
                self.list_clients = []

        else:
            if data_mode == "even":
                self.label_counts = np.array([[5000 // total_clients for _ in range(num_labels)] for _ in range(total_clients)])
            else:
                if refresh_each_round:
                    self.non_iid_label = [src.Utils.non_iid_rate(num_labels, non_iid_rate) for _ in range(self.total_clients)]
                self.label_counts = [np.array([random.randint(data_range[0]//non_iid_rate, data_range[1]//non_iid_rate)
                                              for _ in range(num_labels)]) *
                                     self.non_iid_label[i] for i in range(total_clients)]
    
    def interference_each_client(self):
        if data_name == "MNIST":
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            trainset_foward = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
        elif data_name == "CIFAR10":
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            trainset_foward = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        elif data_name == "DOMAIN":
            trainset_foward = src.Utils.load_dataset("domain_data/domain_test_dataset.pkl")
            trainset_foward = src.Utils.modify_labels(trainset_foward)
        elif data_name == "DOMAIN2":
            benign_train_ds = src.Utils.load_dataset("domain2/benign_train.pkl")
            dga_1_train_ds = src.Utils.load_dataset("domain2/dga_1_train.pkl")
            dga_2_train_ds = src.Utils.load_dataset("domain2/dga_2_train.pkl")
            dga_3_train_ds = src.Utils.load_dataset("domain2/dga_3_train.pkl")
            dga_4_train_ds = src.Utils.load_dataset("domain2/dga_4_train.pkl")

            self.train_set = [ConcatDataset([benign_train_ds, dga_1_train_ds]),
                                ConcatDataset([benign_train_ds, dga_2_train_ds]),
                                ConcatDataset([benign_train_ds, dga_3_train_ds]),
                                ConcatDataset([benign_train_ds, dga_4_train_ds])]
        else:
            raise ValueError(f"Do not have data name '{data_name}.")
        if data_name != "DOMAIN2":
            for i in self.all_model_parameters_temp:
                client_id = i['client_id']
                model_state_dict_1 = i['weight']
                indices = list(range(sample_foward_propagation))  
                subset = Subset(trainset_foward, indices)
                trainloader = DataLoader(subset, batch_size=sample_foward_propagation, shuffle=False)

                if self.model_foward is None:
                        klass = getattr(src.Model, model_name)
                        self.model_foward = klass()
                        self.model_foward.to(device)
                self.model_foward.eval()
                self.model_foward.load_state_dict(model_state_dict_1)
                for inputs, labels in trainloader:
                    inputs = inputs.to(device)
                    with torch.no_grad():  
                        outputs = self.model_foward(inputs)
                    
                        outputs_list = outputs.tolist()  
                    
                        output_array = np.array(outputs_list).flatten()
                        
                        self.neural_last_layer.append({"client_id": client_id, "output": output_array, "cluster_index": None})

                    break   
            self.interference_output = np.array([item["output"] for item in self.neural_last_layer])
            return self.interference_output    
        else:  
            for i in self.all_model_parameters_temp:
                client_id = i['client_id']
                model_state_dict_1 = i['weight']
                
                # Assuming self.train_set already contains your DOMAIN2 data
                dataset = self.train_set[0]  # Fetch appropriate dataset for the client
                indices = list(range(sample_foward_propagation))  # Define sample size for forward propagation
                subset = Subset(dataset, indices)  # Use Subset to create a smaller dataset for the client
                
                trainloader = DataLoader(subset, batch_size=sample_foward_propagation, shuffle=False)

                if self.model_foward is None:
                    klass = getattr(src.Model, model_name)
                    self.model_foward = klass()
                    self.model_foward.to(device)

                self.model_foward.eval()
                self.model_foward.load_state_dict(model_state_dict_1)
                
                for inputs, labels in trainloader:
                    inputs = inputs.to(device)
                    with torch.no_grad():
                        outputs = self.model_foward(inputs)
                        outputs_list = outputs.tolist()
                        output_array = np.array(outputs_list).flatten()

                        self.neural_last_layer.append({"client_id": client_id, "output": output_array, "cluster_index": None})

                    break  # Only process the first batch

            # After processing all clients, compile the outputs into a interference_output array
            self.interference_output = np.array([item["output"] for item in self.neural_last_layer])
            return self.interference_output    
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

        if data_for_cluster == 'data-distribution':
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
                                                    'size': client_size,'cluter_index':None})

                # If consumed all client's parameters
                if self.updated_clients == len(self.selected_client):
                    self.process_consumer()
        #FLIS
        elif data_for_cluster == 'interference-each-client':  
            if action == "REGISTER":
                if str(client_id) not in self.list_clients:
                    self.list_clients.append(str(client_id))
                    src.Log.print_with_color(f"[<<<] Received message from client: {message}", "blue")

                # If consumed all clients - Register for first time
                if len(self.list_clients) == self.total_clients:
                    src.Log.print_with_color("All clients are connected. Sending notifications.", "green")
                    self.data_distribution()
                    if self.round < self.num_round:
                        self.client_selection()
                    src.Log.print_with_color(f"Start training round {self.num_round - self.round + 1}", "yellow")
                    self.notify_clients()
            elif action == "UPDATE":
                data_message = message["message"]
                result = message["result"]
                cluster = message['cluster']
                src.Log.print_with_color(f"[<<<] Received message from client: {data_message}", "blue")
                self.updated_clients += 1
                # Save client's model parameters
                if cluster: 
                    if not result:
                        self.round_result = False

                    if save_parameters and self.round_result:
                        model_state_dict = message["parameters"]
                        client_size = message["size"]
                        self.all_model_parameters.append({'client_id': client_id, 'weight': model_state_dict,
                                                        'size': client_size,'cluster_index':None})

                    # If consumed all client's parameters
                    if self.updated_clients == self.total_clients:
                        self.process_consumer()
                else:
                    if not result:
                        self.round_result = False

                    if save_parameters and self.round_result:
                        model_state_dict = message["parameters"]
                        client_size = message["size"]
                        cluster_index = None
                        for client in self.all_model_parameters_temp:
                            if client['client_id'] == client_id:
                                cluster_index = client.get('cluster_index', None)  # Lấy cluster_index nếu có
                                break

                        self.all_model_parameters.append({'client_id': client_id, 'weight': model_state_dict,
                                                        'size': client_size,'cluster_index': cluster_index})
                    # If consumed all client's parameters
                    if self.updated_clients == len(self.selected_client):
                        self.process_consumer()
        else:
            if action == "REGISTER":
                if str(client_id) not in self.list_clients:
                    self.list_clients.append(str(client_id))
                    src.Log.print_with_color(f"[<<<] Received message from client: {message}", "blue")

                # If consumed all clients - Register for first time
                if len(self.list_clients) == self.total_clients:
                    self.data_distribution()
                    src.Log.print_with_color("All clients are connected. Sending notifications.", "green")
                    src.Log.print_with_color(f"Start training round {self.num_round - self.round + 1}", "yellow")
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
                                                    'size': client_size,'cluter_index':None})

                # If consumed all client's parameters
                if self.updated_clients == len(self.selected_client):
                    self.process_consumer()
            elif action == "UPDATE-INFO" :
                features = message["features"]
                client_id = message["client_id"]
                dga_label = message["dga_label"]
                src.Log.print_with_color(f"[<<<] Received update-information from client {client_id} with DGA label {dga_label}", "blue")
                self.all_features.append(features)
                self.list_clients.append(str(client_id))
                if len(self.list_clients) == self.total_clients:
                    self.client_selection()
                    self.notify_clients()




        # Ack the message
        ch.basic_ack(delivery_tag=method.delivery_tag)

    def process_consumer(self):
        """
        After collect all training clients, start validation and make decision for the next training round
        :return:
        """
        if data_for_cluster == "data-distribution":
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
        #FLIS
        elif data_for_cluster == "interference-each-client":
            if self.cluster:
                self.updated_clients = 0
                src.Log.print_with_color("Collected all parameters.", "yellow")
                # TODO: detect model poisoning with self.all_model_parameters at here
                if save_parameters and self.round_result:
                    #self.all_model_paremeters_all_client = self.all_model_parameters
                    self.all_model_parameters_temp = self.all_model_parameters
                    self.all_model_parameters = []
                    self.cluster = False
                # Server validation

                if not self.round_result:
                    src.Log.print_with_color(f"Training failed!", "yellow")
                    send_mail(email_config, f"Quá trình training bị lỗi tại round {self.num_round - self.round + 1}")
                    if stop_when_false:
                        # Stop training
                        self.notify_clients(start=False)
                        delete_old_queues()
                        sys.exit()
                else:
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
            else:
                self.updated_clients = 0
                src.Log.print_with_color("Collected all parameters.", "yellow")
                # TODO: detect model poisoning with self.all_model_parameters at here
                if save_parameters and self.round_result:
                    
                    self.avg_parameter_each_cluster()
                    cluster_weights = {}

                # Server validation
                total_accuracy = 0.0
                if save_parameters and validation and self.round_result:
                    a = self.get_weights_for_each_cluster()
                    for cluster_index, cluster_weights in a.items():
                        print(f"Accuracy of {cluster_index}")
                        state_dict = cluster_weights
                        self.round_result, accuracy = self.validation.test(state_dict, device)
                        total_accuracy += accuracy
                    total_accuracy /= len(a)  # Tính độ chính xác trung bình của các cluster
                    print(f"Accuracy: {total_accuracy}")
                self.all_model_parameters = []
                if not self.round_result:
                    src.Log.print_with_color(f"Training failed!", "yellow")
                    send_mail(email_config, f"Quá trình training bị lỗi tại round {self.num_round - self.round + 1}")
                    if stop_when_false:
                        # Dừng huấn luyện
                        self.notify_clients(start=False)
                        delete_old_queues()
                        sys.exit()
                elif self.last_accuracy - total_accuracy > accuracy_drop:
                    src.Log.print_with_color(f"Accuracy drop!", "yellow")
                else:
                    self.last_accuracy = total_accuracy
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
        #VAE
        else:
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
            if data_for_cluster == 'data-distribution':
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
                    print(self.label_counts[i])
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
                                "clip_grad_norm": clip_grad_norm,
                                "epoch":1,
                                "cluster": None
                               }
                    count_labels += self.label_counts[i]
                    self.send_to_response(client_id, pickle.dumps(response))

                    self.logger.log_info(f"All training labels count = {count_labels.tolist()}")
            #FLIS
            elif data_for_cluster == 'interference-each-client':
                if self.cluster == True:
                    filepath = f'{model_name}.pth'
                # Read parameters file
                    state_dict = None
                    if load_parameters:
                        if os.path.exists(filepath):
                            state_dict = torch.load(filepath, weights_only=True)
                    count_labels = np.zeros(num_labels)
                    for i in range(self.total_clients):
                        client_id = self.list_clients[i]
                        # Request clients to start training
                        count_labels = np.zeros(num_labels)
                        src.Log.print_with_color(f"[>>>] Sent start training round request to client {client_id}", "red")
                        response = {"action": "START",
                                    "message": "Server accept the connection!",
                                    "model_name": model_name,
                                    "data_name": data_name,
                                    "parameters": state_dict,
                                    "label_counts": self.label_counts[i],
                                    "batch_size": batch_size,
                                    "lr": lr,
                                    "momentum": momentum,
                                    "clip_grad_norm": clip_grad_norm,
                                    "epoch":epoch_round_cluster,
                                    "cluster": self.cluster,
                                    "dga_index":self.dga_label[i],
                                    "data_cluster": data_for_cluster}
                        count_labels += self.label_counts[i]
                        self.send_to_response(client_id, pickle.dumps(response))
                if self.cluster == False:
                    for i in self.selected_client:
                        #client_id = self.list_clients[i]
                        client_id = self.all_model_parameters_temp[i]["client_id"]
                        model_state_dict = None
                        cluster_index1 = None
                        for client in self.all_model_parameters_temp:
                            if client['client_id'] == client_id:
                                model_state_dict = client['weight']
                                cluster_index1 = client['cluster_index']
                                break 
                        # Request clients to start training
                        count_labels = np.zeros(num_labels)
                        src.Log.print_with_color(f"[>>>] Sent start training request to client {client_id}, cluster index {cluster_index1}", "red")
                        response = {"action": "START",
                                    "message": "Server accept the connection!",
                                    "model_name": model_name,
                                    "data_name": data_name,
                                    "parameters": model_state_dict,
                                    "label_counts": self.label_counts[i],
                                    "batch_size": batch_size,
                                    "lr": lr,
                                    "momentum": momentum,
                                    "clip_grad_norm": clip_grad_norm,
                                    "epoch":1,
                                    "cluster": self.cluster,
                                    "dga_index":self.dga_label[i],
                                    "data_cluster": data_for_cluster}
                        count_labels += self.label_counts[i]
                        self.send_to_response(client_id, pickle.dumps(response))

                self.logger.log_info(f"All training labels count = {count_labels.tolist()}")
            #VAE
            else:
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
                    print(self.label_counts[i])
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
                                "clip_grad_norm": clip_grad_norm,
                                "epoch":1,
                                "cluster": None,
                                "dga_index": None,
                                "data_cluster": "VAE"
                                }
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

        if data_for_cluster == 'data-distribution':
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
        #FLIS
        elif data_for_cluster == 'interference-each-client':
            if client_selection_config['enable']:
                if client_cluster_config['enable']:
                    if self.round == self.num_round - 1 :
                        self.num_cluster, self.labels, _ = clustering_algorithm(self.interference_each_client(), client_cluster_config)
                        for i, client_data in enumerate(self.neural_last_layer):
                            client_data['cluster_index'] = self.labels[i]
                        for i, client_data in enumerate(self.all_model_parameters_temp):
                            client_data['cluster_index'] = self.neural_last_layer[i]['cluster_index']
                    print(f"Labels: {self.labels}")
                    print(f"The number of clutser: {self.num_cluster}")
                    self.logger.log_info(f"Num cluster = {self.num_cluster}, labels = {self.labels}")
                    self.selected_client = []
                    for i in range(self.num_cluster):
                        cluster_client = [index for index, label in enumerate(self.labels) if label == i]
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
                if self.round == self.num_round - 1 :
                    self.num_cluster, self.labels, _ = clustering_algorithm(self.interference_each_client(), client_cluster_config)
                    for i, client_data in enumerate(self.neural_last_layer):
                        client_data['cluster_index'] = self.labels[i]
                    for i, client_data in enumerate(self.all_model_parameters_temp):
                        client_data['cluster_index'] = self.neural_last_layer[i]['cluster_index']
                self.selected_client = []
                self.selected_client = [i for i in range(len(self.list_clients))]
                

            # From client selected, calculate and log training time
            training_time = np.max([total_training_time[i] for i in self.selected_client])
            self.logger.log_info(f"Active with {len(self.selected_client)} client: {self.selected_client}")
            self.logger.log_info(f"Total training time round = {training_time}")
        #VAE
        else:
            if client_selection_config['enable']:
                if client_cluster_config['enable']:
                    num_cluster, labels, _ = clustering_algorithm(self.all_features, client_cluster_config)
                    self.all_features = []
                    print(f"Labels: {labels}")
                    print(f"The number of clutser: {num_cluster}")
                    self.logger.log_info(f"Num cluster = {num_cluster}, labels = {labels}")
                    self.selected_client = []
                    self.client_indices = []
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
    def get_weights_for_each_cluster(self):
     
        clusters = {}

        for client in self.all_model_parameters_temp:
            cluster_index = client['cluster_index']
            model_state_dict = client['weight']

            # Nếu cluster_index chưa có trong danh sách, lưu client này làm đại diện
            if cluster_index not in clusters:
                clusters[cluster_index] = model_state_dict 

        return clusters



    def avg_parameter_each_cluster(self):
        clusters = {}

        # Gom nhóm client theo cluster_index
        for client in self.all_model_parameters:
            print(client["cluster_index"])
            if 'cluster_index' not in client:
                raise KeyError(f"Client {client.get('client_id', 'Unknown ID')} does not have 'cluster_index'. Please ensure it is set correctly.")

            cluster_index = client['cluster_index']
            if cluster_index not in clusters:
                clusters[cluster_index] = []
            clusters[cluster_index].append(client)


        # Tính toán trung bình cho từng cụm
        for cluster_index, clients_in_cluster in clusters.items():

            # Khởi tạo lại all_client_sizes mỗi vòng
            all_client_sizes = [client['size'] for client in clients_in_cluster]

            total_size = sum(all_client_sizes)
            if total_size == 0:
                total_size = 1  # Tránh chia 0

            # Kiểm tra kiểu dữ liệu của weight
            first_weight = clients_in_cluster[0]['weight']
            if isinstance(first_weight, torch.Tensor):
                avg_state_dict = torch.zeros_like(first_weight)
            elif isinstance(first_weight, dict) or isinstance(first_weight, collections.OrderedDict):
                avg_state_dict = {key: torch.zeros_like(val) for key, val in first_weight.items()}
            else:
                raise TypeError("Unknown type for weight in clients_in_cluster")

            # Tính tổng trọng số theo kích thước client
            for client in clients_in_cluster:
                model_state_dict = client['weight']
                if isinstance(model_state_dict, dict) or isinstance(model_state_dict, collections.OrderedDict):
                    for key in avg_state_dict.keys():
                        avg_state_dict[key] += model_state_dict[key] * client['size']
                else:
                    avg_state_dict += model_state_dict * client['size']

            # Chia để tính trung bình
            for key in avg_state_dict.keys():
                avg_state_dict[key] = avg_state_dict[key].float() / total_size

            # Cập nhật lại trọng số cho từng client trong cụm
            for client in clients_in_cluster:
                client['weight'] = {key: avg_state_dict[key].clone() for key in avg_state_dict.keys()}

        # Cập nhật trọng số cho các client còn lại trong cụm
        cluster_weights = {}
        for client in self.all_model_parameters:
            cluster_index = client.get('cluster_index')  
            if cluster_index is not None:  
                cluster_weights[cluster_index] = client['weight']  

        for client in self.all_model_parameters_temp:
            cluster_index = client.get('cluster_index')  
            if cluster_index in cluster_weights:
                client['weight'] = cluster_weights[cluster_index]
                        
        return self.all_model_parameters


    
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
