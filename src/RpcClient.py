import time
import pickle
import pika
import torch
import random

import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from pika.exceptions import AMQPConnectionError
from collections import defaultdict
from tqdm import tqdm

import src.Log
import src.Model
import src.Utils


class RpcClient:
    def __init__(self, client_id, address, username, password, train_func, device):
        self.model = None
        self.client_id = client_id
        self.address = address
        self.username = username
        self.password = password
        self.train_func = train_func
        self.device = device

        self.channel = None
        self.connection = None
        self.response = None

        self.train_set = None
        self.label_to_indices = None

        self.connect()

    def wait_response(self):
        status = True
        reply_queue_name = f'reply_{self.client_id}'
        self.channel.queue_declare(reply_queue_name, durable=False)
        while status:
            try:
                method_frame, header_frame, body = self.channel.basic_get(queue=reply_queue_name, auto_ack=True)
                if body:
                    status = self.response_message(body)
                time.sleep(0.5)
            except AMQPConnectionError as e:
                print(f"Connection failed, retrying in 5 seconds: {e}")
                self.connect()
                time.sleep(5)

    def response_message(self, body):
        self.response = pickle.loads(body)
        src.Log.print_with_color(f"[<<<] Client received: {self.response['message']}", "blue")
        action = self.response["action"]
        state_dict = self.response["parameters"]

        if action == "START":
            model_name = self.response["model_name"]
            if self.model is None:
                klass = getattr(src.Model, model_name)
                self.model = klass()
                self.model.to(self.device)

            # Read parameters and load to model
            if state_dict:
                self.model.load_state_dict(state_dict)

            data_name = self.response["data_name"]
            batch_size = self.response["batch_size"]
            lr = self.response["lr"]
            momentum = self.response["momentum"]
            clip_grad_norm = self.response["clip_grad_norm"]
            label_counts = self.response["label_counts"]
            epoch_round_cluster = self.response["epoch_round_cluster"]
            cluster = self.response["cluster"]
            src.Log.print_with_color(f"Label distribution of client: {label_counts.tolist()}", "yellow")

            if data_name and not self.train_set and not self.label_to_indices:
                if data_name == "MNIST":
                    transform_train = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,))
                    ])
                    self.train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                                                transform=transform_train)
                elif data_name == "CIFAR10":
                    transform_train = transforms.Compose([
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    ])
                    self.train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                                                  transform=transform_train)
                elif data_name == "DOMAIN":
                    self.train_set = src.Utils.load_dataset("domain_data/domain_train_dataset.pkl")

                else:
                    raise ValueError(f"Data name '{data_name}' is not valid.")

                self.label_to_indices = defaultdict(list)
                for idx, (_, label) in tqdm(enumerate(self.train_set)):
                    self.label_to_indices[int(label)].append(idx)

            selected_indices = []
            for label, count in enumerate(label_counts):
                selected_indices.extend(random.sample(self.label_to_indices[label], count))

            if data_name == "DOMAIN":
                subset = src.Utils.CustomDataset(self.train_set, selected_indices)
            else:
                subset = torch.utils.data.Subset(self.train_set, selected_indices)

            train_loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True)

            if data_name == "DOMAIN":
                criterion = nn.BCELoss()
            else:
                criterion = nn.CrossEntropyLoss()

            result = self.train_func(self.model, lr, momentum, train_loader, criterion, epoch_round_cluster, clip_grad_norm)

            # Stop training, then send parameters to server
            model_state_dict = self.model.state_dict()
            if self.device != "cpu":
                for key in model_state_dict:
                    model_state_dict[key] = model_state_dict[key].to('cpu')
            if cluster:
                data = {"action": "UPDATE", "client_id": self.client_id, "result": result, "size": sum(label_counts),
                        "message": "Sent parameters to Server", "parameters": model_state_dict,"cluster": True}
                src.Log.print_with_color("[>>>] Client sent parameters to server", "red")
                self.send_to_server(data)
                return True
            else:
                data = {"action": "UPDATE", "client_id": self.client_id, "result": result, "size": sum(label_counts),
                        "message": "Sent parameters to Server", "parameters": model_state_dict,"cluster": False}
                src.Log.print_with_color("[>>>] Client sent parameters to server", "red")
                self.send_to_server(data)
                return True

        elif action == "STOP":
            return False

    def connect(self):
        credentials = pika.PlainCredentials(self.username, self.password)
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(self.address, 5672, '/', credentials))
        self.channel = self.connection.channel()

    def send_to_server(self, message):
        self.connect()
        self.response = None

        self.channel.queue_declare('rpc_queue', durable=False)
        self.channel.basic_publish(exchange='',
                                   routing_key='rpc_queue',
                                   body=pickle.dumps(message))

        return self.response
