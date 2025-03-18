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
from torch.utils.data import ConcatDataset, Subset, DataLoader
from src.Model import *
import src.Log
import src.Model
import src.Utils
import numpy as np
#from src.PositionalEncodingTransformer import PositionalEncodingTransformer


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
        self.model_ae = None
        self.subset = None
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
        data_for_cluster = self.response["data_cluster"]
        print(data_for_cluster)
        data_name = self.response["data_name"]
        if data_name == "DOMAIN2" and data_for_cluster == "VAE" and self.model_ae == None:
            self.model_ae = DomainVAE(vocab_size=vocab_size, embed_dim=16, hidden_dim=64, latent_dim=32, dropout_p=0.2)
            self.model_ae.to(self.device)
            self.model_ae.load_state_dict(torch.load("model_ae_3.pth", map_location=self.device))
        if action == "START":
            model_name = self.response["model_name"]
            if self.model is None:
                klass = getattr(src.Model, model_name)
                self.model = klass()
                self.model.to(self.device)

            # Read parameters and load to model
            if state_dict:
                self.model.load_state_dict(state_dict)

            
            batch_size = self.response["batch_size"]
            lr = self.response["lr"]
            momentum = self.response["momentum"]
            clip_grad_norm = self.response["clip_grad_norm"]
            label_counts = self.response["label_counts"]
            epoch = self.response["epoch"]
            cluster = self.response['cluster']
            dga_index = self.response["dga_index"]
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
                elif data_name == "DOMAIN2":
                    if data_for_cluster != "VAE":
                        benign_train_ds = src.Utils.load_dataset("domain2/benign_train.pkl")
                        dga_1_train_ds = src.Utils.load_dataset("domain2/dga_1_train.pkl")
                        dga_2_train_ds = src.Utils.load_dataset("domain2/dga_2_train.pkl")
                        dga_3_train_ds = src.Utils.load_dataset("domain2/dga_3_train.pkl")
                        dga_4_train_ds = src.Utils.load_dataset("domain2/dga_4_train.pkl")
                        all_train_datasets = [ConcatDataset([benign_train_ds, dga_1_train_ds]),
                            ConcatDataset([benign_train_ds, dga_2_train_ds]),
                            ConcatDataset([benign_train_ds, dga_3_train_ds]),
                            ConcatDataset([benign_train_ds, dga_4_train_ds])]
                        self.train_set = all_train_datasets[dga_index]
                    
                else:
                    raise ValueError(f"Data name '{data_name}' is not valid.")
                if data_for_cluster != "VAE":
                    self.label_to_indices = defaultdict(list)
                    for idx, (_, label) in tqdm(enumerate(self.train_set)):
                        self.label_to_indices[int(label)].append(idx)
            if data_for_cluster != "VAE":    
                selected_indices = []
                for label, count in enumerate(label_counts):
                    selected_indices.extend(random.sample(self.label_to_indices[label], count))
            if data_name == "DOMAIN":
                subset = src.Utils.CustomDataset(self.train_set, selected_indices)
            elif data_name == "DOMAIN2":
                if data_for_cluster != "VAE":
                    subset =src.Utils.CustomDataset(self.train_set, selected_indices)
                else:
                    subset = self.subset
            else:
                subset = torch.utils.data.Subset(self.train_set, selected_indices)

            train_loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True)

            if data_name == "DOMAIN" or data_name == "DOMAIN2":
                criterion = nn.BCELoss()
            else:
                criterion = nn.CrossEntropyLoss()

            result = self.train_func(self.model, lr, momentum, train_loader, criterion,epoch , clip_grad_norm)

            # Stop training, then send parameters to server
            model_state_dict = self.model.state_dict()
            if self.device != "cpu":
                for key in model_state_dict:
                    model_state_dict[key] = model_state_dict[key].to('cpu')
            if cluster == True:
                data = {"action": "UPDATE", "client_id": self.client_id, "result": result, "size": sum(label_counts),
                        "message": "Sent parameters to Server", "parameters": model_state_dict, "cluster": True}
            elif cluster == False:
                data = {"action": "UPDATE", "client_id": self.client_id, "result": result, "size": sum(label_counts),
                        "message": "Sent parameters to Server", "parameters": model_state_dict, "cluster": False}
            else :
                data = {"action": "UPDATE", "client_id": self.client_id, "result": result, "size": sum(label_counts),
                        "message": "Sent parameters to Server", "parameters": model_state_dict, "cluster": None}
            src.Log.print_with_color("[>>>] Client sent parameters to server", "red")
            self.send_to_server(data)
            return True
        elif action == "INFOR":
            if self.train_set is None:
                self.data_name = self.response["data_name"]
                self.label_counts = self.response["label_counts"]
                self.dga_label = self.response["dga_label"]
                self.data_cluster = self.response["data_cluster"]
                benign_train_ds = src.Utils.load_dataset("domain2/benign_train.pkl")
                dga_1_train_ds = src.Utils.load_dataset("domain2/dga_1_train.pkl")
                dga_2_train_ds = src.Utils.load_dataset("domain2/dga_2_train.pkl")
                dga_3_train_ds = src.Utils.load_dataset("domain2/dga_3_train.pkl")
                dga_4_train_ds = src.Utils.load_dataset("domain2/dga_4_train.pkl")
                all_train_datasets = [ConcatDataset([benign_train_ds, dga_1_train_ds]),
                        ConcatDataset([benign_train_ds, dga_2_train_ds]),
                        ConcatDataset([benign_train_ds, dga_3_train_ds]),
                        ConcatDataset([benign_train_ds, dga_4_train_ds])]
                self.train_set = all_train_datasets[self.dga_label]
                self.label_to_indices = defaultdict(list)
                for idx, (_, label) in tqdm(enumerate(self.train_set)):
                    self.label_to_indices[int(label)].append(idx)
                selected_indices = []
                for label, count in enumerate(self.label_counts):
                    selected_indices.extend(random.sample(self.label_to_indices[label], count))
                self.subset = src.Utils.CustomDataset(self.train_set,selected_indices)
            self.all_features = []
            dataloader = DataLoader(self.subset, batch_size=32, shuffle=False)
            features, _ = src.Utils.extract_latent_features(self.model_ae, dataloader, self.device)
            self.all_features.append(features.mean(axis=0))  # Lấy trung bình latent vector của client
            self.all_features = np.array(self.all_features).flatten()
            data = {"action": "UPDATE-INFO", "client_id": self.client_id, "message": "Sent latent features to Server", "features": self.all_features, "dga_label": self.dga_label}
            self.all_features = []
            src.Log.print_with_color("[>>>] Client sent latents feature to server", "red")
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
