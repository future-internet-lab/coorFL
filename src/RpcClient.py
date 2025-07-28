import time
import pickle
import pika
import torch
import random

import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from pika.exceptions import AMQPConnectionError
from collections import defaultdict
from tqdm import tqdm
from torch.utils.data import ConcatDataset
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
import src.Log
import src.Model
import src.Utils


class RpcClient:
    def __init__(self, client_id, address, username, password, train_func, device, zone):
        self.model = None
        self.client_id = client_id
        self.address = address
        self.username = username
        self.password = password
        self.train_func = train_func
        self.device = device
        self.data_zone = zone

        self.channel = None
        self.connection = None
        self.response = None
        self.data_range = [0,0]
        self.selected_flag = False
        self.train_set = None
        self.label_to_indices = None
        self.voi_estimator = src.Utils.PPOVoIEstimator(state_dim=4)
        self.staleness = 0
        self.age = 0
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
        

        if action == "START":
            state_dict = self.response["parameters"]
            model_name = self.response["model_name"]
            self.data_name = self.response["data_name"]
            if self.data_name == "CIFAR10" or self.data_name == "MNIST":
                num_labels = 10
            elif self.data_name == "DOMAIN":
                num_labels = 21
            elif self.data_name == "DOMAIN2":
                num_labels = 2
            else:
                num_labels = 0
            batch_size = self.response["batch_size"]
            self.lr = self.response["lr"]
            self.momentum = self.response["momentum"]
            self.clip_grad_norm = self.response["clip_grad_norm"]
            self.data_range[0] = self.response["range[0]"]
            self.data_range[1] = self.response["range[1]"]
            self.algorithm = self.response["algorithm_name"]
            self.speed = self.response["speed"]
            self.training_time = None
            if self.algorithm != "fedcls":
                self.label_counts = np.array([random.randint(self.data_range[0], self.data_range[1]) for _ in range(num_labels)])
            self.num_data = np.sum(self.label_counts)
            data_mode = self.response["data_mode"]
            self.epoch = self.response["epoch"]

            if self.model is None:
                if self.data_name == "DOMAIN":
                    if model_name == "CNN":
                        self.model = src.Model.CNNClassifier()
                    elif model_name == "Transformer":
                        self.model = src.Model.PositionalEncodingTransformer()
                    else:
                        raise ValueError(f"[ERROR] Unknown model '{model_name}' for data 'DOMAIN'")
                
                elif self.data_name == "CIFAR10":
                    if model_name == "ResNet":
                        self.model = src.Model.ResNet18()
                    elif model_name == "MobileNet":
                        self.model = src.Model.MobileNetV2()
                    else:
                        raise ValueError(f"[ERROR] Unknown model '{model_name}' for data 'CIFAR10'")
                
                elif self.data_name == "DOMAIN2":
                    if model_name == "CNN":
                        self.model = src.Model.CNNClassifier()
                    elif model_name == "Transformer":
                        self.model = src.Model.PositionalEncodingTransformer()
                    else:
                        raise ValueError(f"[ERROR] Unknown model '{model_name}' for data 'DOMAIN2'")

                else:
                    raise ValueError(f"[ERROR] Unknown data_name: '{self.data_name}'")

                self.model.to(self.device)


            # Read parameters and load to model
            if state_dict:
                self.model.load_state_dict(state_dict)

            
            src.Log.print_with_color(f"Label distribution of client: {self.label_counts.tolist()}", "yellow")
            src.Log.print_with_color(f"Data zone: {self.data_zone}", "yellow")
            src.Log.print_with_color(f"Epoch: {self.epoch}", "yellow")
            

            if self.data_name and not self.train_set and not self.label_to_indices:
                src.Log.print_with_color(f"Data name: {self.data_name}", "yellow")
                if self.data_name == "MNIST":
                    transform_train = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,))
                    ])
                    self.train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                                                transform=transform_train)
                elif self.data_name == "CIFAR10":
                    train_transform = transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.4914, 0.4822, 0.4465],
                            std=[0.2023, 0.1994, 0.2010]
                        ),
                    ])
                    if data_mode == "uneven":
                        print("UNEVEN MODE")
                        zone_spec = {
                            "zoneA": [0, 1, 2, 3],
                            "zoneB": [2, 3, 4, 5],
                            "zoneC": [4, 5, 6, 7],
                            "zoneD": [6, 7, 8, 9],
                            "zoneE": [8, 9, 0, 1],
                        }
                        train_zones = src.Utils.split_cifar10_to_zones(
                            root="data",
                            zone_spec=zone_spec,
                            transform_train=train_transform
                        )

                        self.all_train_set = [train_zones["zoneA"], train_zones["zoneB"], train_zones["zoneC"],
                                            train_zones["zoneD"], train_zones["zoneE"]]
                    else:
                        print("EVEN MODE")
                        zone_spec = {
                            "all": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                        }
                        train_zones = src.Utils.split_cifar10_to_zones(
                            root="data",
                            zone_spec=zone_spec,
                            transform_train=train_transform
                        )

                        self.all_train_set = [train_zones["all"]]
                elif self.data_name == "DOMAIN":
                    self.train_set = src.Utils.load_dataset("domain_data/domain_train_dataset.pkl")
                elif self.data_name == "DOMAIN2":
                    
                    benign_train_ds = src.Utils.load_dataset("domain2/benign_train.pkl")
                    dga_1_train_ds = src.Utils.load_dataset("domain2/dga_1_train.pkl")
                    dga_2_train_ds = src.Utils.load_dataset("domain2/dga_2_train.pkl")
                    dga_3_train_ds = src.Utils.load_dataset("domain2/dga_3_train.pkl")
                    dga_4_train_ds = src.Utils.load_dataset("domain2/dga_4_train.pkl")
                    if data_mode == "uneven":
                        print("UNEVEN MODE")
                        self.all_train_set = [ConcatDataset([benign_train_ds, dga_1_train_ds]),
                                            ConcatDataset([benign_train_ds, dga_2_train_ds]),
                                            ConcatDataset([benign_train_ds, dga_3_train_ds]),
                                            ConcatDataset([benign_train_ds, dga_4_train_ds])]
                    else:
                        print("EVEN MODE")
                        self.all_train_set = [ConcatDataset([benign_train_ds, dga_1_train_ds, dga_2_train_ds, dga_3_train_ds, dga_4_train_ds])]
                else:
                    raise ValueError(f"Data name '{self.data_name}' is not valid.")
                if self.algorithm == "fedcls": 
                    data = {"action": "fedcls", "client_id": self.client_id, "label_count": self.label_counts, "num_label": num_labels}
                    
                self.all_class_indices = []
                for train_ds in self.all_train_set:
                    self.class_indices = {label: [] for label in set(sample[1] for sample in train_ds)}
                    for idx, (_, label) in enumerate(train_ds):
                        self.class_indices[label].append(idx)
                    self.all_class_indices.append(self.class_indices)
                   
            self.train_set = self.all_train_set[self.data_zone]
            self.class_indices = self.all_class_indices[self.data_zone]
            selected_indices = []
            self.client_sizes = 0
            for label, num_samples in zip(self.class_indices.keys(), self.label_counts):
                    available_indices = self.class_indices[label]
                    actual_sample_size = min(num_samples, len(available_indices))  # giới hạn lại
                    self.client_sizes += actual_sample_size
                    print(self.client_sizes)
                    selected_indices.extend(random.sample(available_indices, actual_sample_size))
            
            src.Log.print_with_color(f"Chuan bi ghep subset", "yellow")
            if self.data_name == "DOMAIN" or self.data_name == "DOMAIN2":
                self.subset = src.Utils.CustomDataset(self.train_set, selected_indices)
            else:
                self.subset = Subset(self.train_set, selected_indices)
            labels_in_subset = [self.train_set[idx][1] for idx in selected_indices]
            unique_labels = set(labels_in_subset)
            self.train_loader = torch.utils.data.DataLoader(self.subset, batch_size=batch_size, shuffle=True)
            
            if self.algorithm != "fedrhlp" and self.algorithm != "hicsfl" and self.algorithm != "feel": 
                
                criterion = nn.CrossEntropyLoss()
            
                src.Log.print_with_color(f"Strat trainning: {self.data_name}", "yellow")
                result = self.train_func(self.model, self.lr,self.data_name, self.momentum, self.train_loader, criterion, self.clip_grad_norm,self.epoch)
                self.training_time = self.epoch * self.client_sizes / self.speed
                self.train_loader = None
        
                model_state_dict = self.model.state_dict()
                if self.device != "cpu":
                    for key in model_state_dict:
                        model_state_dict[key] = model_state_dict[key].to('cpu')

                data = {"action": "UPDATE", "client_id": self.client_id, "result": result, "size": sum(self.label_counts),
                        "message": "Sent parameters to Server", "parameters": model_state_dict, "num_data": self.client_sizes, "training_time": self.training_time, "speed": self.speed}
                src.Log.print_with_color(f"Data gui di: {self.client_id},{self.client_sizes}", "yellow")
                src.Log.print_with_color("[>>>] Client sent parameters to server", "red")
                self.send_to_server(data)
                return True
            elif self.algorithm == "fedrhlp":
                training_data = []
                training_data.append(self.train_loader)
                all_labels = []
                for _, labels in training_data[0]:
                    all_labels.extend(labels.tolist())  # Chuyển từ Tensor sang list và nối
                diversity = len(set(all_labels))
                size_label = (len(training_data[0]), diversity)
                data = {"action": "UPDATE-fedrhlp", "client_id": self.client_id, "size_label": size_label}
                src.Log.print_with_color(f"Data gui di: {size_label}", "yellow")
                self.send_to_server(data)
                return True
            elif self.algorithm == "hicsfl":
                print("algorithm hicsfl")
                
                criterion = nn.CrossEntropyLoss()
            
                src.Log.print_with_color(f"Strat trainning: {self.data_name}", "yellow")
                self.training_time = self.epoch * self.self.client_sizes / self.speed
                result,bias,_ = self.train_func(self.model, self.lr,self.data_name, self.momentum, self.train_loader, criterion, self.clip_grad_norm,self.epoch, return_bias = True)
                self.train_loader = None
        
                model_state_dict = self.model.state_dict()
                if self.device != "cpu":
                    for key in model_state_dict:
                        model_state_dict[key] = model_state_dict[key].to('cpu')

                print(f"Bias: {bias}")
                data = {"action": "UPDATE", "client_id": self.client_id, "result": result, "size": sum(self.label_counts),
                        "message": "Sent parameters to Server", "parameters": model_state_dict, "num_data":self.client_sizes, "bias": bias, "num_class": num_labels, "training_time": self.training_time, "speed": self.speed}
                src.Log.print_with_color(f"Data gui di: {self.client_id},{self.client_sizes}", "yellow")
                src.Log.print_with_color("[>>>] Client sent parameters to server", "red")
                self.send_to_server(data)
                return True
            elif self.algorithm == "feel":
                print("Gui goi tin update-feel")
                if self.selected_flag == True:
                    self.staleness = 0
                    self.selected_flag = False
                elif self.selected_flag == False and self.model != None :
                    self.staleness +=1
                    self.age +=1
                size = self.client_sizes
                staleness = self.staleness
                age = self.age
                data = {"action": "UPDATE-feel", "client_id": self.client_id, "size": self.client_sizes, "staleness": staleness, "age": age}
                src.Log.print_with_color(f"Data gui di: size: {size}, staleness: {staleness}, age: {age}", "yellow")
                self.send_to_server(data)
                return True


        elif action == "INFOR":
            self.data_name = self.response["data_name"]
            if self.data_name == "CIFAR10" or self.data_name == "MNIST":
                num_labels = 10
                
            elif self.data_name == "DOMAIN":
                num_labels = 21
            elif self.data_name == "DOMAIN2":
                num_labels = 2
            else:
                num_labels = 0
            self.data_range[0] = self.response["range[0]"]
            self.data_range[1] = self.response["range[1]"]
            self.label_counts = np.array([random.randint(self.data_range[0], self.data_range[1]) for _ in range(num_labels)])
            data = {"action": "UPDATE-INFOR", "client_id": self.client_id, "label_counts": self.label_counts, "num_labels": num_labels}
            src.Log.print_with_color("[>>>] Client sent Label count to server", "red")
            self.send_to_server(data)
            return True
        elif action == "START-RHLP":
           
            criterion = nn.CrossEntropyLoss()
        
            src.Log.print_with_color(f"Strat trainning: {self.data_name}", "yellow")
            self.training_time = self.epoch * self.client_sizes/ self.speed
            result = self.train_func(self.model, self.lr,self.data_name, self.momentum, self.train_loader, criterion, self.clip_grad_norm,1)
            self.train_loader = None
    
            model_state_dict = self.model.state_dict()
            if self.device != "cpu":
                for key in model_state_dict:
                    model_state_dict[key] = model_state_dict[key].to('cpu')

            data = {"action": "UPDATE", "client_id": self.client_id, "result": result, "size": sum(self.label_counts),
                    "message": "Sent parameters to Server", "parameters": model_state_dict, "num_data":self.client_sizes, "training_time": self.training_time, "speed": self.speed}
            
            src.Log.print_with_color(f"Data gui di: {self.client_id},{self.client_sizes}", "yellow")
            src.Log.print_with_color("[>>>] Client sent parameters to server", "red")
            self.send_to_server(data)
            return True
        elif action == "START-feel":
            logits = self.response["logits"]
            state_tensor = self.response["state_tensor"]
            selected_flag = self.response["selected_flag"]
            stats = self.response["stats"]
            selected_index = self.response["selected_index"]
            voi_state_dict = self.response["voi_state_dict"]
            self.voi_estimator.load_state_dict(voi_state_dict)
            if selected_flag :
                self.selected_flag = True

            criterion = nn.CrossEntropyLoss()
            
            src.Log.print_with_color(f"Strat trainning: {self.data_name}", "yellow")
            self.training_time = self.epoch * self.client_sizes/ self.speed
            result,_,loss = self.train_func(self.model, self.lr,self.data_name, self.momentum, self.train_loader, criterion, self.clip_grad_norm,1)
            self.train_loader = None
            stats[0] = loss

            reward = 5.0 - loss
            actions = logits
            value = self.voi_estimator.get_value(state_tensor)
            log_prob = -((actions - actions.detach()) ** 2)
            
            model_state_dict = self.model.state_dict()
            if self.device != "cpu":
                for key in model_state_dict:
                    model_state_dict[key] = model_state_dict[key].to('cpu')

            data = {"action": "UPDATE", "client_id": self.client_id, "result": result, "size": sum(self.label_counts),
                    "message": "Sent parameters to Server", "parameters": model_state_dict, "num_data":self.client_sizes, "reward": reward, "actions": actions, "value": value, "log_prob":log_prob , "selected_index": selected_index,"training_time": self.training_time, "speed": self.speed}
            
            src.Log.print_with_color(f"Data gui di: {self.client_id},{self.client_sizes}", "yellow")
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
