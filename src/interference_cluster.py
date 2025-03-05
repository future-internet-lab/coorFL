import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import yaml
from src.Model import *  
import numpy as np
from src.Cluster import *
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

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

data_name = config["server"]["data-name"]
if data_name == "MNIST":
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                                transform=transform_train)
elif data_name == "CIFAR10":
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                                    transform=transform_train)
elif data_name == "DOMAIN":
    trainset = src.Utils.load_dataset("domain_data/domain_train_dataset.pkl")

else:
    raise ValueError(f"Data name '{data_name}' is not valid.")


def create_trainloader(trainset, num_samples):
    indices = list(range(num_samples))  
    subset = Subset(trainset, indices)
    trainloader = DataLoader(subset, batch_size=num_samples, shuffle=False)
    return trainloader



def interference(all_model_parameters):
    neural_last_layer = []  
    num_samples = config["server"]["sample_cluster"]  
    model_name = config["server"]["model"]   
    
    for i in all_model_parameters:
        client_id = i['client_id']
        model_state_dict = i['weight']
        
        trainloader = create_trainloader(trainset, num_samples)

        if model_name == "LeNet_MNIST":
            model = LeNet_MNIST()  
        elif model_name =="LeNet_CIFAR10":
            model = LeNet_CIFAR10()
        elif model_name =="Block":
            model = Block()
        elif model_name == "MobileNetV2":
            model = MobileNetV2()
        elif model_name == "BasicBlock":
            model = BasicBlock()
        elif model_name == "Bottleneck":
            model = Bottleneck()
        elif model_name == "ResNet":
            model = ResNet()
        elif model_name == "VGG16":
            model = VGG16()
        elif model_name == "VGG19":
            model = VGG19()
        elif model_name == "DGAClassifier":
            model = DGAClassifier()
        elif model_name == "LSTM":
            model = LSTM()
        
        model.eval()
        model.load_state_dict
        
        for inputs, labels in trainloader:
            with torch.no_grad():  
                outputs = model(inputs)
                
                outputs_list = outputs.tolist()  
                 
                output_array = np.array(outputs_list).flatten()
                
                neural_last_layer.append({"client_id": client_id, "output": output_array, "cluster_index": None})

            break  
    label_counts = np.array([item["output"] for item in neural_last_layer])

    
    num_clusters, labels, _  = clustering_algorithm(label_counts, config)
    
    for i, client_data in enumerate(neural_last_layer):
        client_data['cluster_index'] = labels[i]
    for i, client_data in enumerate(all_model_parameters):
        
        client_data['cluster_index'] = neural_last_layer[i]['cluster_index']
    return all_model_parameters

