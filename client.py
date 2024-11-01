import time
import pika
import uuid
import pickle
import argparse
import yaml
import random
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import src.Log
from src.RpcClient import RpcClient
from src.Model import ResNet50

parser = argparse.ArgumentParser(description="Split learning framework")
# parser.add_argument('--id', type=int, required=True, help='ID of client')

args = parser.parse_args()

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

client_id = uuid.uuid4()
address = config["rabbit"]["address"]
username = config["rabbit"]["username"]
password = config["rabbit"]["password"]

batch_size = config["learning"]["batch-size"]
lr = config["learning"]["learning-rate"]
momentum = config["learning"]["momentum"]

device = None

if torch.cuda.is_available():
    device = "cuda"
    print(f"Using device: {torch.cuda.get_device_name(device)}")
else:
    device = "cpu"
    print(f"Using device: CPU")

model = ResNet50(10)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
criterion = nn.CrossEntropyLoss()

credentials = pika.PlainCredentials(username, password)

# Read and load dataset
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)


label_to_indices = defaultdict(list)
for idx, (_, label) in enumerate(trainset):
    label_to_indices[label].append(idx)


def train_on_device(label_counts):
    selected_indices = []
    for label, count in enumerate(label_counts):
        selected_indices.extend(random.sample(label_to_indices[label], count))

    subset = torch.utils.data.Subset(trainset, selected_indices)

    trainloader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True)

    model.to(device)
    model.train()
    for (training_data, label) in tqdm(trainloader):
        if training_data.size(0) == 1:
            continue
        training_data = training_data.to(device)
        output = model(training_data)
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    src.Log.print_with_color("[>>>] Client sending registration message to server...", "red")
    data = {"action": "REGISTER", "client_id": client_id, "message": "Hello from Client!"}
    client = RpcClient(client_id, model, address, username, password, train_on_device, device)
    client.send_to_server(data)
    client.wait_response()
