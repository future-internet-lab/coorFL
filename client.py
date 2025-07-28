import pika
import uuid
import argparse
import yaml
from tqdm import tqdm
import random
import numpy as np
import torch
import torch.optim as optim

import src.Log
from src.RpcClient import RpcClient
from src.Utils import DomainDataset

parser = argparse.ArgumentParser(description="Split learning framework")
parser.add_argument('--device', type=str, required=False, help='Device of client')
parser.add_argument('--zone', type=int, required=True, help='ID của client (số nguyên)')

args = parser.parse_args()

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

client_id = uuid.uuid4()
address = config["rabbit"]["address"]
username = config["rabbit"]["username"]
password = config["rabbit"]["password"]

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


credentials = pika.PlainCredentials(username, password)

def train_domain(model, client_loader, criterion, optimizer, return_bias=False, num_classes=2, num_epochs = 1, clip_grad_norm=None):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for epoch in range(num_epochs):
        print(f"Epoch: {epoch}")
        for x_batch, y_batch in client_loader:
            if x_batch.size(0) == 1:
                continue
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            y_batch = y_batch.long()

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()

            if clip_grad_norm and clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

            optimizer.step()

            total_loss += loss.item() * x_batch.size(0)
            total_samples += x_batch.size(0)

    avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')

    if not return_bias:
        return True, None,avg_loss

    # --- Bias estimation ---
    num_classes=2
    assert num_classes is not None, "You must provide num_classes when return_bias=True"
    model.eval()
    bias_vector = torch.zeros(num_classes, device=device)
    bias_total = 0
    with torch.no_grad():
        for x_batch, _ in client_loader:
            x_batch = x_batch.to(device)
            logits = model(x_batch)
            probs = torch.softmax(logits, dim=1)
            bias_vector += probs.sum(dim=0)
            bias_total += x_batch.size(0)

    bias_vector /= bias_total if bias_total > 0 else 1
    return True, bias_vector.detach().cpu().numpy(),avg_loss

def train_cifar10(model, client_loader, criterion, optimizer, return_bias=False, num_classes = 10, num_epochs = 1, clip_grad_norm=None):
    model.train()
    total_loss = 0.0
    total_samples = 0
    num_classes = 10
    for epoch in range(num_epochs):
        print(f"Epoch: {epoch}")
        for x_batch, y_batch in client_loader:
            if x_batch.size(0) == 1:
                continue
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            y_batch = y_batch.long()

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()

            if clip_grad_norm and clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

            optimizer.step()

            total_loss += loss.item() * x_batch.size(0)
            total_samples += x_batch.size(0)

    avg_loss = total_loss / total_samples if total_samples > 0 else float("inf")

    if not return_bias:
        return True,None,avg_loss

    # Nếu ai cố gọi return_bias=True thì ta trả lời thẳng thắn:
    raise NotImplementedError("Bias estimation is not implemented in global CIFAR-10 training.")
def train_on_device(model, lr, data_name, momentum, trainloader, criterion, clip_grad_norm=None, epoch=1, return_bias = False,num_classes = None):
    #optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if data_name == "DOMAIN" or data_name == "DOMAIN2" :
        return train_domain(model, trainloader, criterion, optimizer, return_bias, num_classes, epoch, clip_grad_norm)
    elif data_name == "CIFAR10":
        return train_cifar10(model, trainloader, criterion, optimizer, return_bias, num_classes, epoch, clip_grad_norm)
    

if __name__ == "__main__":
    src.Log.print_with_color("[>>>] Client sending registration message to server...", "red")
    data = {"action": "REGISTER", "client_id": client_id,"Zone": args.zone, "message": "Hello from Client!"}
    client = RpcClient(client_id, address, username, password, train_on_device, device,args.zone)
    client.send_to_server(data)
    client.wait_response()
