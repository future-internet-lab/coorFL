import pika
import uuid
import argparse
import yaml
from tqdm import tqdm

import torch
import torch.optim as optim

import src.Log
from src.RpcClient import RpcClient
from src.Utils import DomainDataset

parser = argparse.ArgumentParser(description="Split learning framework")
parser.add_argument('--device', type=str, required=False, help='Device of client')

args = parser.parse_args()

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

client_id = uuid.uuid4()
address = config["rabbit"]["address"]
username = config["rabbit"]["username"]
password = config["rabbit"]["password"]
data_name = config["server"]["data-name"]

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


def train_on_device(model, lr, momentum, trainloader, criterion, epoch = 1, clip_grad_norm=None):
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    model.train()
    for _ in range(epoch):
        for (training_data, label) in tqdm(trainloader):
            if training_data.size(0) == 1:
                continue
            training_data = training_data.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            if data_name == "DOMAIN2":
                output = model(training_data)
                output = torch.softmax(output, dim=1)[:, 1]
                loss = criterion(output, label.float())
            else:
                output = model(training_data)
                loss = criterion(output, label)

            if torch.isnan(loss).any():
                src.Log.print_with_color("NaN detected in loss, stop training", "yellow")
                return False

            loss.backward()
            if clip_grad_norm and clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            optimizer.step()

    return True


if __name__ == "__main__":
    src.Log.print_with_color("[>>>] Client sending registration message to server...", "red")
    data = {"action": "REGISTER", "client_id": client_id, "message": "Hello from Client!"}
    client = RpcClient(client_id, address, username, password, train_on_device, device)
    client.send_to_server(data)
    client.wait_response()
