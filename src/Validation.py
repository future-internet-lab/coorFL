import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import math

from torch.utils.data import ConcatDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import src.Model
import src.Utils
import src.Log

from tqdm import tqdm


class Validation:
    def __init__(self, model_name, data_name, logger):
        self.model_name = model_name
        self.data_name = data_name
        self.logger = logger

        if data_name == "DOMAIN2":
            if model_name == "Transformer":
                self.model = src.Model.PositionalEncodingTransformer()
            elif model_name == "CNN":
                self.model = src.Model.CNNClassifier()
            else:
                raise ValueError(f"Model name '{model_name}' is not valid.")
        elif data_name == "CIFAR10":
            if model_name == "ResNet":
                self.model = src.Model.ResNet18()
            elif model_name == "MobileNet":
                self.model =src.Model. MobileNetV2()
            else:
                raise ValueError(f"Model name '{model_name}' is not valid.")
        elif self.data_name == "CICIDS":
                self.model = src.Model.FTTransformer(d_token = 192, n_blocks = 8, n_heads = 6, d_ff = 768)
            
        else:
            raise ValueError(f"Data name '{data_name}' is not valid.")

        self.test_loader = None

        test_set = None
        if self.data_name == "MNIST":
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
        elif self.data_name == "CIFAR10":
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        elif self.data_name == "DOMAIN":
            test_set = src.Utils.load_dataset("domain_data/domain_test_dataset.pkl")
            test_set = src.Utils.modify_labels(test_set)
        elif self.data_name == "DOMAIN2":
            benign_test_ds = src.Utils.load_dataset("domain2/benign_test.pkl")
            dga_1_test_ds = src.Utils.load_dataset("domain2/dga_1_test.pkl")
            dga_2_test_ds = src.Utils.load_dataset("domain2/dga_2_test.pkl")
            dga_3_test_ds = src.Utils.load_dataset("domain2/dga_3_test.pkl")
            dga_4_test_ds = src.Utils.load_dataset("domain2/dga_4_test.pkl")

            test_set = ConcatDataset([benign_test_ds, dga_1_test_ds, dga_2_test_ds, dga_3_test_ds, dga_4_test_ds])
        elif self.data_name == "CICIDS":
            benign_test_ds = src.Model.CICIDSDataset("CIC-IDS2017/benign_test.pkl")  
            atk_0_test_ds = src.Model.CICIDSDataset("CIC-IDS2017/attack_test_0.pkl")
            atk_1_test_ds = src.Model.CICIDSDataset("CIC-IDS2017/attack_test_1.pkl")
            atk_2_test_ds = src.Model.CICIDSDataset("CIC-IDS2017/attack_test_2.pkl")
            atk_3_test_ds = src.Model.CICIDSDataset("CIC-IDS2017/attack_test_3.pkl")

            test_set = ConcatDataset([benign_test_ds, atk_0_test_ds, atk_1_test_ds, atk_2_test_ds, atk_3_test_ds])
        else:
            raise ValueError(f"Do not have data name '{self.data_name}.")


        self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2)

    def test(self, avg_state_dict, device):
        self.model.load_state_dict(avg_state_dict)
        self.model.to(device)
        # evaluation mode
        self.model.eval()
        if self.data_name == "MNIST" or self.data_name == "CIFAR10":
            return self.test_image(device)
        elif self.data_name == "DOMAIN":
            return self.test_domain(device)
        elif self.data_name == "CICIDS":
            return self.test_cicids2017(device)
        elif self.data_name == "DOMAIN2":
            return self.test_domain_2(device)
            pass
        
        else:
            raise ValueError(f"Not found test function for data name {self.data_name}")

    def test_image(self, device):
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for x_batch, y_batch in self.test_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                logits = self.model(x_batch)
                loss = criterion(logits, y_batch)
                preds = logits.argmax(dim=1)

                total_loss += loss.item() * x_batch.size(0)
                total_correct += (preds == y_batch).sum().item()
                total_samples += x_batch.size(0)

        avg_loss = total_loss / total_samples if total_samples > 0 else float("inf")
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        

        return True,avg_loss, accuracy

    def test_domain(self, device):
        all_preds = []
        all_labels = []

        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for inputs, labels in tqdm(self.test_loader):
                input = input.to(device)
                labels = labels.to(device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                preds = (outputs > 0.5).float()

                if torch.isnan(loss).any():
                    src.Log.print_with_color("NaN detected in loss, stop training", "yellow")
                    return False

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)

       
        return True, accuracy,precision,recall,f1
    def test_cicids2017(self,device):
        self.model.eval()
        all_preds, all_labels = [], []
        criterion = nn.BCEWithLogitsLoss()
        total_samples = 0
        total_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in self.test_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device).float()

                logits = self.model(x_batch)
                loss = criterion(logits, y_batch)
                preds = torch.sigmoid(logits).cpu() > 0.5

                total_loss += loss.item() * x_batch.size(0)
                total_samples += x_batch.size(0)

                all_preds.extend(preds.int().tolist())
                all_labels.extend(y_batch.tolist())

        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, zero_division=0)
        rec = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        return True,avg_loss, acc,prec,rec,f1
    
    def test_domain_2(self, device):
        self.model.eval()
        all_preds = []
        all_labels = [] 
        criterion = nn.CrossEntropyLoss()
        total_samples = 0
        total_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in self.test_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                logits = self.model(x_batch)
                loss = criterion(logits, y_batch)
                preds = logits.argmax(dim=1)

                total_loss += loss.item() * x_batch.size(0)
                total_samples += x_batch.size(0)

                all_preds.extend(preds.cpu().numpy().flatten())
                all_labels.extend(y_batch.cpu().numpy().flatten())   

        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')

        # Log the results
       
        return True,avg_loss, accuracy,precision,recall,f1

