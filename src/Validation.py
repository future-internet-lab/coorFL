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

        klass = getattr(src.Model, self.model_name)
        if klass is None:
            raise ValueError(f"Class '{model_name}' does not exist.")
        self.model = klass()

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
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        elif self.data_name == "DOMAIN":
            test_set = src.Utils.load_dataset("domain_data/domain_test_dataset.pkl")
            test_set = src.Utils.modify_labels(test_set)
        elif self.data_name == "DOMAIN2":
            benign_test_ds = src.Utils.load_dataset("domain2/benign_test.pkl")
            dga_1_test_ds = src.Utils.load_dataset("domain2/dga_1_test.pkl")
            dga_2_test_ds = src.Utils.load_dataset("domain2/dga_2_test.pkl")
            test_set = ConcatDataset([benign_test_ds, dga_1_test_ds, dga_2_test_ds])
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
        elif self.data_name == "DOMAIN2":
            return self.test_domain_2(device)
            pass
        else:
            raise ValueError(f"Not found test function for data name {self.data_name}")

    def test_image(self, device):
        test_loss = 0
        correct = 0
        for data, target in tqdm(self.test_loader):
            data = data.to(device)
            target = target.to(device)
            output = self.model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

        test_loss /= len(self.test_loader.dataset)
        accuracy = 100.0 * correct / len(self.test_loader.dataset)
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset), accuracy))
        self.logger.log_info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset), accuracy))

        if np.isnan(test_loss) or math.isnan(test_loss) or abs(test_loss) > 10e5:
            return False, 0.0

        return True, accuracy

    def test_domain(self, device):
        all_preds = []
        all_labels = []

        criterion = nn.BCELoss()

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

        print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}')
        self.logger.log_info(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}')

        return True, accuracy*100
    
    def test_domain_2(self, device):
        self.model.eval()
        test_correct = 0
        test_samples = 0

        with torch.no_grad():
            for x_batch, y_batch in tqdm(self.test_loader):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                logits = self.model(x_batch)
                preds = logits.argmax(dim=1)
                test_correct += (preds == y_batch).sum().item()
                test_samples += y_batch.size(0)

        test_acc = test_correct / test_samples

        print(f"Test Acc={test_acc:.4f}")
        return True, test_acc*100
