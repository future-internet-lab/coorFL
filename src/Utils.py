import numpy as np
import random
import pickle

import torch
from torch.utils.data import Dataset, Subset


class DomainDataset(Dataset):
    def __init__(self, domains, labels, max_len):
        self.domains = domains
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.domains)

    def __getitem__(self, idx):
        domain = self.domains[idx]
        label = self.labels[idx]

        # Convert domain to integer indices (ASCII encoding for simplicity)
        encoded = [ord(char) for char in domain]
        if len(encoded) < self.max_len:
            encoded += [0] * (self.max_len - len(encoded))  # Padding
        else:
            encoded = encoded[:self.max_len]  # Truncate

        return torch.tensor(encoded, dtype=torch.long), torch.tensor(label, dtype=torch.float)


class CustomDataset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        data, label = self.dataset[original_idx]
        modified_label = min(label, 1)
        return data, torch.tensor(modified_label, dtype=torch.float)


def generate_random_array(target_sum, size, max_value=None):
    random_array = np.random.randint(1, target_sum * size, size)

    current_sum = random_array.sum()
    current_sum = (target_sum * size * random_array)/np.sum(current_sum)

    for i in range(size):
        if max_value and current_sum[i] > max_value:
            current_sum[i] = max_value

    return current_sum.astype(int).tolist()


def non_iid_rate(num_data, rate):
    result = []
    for _ in range(num_data):
        if rate < random.random():
            result.append(0)
        else:
            result.append(1)
    return np.array(result)


def load_dataset(file_path):
    with open(file_path, 'rb') as file:
        dataloader = pickle.load(file)
    print(f"DataLoader loaded from {file_path}.")
    return dataloader


def modify_labels(dataset):
    dataset.labels = [min(label, 1) for label in dataset.labels]
    return dataset
