import numpy as np
import random
import pickle
import string

import torch
from torch.utils.data import Dataset, Subset, ConcatDataset


ALPHABET = string.ascii_lowercase + string.digits + "."
char2idx = {c: i + 1 for i, c in enumerate(ALPHABET)}  # padding=0
idx2char = {i: c for c, i in char2idx.items()}       # Reverse mapping index -> character
vocab_size = len(char2idx) + 1
MAX_LEN = 50

def domain_to_tensor(domain):
    arr = [char2idx.get(c, 0) for c in domain.lower()][:MAX_LEN]
    arr += [0] * (MAX_LEN - len(arr))
    return torch.tensor(arr, dtype=torch.long)

class DomainDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        dom, lbl = self.samples[idx]
        x = domain_to_tensor(dom)
        return x, lbl
    

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

def extract_latent_features(model, dataloader, device):
    """
    Trích xuất vector latent representation từ DomainVAE cho mỗi sample.
    """
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x = x.to(device)
            mu, _ = model.encode(x)
            features.append(mu.cpu().numpy())
            labels.append(y.cpu().numpy())

    return np.vstack(features), np.hstack(labels)


def dga_label(num_clients):
    random.seed(1)
    all_train_datasets = 4
    dga_distribution = [random.randint(0, all_train_datasets-1) for _ in range(num_clients)]
    return dga_distribution