import numpy as np
import random
import pickle
import string

import torch
from torch.utils.data import Dataset, Subset
from torchvision.datasets import CIFAR10
import torch.nn as nn

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

def split_cifar10_to_zones(root, zone_spec, transform_train=None, download=True):
    train_ds = CIFAR10(root, train=True, transform=transform_train, download=download)

    train_labels = np.array(train_ds.targets)

    train_sets = {}
    for zone, cls_list in zone_spec.items():
        train_idx = np.where(np.isin(train_labels, cls_list))[0]
        train_sets[zone] = Subset(train_ds, train_idx)

    return train_sets

def extract_all_linear_weights(state_dict):
    all_flattened = []

    for key in state_dict:
        if not key.endswith("weight"):
            continue

        w = state_dict[key]
        if w.dim() != 2:
            continue  # chỉ lấy weight 2D (Linear)

        bias_key = key.replace("weight", "bias")
        if bias_key not in state_dict:
            continue  # bỏ qua nếu không có bias (vd: embedding.weight)

        b = state_dict[bias_key]

        # Chuyển về CPU và detach
        w = w.detach().cpu().flatten()
        b = b.detach().cpu().flatten()

        all_flattened.append(torch.cat([w, b]))

    if not all_flattened:
        raise ValueError("Không tìm thấy lớp Linear hợp lệ nào.")

    return torch.cat(all_flattened).numpy()

def compute_divergence(state_dict_a, state_dict_b):
    divergence = 0.0
    norm_b = 0.0

    for key in state_dict_a:
        # Bỏ qua nếu không phải float tensor
        if not torch.is_floating_point(state_dict_a[key]):
            continue

        # Đưa state_dict_b[key] về cùng device với state_dict_a[key]
        tensor_a = state_dict_a[key]
        tensor_b = state_dict_b[key].to(tensor_a.device)

        # Tính hiệu và chuẩn
        diff = tensor_a - tensor_b
        divergence += torch.norm(diff) ** 2
        norm_b += torch.norm(tensor_b) ** 2

    if norm_b == 0:
        return 0.0  # tránh chia 0

    return (divergence.sqrt() / norm_b.sqrt()).item()

def fedcls_cluster_clients(client_labels, n_classes):
    # encoded = [one_hot_encode(lbls, n_classes) for lbls in client_labels]
    encoded = torch.tensor(client_labels)
    clusters = []
    cluster_labels = []
    assigned = [False] * len(client_labels)

    for i, vec_i in enumerate(encoded):
        if assigned[i]:
            continue
        cluster = [i]
        cl_label = vec_i.clone()
        assigned[i] = True
        for j in range(i + 1, len(encoded)):
            if not assigned[j] and torch.equal(vec_i, encoded[j]):
                cluster.append(j)
                assigned[j] = True
        clusters.append(cluster)
        cluster_labels.append(cl_label)
    print(f"Cluster {clusters} and labels {cluster_labels}")
    return clusters, cluster_labels


def fedcls_select_clusters(cluster_labels, n_classes, top_k=1):
    selected = set()
    cover = torch.zeros(n_classes, dtype=torch.bool)

    while not torch.equal(cover, torch.ones(n_classes, dtype=torch.bool)):
        gain_list = []
        for i, cl in enumerate(cluster_labels):
            if i in selected:
                continue
            cl_bool = cl.bool()
            new_gain = (cover | cl_bool).sum() - cover.sum()
            gain_list.append((new_gain.item(), i))

        if not gain_list:
            break

        # Sắp xếp theo gain giảm dần và chọn top_k cluster
        gain_list.sort(reverse=True)
        chosen = [i for _, i in gain_list[:top_k]]

        for i in chosen:
            selected.add(i)
            cover |= cluster_labels[i].bool()

    return list(selected)

def HACCS_select_clients(cluster_labels, latencies, ratio=0.2):
    num_clients = len(cluster_labels)
    num_to_select = max(1, int(ratio * num_clients))

    selected_clients = []
    unique_clusters = np.unique(cluster_labels)
    num_clusters = len(unique_clusters)

    per_cluster = max(1, num_to_select // num_clusters)

    for cluster in unique_clusters:
        idx = np.where(cluster_labels == cluster)[0]
        if len(idx) == 0:
            continue
        sorted_idx = sorted(idx, key=lambda i: latencies[i])
        selected_clients.extend(sorted_idx[:min(len(idx), per_cluster)])

    return selected_clients[:num_to_select]  # Trường hợp dư thì cắt bớt
def hellinger_dist(p, q):
    return np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))
class PPOVoIEstimator(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )
    def get_action(self, states): return self.policy_net(states).squeeze()
    def get_value(self, states): return self.value_net(states).squeeze()
def find_speed(client_id, speed_id_list):
    for item in speed_id_list:
        if client_id in item:
            return item[client_id]
    return None  # nếu không tìm thấy