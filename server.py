import os
import pika
import pickle
import argparse
import sys
import yaml
import signal
import copy
import torch
import requests
import random
import numpy as np
import math
import src.Utils
import src.Utils
import src.Utils
import src.Utils
import src.Utils
import src.Validation
import src.Log
from src.Selection import client_selection_speed_base, client_selection_random
from src.Cluster import clustering_algorithm
from src.Utils import DomainDataset
from src.Notify import send_mail
from src import *
from sklearn.metrics import pairwise_distances
from sklearn.cluster import OPTICS
from requests.auth import HTTPBasicAuth
from sklearn.cluster import KMeans, AffinityPropagation
import torch.optim as optim
import torch.nn as nn

parser = argparse.ArgumentParser(description="Federated learning framework with controller.")

parser.add_argument('--device', type=str, required=False, help='Device of server')

args = parser.parse_args()

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

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

total_clients = config["server"]["clients"]
model_name = config["server"]["model"]
data_name = config["server"]["data-name"]
address = config["rabbit"]["address"]
username = config["rabbit"]["username"]
password = config["rabbit"]["password"]

num_round = config["server"]["num-round"]
save_parameters = config["server"]["parameters"]["save"]
load_parameters = config["server"]["parameters"]["load"]
validation = config["server"]["validation"]
random_seed = config["server"]["random-seed"]
accuracy_drop = config["server"]["accuracy-drop"]

data_distribution = config["server"]["data-distribution"]
data_range = data_distribution["num-data-range"]
non_iid_rate = data_distribution["non-iid-rate"]
refresh_each_round = data_distribution["refresh-each-round"]

stop_when_false = config["server"]["stop-when-false"]
email_config = config["server"]["send-mail"]

# Algorithm
algorithm_name = config["server"]["algorithm"]
data_mode = config["server"]["data-mode"]
client_selection_config = config["server"]["client-selection"]
client_cluster_config = config["server"]["client-cluster"]
client_cluster_bool = client_cluster_config["enable"]
client_selection_bool = client_selection_config["enable"]
cluster_iter  = client_cluster_config["cluster_iter"]

#fedcls
ratio = config["server"]["fedcls"]["ratio"]
#csfedavg
p = config["server"]["csfedavg"]["p"]
alpha = config["server"]["csfedavg"]["alpha"]
#fedrhlp
c1 = config["server"]["fedrhlp"]["c1"]
c2 = config["server"]["fedrhlp"]["c2"]
#hicsfl
ratio_hicsfl = config["server"]["hicsfl"]["selection_ratio"]
num_cluster = config["server"]["hicsfl"]["num_cluster"]
CLIENTS_PER_ROUND = max(1, int(ratio_hicsfl * total_clients))
#feel
ratio_feel = config["server"]["feel"]["selected_ratio"]

# Clients
batch_size = config["learning"]["batch-size"]
lr = config["learning"]["learning-rate"]
momentum = config["learning"]["momentum"]
clip_grad_norm = config["learning"]["clip-grad-norm"]

log_path = config["log_path"]

if data_name == "CIFAR10" or data_name == "MNIST":
    num_labels = 10
elif data_name == "DOMAIN":
    num_labels = 21
elif data_name == "DOMAIN2":
    num_labels = 2
else:
    num_labels = 0

if random_seed:
    random.seed(random_seed)


class Server:
    def __init__(self):
        credentials = pika.PlainCredentials(username, password)
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(address, 5672, '/', credentials))
        self.channel = self.connection.channel()
        self.num_round = num_round
        self.round = self.num_round

        self.channel.queue_declare(queue='rpc_queue')

        self.total_clients = total_clients
        self.current_clients = 0
        self.updated_clients = 0
        self.last_accuracy = 0.0
        self.responses = {}  # Save response
        self.list_clients = []
        self.all_model_parameters = []
        self.avg_state_dict = None
        self.round_result = True
        self.label_counts = []
        self.non_iid_label = None
        self.zone_distribution = []
        self.device = device
        self.selected_client_temp = []
        self.divergence_scores = {}
        self.size_label = []
        self.stats = []
        self.size = []
        self.count_i = 0 #csfedavg
        self.list_i = [] # csfedavg
        self.client_biases = []
        self.voi_estimator = src.Utils.PPOVoIEstimator(state_dim=4)
        self.voi_optimizer = optim.Adam(self.voi_estimator.parameters(), lr=1e-3)
        self.list_staleness = [0 for _ in range(total_clients)]
        self.list_age = [0 for _ in range(total_clients)]
        self.reward = []
        self.actions = []
        self.values = []
        self.log_prob = []
        self.local_speed = []
        self.training_time = []

        if client_cluster_bool:
            self.cluster_state = True
        else:
            self.cluster_state = False
        self.client_vectors = []
        if model_name == "Transformer":
            self.speeds = [random.randrange(224, 1792) for _ in range(total_clients)]
        elif model_name == "CNN":
            self.speeds = [random.randrange(217, 1736) for _ in range(total_clients)]
        elif model_name == "ResNet":
            self.speeds = [random.randrange(13, 104) for _ in range(total_clients)]
        elif model_name == "MobileNet":
            self.speeds = [random.randrange(14, 112) for _ in range(total_clients)]
        else:
            raise ValueError(f"Model name '{model_name}' is not valid.")
        print(f"speeds_list = {self.speeds}")
        self.selected_client = []
        self.id_speed_dict = []

        self.logger = src.Log.Logger(f"{log_path}/app.log")
        self.validation = src.Validation.Validation(model_name, data_name, self.logger)

        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(queue='rpc_queue', on_message_callback=self.on_request)

        self.logger.log_info("### Application start ###\n")
        src.Log.print_with_color(f"Server is waiting for {self.total_clients} clients.", "green")

    def start(self):
        self.channel.start_consuming()


    def send_to_response(self, client_id, message):
        """
        Response message to clients
        :param client_id: client ID
        :param message: message
        :return:
        """
        reply_channel = self.channel
        reply_queue_name = f'reply_{client_id}'
        reply_channel.queue_declare(reply_queue_name, durable=False)

        src.Log.print_with_color(f"[>>>] Sent notification to client {client_id}", "red")
        reply_channel.basic_publish(
            exchange='',
            routing_key=reply_queue_name,
            body=message
        )

    def on_request(self, ch, method, props, body):
        """
        Handler request from clients
        :param ch: channel
        :param method:
        :param props:
        :param body: message body
        :return:
        """
        message = pickle.loads(body)
        routing_key = props.reply_to
        action = message["action"]
        client_id = message["client_id"]
        self.responses[routing_key] = message

        if action == "REGISTER":
            src.Log.print_with_color(f"[<<<] Total client: {self.total_clients}", "blue")
            zone = message["Zone"]
            self.zone_distribution.append(zone)
            if str(client_id) not in self.list_clients:
                self.list_clients.append(str(client_id))

                src.Log.print_with_color(f"[<<<] Received message from client: {message}", "blue")

            # If consumed all clients - Register for first time
            if algorithm_name == "our":
                
                if len(self.list_clients) == self.total_clients:
                    for i in range(total_clients):
                        self.id_speed_dict.append({self.list_clients[i]:self.speeds[i]})
                    print(f"id speed list = {self.id_speed_dict}")
                    print(f"Algorithm name {algorithm_name}")
                    src.Log.print_with_color("All clients are connected. Sending notifications.", "green")
                    src.Log.print_with_color(f"Zone_distribution {self.zone_distribution}", "green")
                    #self.client_selection()
                    src.Log.print_with_color(f"Start training round {self.num_round - self.round + 1}", "yellow")
                    self.notify_clients()
            elif algorithm_name == "csfedavg":
                if len(self.list_clients) == self.total_clients:
                    for i in range(total_clients):
                        self.id_speed_dict.append({self.list_clients[i]:self.speeds[i]})
                    print(f"id speed list = {self.id_speed_dict}")
                    print(f"Algorithm name {algorithm_name}")
                    src.Log.print_with_color("All clients are connected. Sending notifications.", "green")
                    src.Log.print_with_color(f"Zone_distribution {self.zone_distribution}", "green")
                    self.client_selection()
                    src.Log.print_with_color(f"Start training round {self.num_round - self.round + 1}", "yellow")
                    self.notify_clients()
            elif algorithm_name == "fedcls":
                if len(self.list_clients) == self.total_clients:
                    for i in range(total_clients):
                        self.id_speed_dict.append({self.list_clients[i]:self.speeds[i]})
                    print(f"id speed list = {self.id_speed_dict}")
                    print(f"Algorithm name {algorithm_name}")
                    self.send_infor()
            elif algorithm_name == "fedrhlp":
                if len(self.list_clients) == self.total_clients:
                    for i in range(total_clients):
                        self.id_speed_dict.append({self.list_clients[i]:self.speeds[i]})
                    print(f"id speed list = {self.id_speed_dict}")
                    print(f"Algorithm name {algorithm_name}")
                    src.Log.print_with_color("All clients are connected. Sending notifications.", "green")
                    src.Log.print_with_color(f"Zone_distribution {self.zone_distribution}", "green")
                    src.Log.print_with_color(f"Start training round {self.num_round - self.round + 1}", "yellow")
                    self.notify_clients()
            elif algorithm_name == "haccs":
                if len(self.list_clients) == self.total_clients:
                    for i in range(total_clients):
                        self.id_speed_dict.append({self.list_clients[i]:self.speeds[i]})
                    print(f"id speed list = {self.id_speed_dict}")
                    print(f"Algorithm name {algorithm_name}")
                    self.send_infor()
            elif algorithm_name == "hicsfl":
                if len(self.list_clients) == self.total_clients:
                    for i in range(total_clients):
                        self.id_speed_dict.append({self.list_clients[i]:self.speeds[i]})
                    print(f"id speed list = {self.id_speed_dict}")
                    print(f"Algorithm name {algorithm_name}")
                    src.Log.print_with_color("All clients are connected. Sending notifications.", "green")
                    src.Log.print_with_color(f"Zone_distribution {self.zone_distribution}", "green")
                    src.Log.print_with_color(f"Start training round {self.num_round - self.round + 1}", "yellow")
                    self.notify_clients()
            elif algorithm_name == "feel":
                if len(self.list_clients) == self.total_clients:
                    for i in range(total_clients):
                        self.id_speed_dict.append({self.list_clients[i]:self.speeds[i]})
                    print(f"id speed list = {self.id_speed_dict}")
                    print(f"Algorithm name {algorithm_name}")
                    src.Log.print_with_color("All clients are connected. Sending notifications.", "green")
                    src.Log.print_with_color(f"Zone_distribution {self.zone_distribution}", "green")
                    src.Log.print_with_color(f"Start training round {self.num_round - self.round + 1}", "yellow")
                    self.notify_clients()

        elif action == "UPDATE":
            data_message = message["message"]
            result = message["result"]
            src.Log.print_with_color(f"[<<<] Received message from client: {data_message}", "blue")
            self.updated_clients += 1
            # Save client's model parameters
            if not result:
                self.round_result = False
            if algorithm_name == "our":
                if save_parameters and self.round_result:
                    model_state_dict = message["parameters"]
                    client_size = message["size"]
                    num_data = message["num_data"]
                    training_time = message["training_time"]
                    speed = message["speed"]
                    self.local_speed.append(speed)
                    self.training_time.append(training_time)    
                    self.all_model_parameters.append({'client_id': client_id, 'weight': model_state_dict,
                                                    'size': client_size, "num_data":num_data})
                    src.Log.print_with_color(f"Data nhan lai: {client_id},{num_data}", "yellow")
                if self.cluster_state:
                    weight = src.Utils.extract_all_linear_weights(model_state_dict)
                    self.client_vectors.append(weight)

                # If consumed all client's parameters
                if self.updated_clients == len(self.selected_client):
                    if self.cluster_state:
                        self.client_vectors = np.stack(self.client_vectors)        
                        self.client_selection()
                        self.selected_client_temp = self.selected_client
                        self.cluster_state = False
                        self.client_vectors = []
                        self.process_consumer()          
                    else:    
                        self.process_consumer()
            elif algorithm_name == "csfedavg":
                if save_parameters and self.round_result:
                    model_state_dict = message["parameters"]
                    client_size = message["size"]
                    num_data = message["num_data"]
                    training_time = message["training_time"]
                    speed = message["speed"]
                    self.local_speed.append(speed)
                    self.training_time.append(training_time)  
                    self.all_model_parameters.append({'client_id': client_id, 'weight': model_state_dict,
                                                    'size': client_size, "num_data":num_data})
                    src.Log.print_with_color(f"Data nhan lai: {client_id},{num_data}", "yellow")
                    state_dict_client = copy.deepcopy(model_state_dict)
                    divergence = src.Utils.compute_divergence(state_dict_client, self.state_dict_server)
                    self.divergence_scores[self.count_i] = divergence
                    self.list_i.append(self.count_i)
                    self.count_i+=1

                # If consumed all client's parameters
                if self.updated_clients == len(self.selected_client): 

                    local_speeds = self.speeds[:len(self.list_clients)]
                    num_datas = [param["num_data"] for param in self.all_model_parameters]
                    print(f"Num_datas: {num_datas}")
                    total_training_time = np.array(num_datas) / np.array(local_speeds)
                    # From client selected, calculate and log training time
                    training_time = np.max([total_training_time[i] for i in self.selected_client])
                    self.logger.log_info(f"Active with {len(self.selected_client)} client: {self.selected_client}")
                    self.logger.log_info(f"Total training time round = {training_time}")
                    
                    sorted_divergence_scores = sorted(self.list_i, key=lambda i: self.divergence_scores[i])
                    selected_index = sorted_divergence_scores[:max(1,int(alpha*len(self.selected_client)))]
                    self.all_model_parameters = [self.all_model_parameters[i] for i in selected_index]
                    self.divergence_scores = {}
                    self.list_i = []
                    self.count_i = 0
                    self.process_consumer()
            elif algorithm_name == "fedcls":
                if save_parameters and self.round_result:
                    model_state_dict = message["parameters"]
                    client_size = message["size"]
                    num_data = message["num_data"]
                    training_time = message["training_time"]
                    speed = message["speed"]
                    self.local_speed.append(speed)
                    self.training_time.append(training_time)  
                    self.all_model_parameters.append({'client_id': client_id, 'weight': model_state_dict,
                                                    'size': client_size, "num_data":num_data})
                    src.Log.print_with_color(f"Data nhan lai: {client_id},{num_data}", "yellow")
                if self.updated_clients == len(self.selected_client):
                    self.process_consumer()
            elif algorithm_name =="fedrhlp":               
                if save_parameters and self.round_result:
                    model_state_dict = message["parameters"]
                    client_size = message["size"]
                    num_data = message["num_data"]
                    training_time = message["training_time"]
                    speed = message["speed"]
                    self.local_speed.append(speed)
                    self.training_time.append(training_time)  
                    self.all_model_parameters.append({'client_id': client_id, 'weight': model_state_dict,
                                                    'size': client_size, "num_data":num_data})
                    src.Log.print_with_color(f"Data nhan lai: {client_id},{num_data}", "yellow")
                if self.updated_clients == len(self.selected_client):
                    self.process_consumer()
            elif algorithm_name == "haccs":
                if save_parameters and self.round_result:
                    model_state_dict = message["parameters"]
                    client_size = message["size"]
                    num_data = message["num_data"]
                    training_time = message["training_time"]
                    speed = message["speed"]
                    self.local_speed.append(speed)
                    self.training_time.append(training_time)  
                    self.all_model_parameters.append({'client_id': client_id, 'weight': model_state_dict,
                                                    'size': client_size, "num_data":num_data})
                    src.Log.print_with_color(f"Data nhan lai: {client_id},{num_data}", "yellow")
                if self.updated_clients == len(self.selected_client):
                    self.process_consumer()
            elif algorithm_name == "hicsfl":
                if save_parameters and self.round_result:
                    model_state_dict = message["parameters"]
                    client_size = message["size"]
                    num_data = message["num_data"]
                    bias = message["bias"]
                    training_time = message["training_time"]
                    speed = message["speed"]
                    self.local_speed.append(speed)
                    self.training_time.append(training_time)  
                    self.client_biases.append(bias)
                    if str(client_id) not in self.list_clients:
                        self.list_clients.append(str(client_id))
                        src.Log.print_with_color(f"[<<<] Received message from client: {client_id}", "blue")
                    self.all_model_parameters.append({'client_id': client_id, 'weight': model_state_dict,
                                                    'size': client_size, "num_data":num_data})
                    src.Log.print_with_color(f"Data nhan lai: {client_id},{num_data}", "yellow")
                if self.updated_clients == len(self.selected_client):
                    print(f"Client biases= {self.client_biases}")
                    bias_matrix = np.stack(self.client_biases)
                    entropies = -np.sum(bias_matrix * np.log(bias_matrix + 1e-6), axis=1)
                    clusters = KMeans(n_clusters=num_cluster, random_state=0).fit_predict(bias_matrix)
                    cluster_scores = np.zeros(num_cluster)
                    print(f"bias_matrix= {bias_matrix}")
                    print(f"entropies= {entropies}")
                    print(f"clusters= {clusters}")
                    for i in range(num_cluster):
                        cluster_scores[i] = entropies[clusters == i].mean()
                    cluster_probs = cluster_scores / cluster_scores.sum()
                    self.selected_client = []
                    while len(self.selected_client) < CLIENTS_PER_ROUND:
                        chosen_cluster = np.random.choice(num_cluster, p=cluster_probs)
                        candidates = np.where(clusters == chosen_cluster)[0]
                        chosen_client = np.random.choice(candidates)
                        self.selected_client.append(chosen_client)
                    list(self.selected_client)
                    print(f"CLIENTS_PER_ROUND: {CLIENTS_PER_ROUND}")
                    print(f"len selected client {self.selected_client}")
                    self.all_model_parameters = [self.all_model_parameters[i] for i in self.selected_client]
                    print(f"length all_model_parameters {len(self.all_model_parameters)} ")
                    self.client_biases = []
                    self.process_consumer()
            elif algorithm_name == "feel":
                if save_parameters and self.round_result:
                    model_state_dict = message["parameters"]
                    client_size = message["size"]
                    num_data = message["num_data"]
                    reward = message["reward"]
                    actions = message["actions"]
                    value = message["value"]
                    log_prob = message["log_prob"]
                    selected_index = message["selected_index"]
                    training_time = message["training_time"]
                    speed = message["speed"]
                    self.local_speed.append(speed)
                    self.training_time.append(training_time)  
                    print("selected_index =", selected_index)
                    print("len(self.rewards) =", len(self.rewards))
                
                    self.rewards[selected_index] = reward
                    self.actions[selected_index] =(actions)
                    self.values[selected_index] =(value)
                    self.old_log_probs[selected_index] =(log_prob)

                    self.all_model_parameters.append({'client_id': client_id, 'weight': model_state_dict,
                                                    'size': client_size, "num_data":num_data})
                    src.Log.print_with_color(f"Data nhan lai: {client_id},{num_data}", "yellow")
                if self.updated_clients == len(self.selected_client):
                    self.rewards = torch.tensor(self.rewards)
                    self.actions = torch.stack(self.actions)
                    self.values = torch.stack(self.values)
                    self.old_log_probs = torch.stack(self.old_log_probs)

                    new_logits = self.voi_estimator.get_action(self.states_tensor[self.selected_client])
                    new_log_probs = -((actions - new_logits) ** 2)
                    advantages = self.rewards - self.values.detach()
                    ratios = torch.exp(new_log_probs - self.old_log_probs)

                    surr1 = ratios * advantages
                    surr2 = torch.clamp(ratios, 0.8, 1.2) * advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = nn.MSELoss()(self.voi_estimator.get_value(self.states_tensor[self.selected_client]), self.rewards)
                    loss = policy_loss + 0.5 * value_loss

                    self.voi_optimizer.zero_grad()
                    loss.backward()
                    self.voi_optimizer.step()

                    self.process_consumer()


        elif action == "UPDATE-INFOR":
            label_count = message["label_counts"]
            num_labels = message["num_labels"]
            if str(client_id) not in self.list_clients:
                self.list_clients.append(str(client_id))
                self.label_counts.append(label_count)
                src.Log.print_with_color(f"[<<<] Received message from client: {message}", "blue")
            if algorithm_name == "fedcls":
                if len(self.list_clients) == self.total_clients:
                    clusters, cluster_labels = src.Utils.fedcls_cluster_clients(self.label_counts, num_labels)
                    selected_clusters = src.Utils.fedcls_select_clusters(cluster_labels, num_labels)
                    for cid in selected_clusters:
                        pool = clusters[cid]  # danh sách client trong cluster cid
                        k = max(1, math.ceil(len(pool) * ratio))  # luôn chọn ≥1 client
                        k = min(k, len(pool))  # không vượt quá số client hiện có
                        self.selected_client.extend(random.sample(pool, k=k))
                    self.label_counts = []
                    self.notify_clients()
            elif algorithm_name == "haccs":
                if len(self.list_clients) == self.total_clients:
                    self.label_counts = np.array(self.label_counts)
                    hists = self.label_counts / self.label_counts.sum(axis=1, keepdims=True)
                    dist_matrix = pairwise_distances(hists, metric=src.Utils.hellinger_dist)
                    cluster_labels = OPTICS(metric='precomputed', min_samples = 5).fit_predict(dist_matrix)
                    latencies = np.random.uniform(0.5, 5.0, size=total_clients)
                    self.label_counts=[]
                    self.selected_client = src.Utils.HACCS_select_clients(cluster_labels, latencies)
                    self.selected_client = list(self.selected_client)
                    self.notify_clients()
        elif action == "UPDATE-fedrhlp":
            size_label = message["size_label"]
            self.size_label.append(size_label)
            if str(client_id) not in self.list_clients:
                self.list_clients.append(str(client_id))
                src.Log.print_with_color(f"[<<<] Received message from client: {message}", "blue")
            if len(self.list_clients) == self.total_clients:
                n = len(self.size_label)
                scores = np.array([s * l for s, l in self.size_label])
                self.size_label = []
                probs1 = scores / scores.sum()
                m1 = max(1, int(c1 * n))  # số client cần chọn
                self.selected_client = np.random.choice(n, m1, replace=False, p=probs1)
                self.selected_client = list(self.selected_client)
                for i in self.selected_client:
                    client_id = self.list_clients[i]
                    src.Log.print_with_color(f"[>>>] Sent start training request to client {client_id}", "red")
                    response = {"action": "START-RHLP","message": "UPDATE-fedrhlp"}
                    self.send_to_response(client_id, pickle.dumps(response))
        elif action == "UPDATE-feel":
            print("nhan goi tin update-feel")
            size = message["size"]
            staleness = message["staleness"]
            age = message["age"]
            self.size.append(size)
            self.list_staleness.append(staleness)
            self.list_age.append(age)
            self.stats.append(torch.tensor([0.0, size, staleness, age], dtype=torch.float32))

            if str(client_id) not in self.list_clients:
                self.list_clients.append(str(client_id))
                src.Log.print_with_color(f"[<<<] Received message from client: {message}", "blue")
            
            if len(self.stats) == self.total_clients:
                self.states_tensor = torch.stack(self.stats)

                logit = self.voi_estimator.get_action(self.states_tensor)
                k = max(1, int(ratio_feel * total_clients))
                self.selected_client = torch.topk(logit, k=k).indices.tolist()
                
                self.rewards = [0 for i in range(len(self.selected_client))]
                self.actions = [0 for i in range(len(self.selected_client))]
                self.values = [0 for i in range(len(self.selected_client))]
                self.old_log_probs = [0 for i in range(len(self.selected_client))]
                voi_state_dict = self.voi_estimator.state_dict()
                for i in self.selected_client:
                    client_id = self.list_clients[i]
                    logits = logit[i]
                    states_tensor = self.states_tensor[i]
                    stats = self.stats[i]
                    selected_index = self.selected_client.index(i)
                    print(f"len selected client {len(self.selected_client)}")
                    print(selected_index)
                    src.Log.print_with_color(f"[>>>] Sent start training request to client {client_id}", "red")
                    response = {"action": "START-feel","message": "UPDATE-feel","logits": logits, "state_tensor":states_tensor, "selected_flag": True, "stats": stats, "selected_index": selected_index, "voi_state_dict": voi_state_dict }

                    self.send_to_response(client_id, pickle.dumps(response))

        # Ack the message
        ch.basic_ack(delivery_tag=method.delivery_tag)

    def process_consumer(self):
        """
        After collect all training clients, start validation and make decision for the next training round
        :return:
        """
        if algorithm_name == "our":
            self.updated_clients = 0
            src.Log.print_with_color("Collected all parameters.", "yellow")
            # TODO: detect model poisoning with self.all_model_parameters at here
            if save_parameters and self.round_result:
                self.avg_selected_parameters()
                self.all_model_parameters = []
            # Server validation
            accuracy = 0.0
            if save_parameters and validation and self.round_result:
                if  data_name == "DOMAIN2" or data_name == "DOMAIN":
                    self.round_result,loss, accuracy,precision,recall,f1 = self.validation.test(self.avg_state_dict, device)
                    infor = f"Round {self.num_round - self.round + 1}/{self.num_round} with {len(self.selected_client)} client(s). Final Result: loss: {loss:.4f}, accuracy: {accuracy:.4f}, precision: {precision:.4f},recall: {recall:.4f},f1: {f1:.4f} with training time: {max(self.training_time)}"
                    print(infor)
                    self.logger.log_info(infor)

                elif data_name == "CIFAR10":
                    self.round_result,loss, accuracy = self.validation.test(self.avg_state_dict, device)
                    infor = f"Round {self.num_round - self.round + 1}/{self.num_round} with {len(self.selected_client)} client(s). Final Result: loss: {loss:.4f} ,accuracy: {accuracy:.4f} with training time: {max(self.training_time)}"
                    self.logger.log_info(infor)
                    print(infor)

            if not self.round_result:
                src.Log.print_with_color(f"Training failed!", "yellow")
                send_mail(email_config, f"Quá trình training bị lỗi tại round {self.num_round - self.round + 1}")
                if stop_when_false:
                    # Stop training
                    self.notify_clients(start=False)
                    delete_old_queues()
                    sys.exit()
            elif self.last_accuracy - accuracy > accuracy_drop:
                src.Log.print_with_color(f"Accuracy drop!", "yellow")
            else:
                self.last_accuracy = accuracy
                # Save to files
                torch.save(self.avg_state_dict, f'{model_name}.pth')
                self.round -= 1
            self.round_result = True

            if self.round > 0:
                # Start a new training round
                src.Log.print_with_color(f"Start training round {self.num_round - self.round + 1}", "yellow")
                #self.data_distribution()
                #self.client_selection()
                self.local_speed = []
                self.training_time = []
                self.notify_clients()
            else:
                # Stop training
                send_mail(email_config, f"Đã hoàn thành quá trình training")
                self.notify_clients(start=False)
                delete_old_queues()
                sys.exit()
        elif algorithm_name == "csfedavg":
            self.updated_clients = 0
            src.Log.print_with_color("Collected all parameters.", "yellow")
            # TODO: detect model poisoning with self.all_model_parameters at here
            if save_parameters and self.round_result:
                training_time = max(self.training_time)
                print(f"Training time = {training_time}")
                self.avg_selected_parameters()
                self.all_model_parameters = []
            # Server validation
            accuracy = 0.0
            if save_parameters and validation and self.round_result:
                if  data_name == "DOMAIN2" or data_name == "DOMAIN":
                    self.round_result,loss, accuracy,precision,recall,f1 = self.validation.test(self.avg_state_dict, device)
                    infor = f"Round {self.num_round - self.round + 1}/{self.num_round} with {len(self.selected_client)} client(s). Final Result: loss: {loss:.4f}, accuracy: {accuracy:.4f}, precision: {precision:.4f},recall: {recall:.4f},f1: {f1:.4f} with training time: {max(self.training_time)}"
                    print(infor)
                    self.logger.log_info(infor)

                elif data_name == "CIFAR10":
                    self.round_result,loss, accuracy = self.validation.test(self.avg_state_dict, device)
                    infor = f"Round {self.num_round - self.round + 1}/{self.num_round} with {len(self.selected_client)} client(s). Final Result: loss: {loss:.4f} ,accuracy: {accuracy:.4f} with training time: {max(self.training_time)}"
                    self.logger.log_info(infor)
                    print(infor)


            if not self.round_result:
                src.Log.print_with_color(f"Training failed!", "yellow")
                send_mail(email_config, f"Quá trình training bị lỗi tại round {self.num_round - self.round + 1}")
                if stop_when_false:
                    # Stop training
                    self.notify_clients(start=False)
                    delete_old_queues()
                    sys.exit()
            elif self.last_accuracy - accuracy > accuracy_drop:
                src.Log.print_with_color(f"Accuracy drop!", "yellow")
            else:
                self.last_accuracy = accuracy
                # Save to files
                torch.save(self.avg_state_dict, f'{model_name}.pth')
                self.round -= 1
            self.round_result = True

            if self.round > 0:
                # Start a new training round
                src.Log.print_with_color(f"Start training round {self.num_round - self.round + 1}", "yellow")
                #self.data_distribution()
                self.local_speed = []
                self.training_time = []
                self.client_selection()
                self.notify_clients()
            else:
                # Stop training
                send_mail(email_config, f"Đã hoàn thành quá trình training")
                self.notify_clients(start=False)
                delete_old_queues()
                sys.exit()
        elif algorithm_name == "fedcls":
            self.updated_clients = 0
            src.Log.print_with_color("Collected all parameters.", "yellow")
            # TODO: detect model poisoning with self.all_model_parameters at here
            if save_parameters and self.round_result:
                training_time = max(self.training_time)
                print(f"Training time = {training_time}")
                self.avg_selected_parameters()
                self.all_model_parameters = []
            # Server validation
            accuracy = 0.0
            if save_parameters and validation and self.round_result:
                if  data_name == "DOMAIN2" or data_name == "DOMAIN":
                    self.round_result,loss, accuracy,precision,recall,f1 = self.validation.test(self.avg_state_dict, device)
                    infor = f"Round {self.num_round - self.round + 1}/{self.num_round} with {len(self.selected_client)} client(s). Final Result: loss: {loss:.4f}, accuracy: {accuracy:.4f}, precision: {precision:.4f},recall: {recall:.4f},f1: {f1:.4f} with training time: {max(self.training_time)}"
                    print(infor)
                    self.logger.log_info(infor)

                elif data_name == "CIFAR10":
                    self.round_result,loss, accuracy = self.validation.test(self.avg_state_dict, device)
                    infor = f"Round {self.num_round - self.round + 1}/{self.num_round} with {len(self.selected_client)} client(s). Final Result: loss: {loss:.4f} ,accuracy: {accuracy:.4f} with training time: {max(self.training_time)}"
                    self.logger.log_info(infor)
                    print(infor)

            if not self.round_result:
                src.Log.print_with_color(f"Training failed!", "yellow")
                send_mail(email_config, f"Quá trình training bị lỗi tại round {self.num_round - self.round + 1}")
                if stop_when_false:
                    # Stop training
                    self.notify_clients(start=False)
                    delete_old_queues()
                    sys.exit()
            elif self.last_accuracy - accuracy > accuracy_drop:
                src.Log.print_with_color(f"Accuracy drop!", "yellow")
            else:
                self.last_accuracy = accuracy
                # Save to files
                torch.save(self.avg_state_dict, f'{model_name}.pth')
                self.round -= 1
            self.round_result = True

            if self.round > 0:
                # Start a new training round
                self.local_speed = []
                self.training_time = []
                src.Log.print_with_color(f"Start training round {self.num_round - self.round + 1}", "yellow")
                self.selected_client = []
                self.send_infor()
            else:
                # Stop training
                send_mail(email_config, f"Đã hoàn thành quá trình training")
                self.notify_clients(start=False)
                delete_old_queues()
                sys.exit()
        elif algorithm_name == "fedrhlp":
            self.updated_clients = 0
            src.Log.print_with_color("Collected all parameters.", "yellow")
            # TODO: detect model poisoning with self.all_model_parameters at here
            if save_parameters and self.round_result:
                training_time = max(self.training_time)
                print(f"Training time = {training_time}")
                self.avg_selected_parameters()
                self.all_model_parameters = []
            # Server validation
            accuracy = 0.0
            if save_parameters and validation and self.round_result:
                if  data_name == "DOMAIN2" or data_name == "DOMAIN":
                    self.round_result,loss, accuracy,precision,recall,f1 = self.validation.test(self.avg_state_dict, device)
                    infor = f"Round {self.num_round - self.round + 1}/{self.num_round} with {len(self.selected_client)} client(s). Final Result: loss: {loss:.4f}, accuracy: {accuracy:.4f}, precision: {precision:.4f},recall: {recall:.4f},f1: {f1:.4f} with training time: {max(self.training_time)}"
                    print(infor)
                    self.logger.log_info(infor)

                elif data_name == "CIFAR10":
                    self.round_result,loss, accuracy = self.validation.test(self.avg_state_dict, device)
                    infor = f"Round {self.num_round - self.round + 1}/{self.num_round} with {len(self.selected_client)} client(s). Final Result: loss: {loss:.4f} ,accuracy: {accuracy:.4f} with training time: {max(self.training_time)}"
                    self.logger.log_info(infor)
                    print(infor)


            if not self.round_result:
                src.Log.print_with_color(f"Training failed!", "yellow")
                send_mail(email_config, f"Quá trình training bị lỗi tại round {self.num_round - self.round + 1}")
                if stop_when_false:
                    # Stop training
                    self.notify_clients(start=False)
                    delete_old_queues()
                    sys.exit()
            elif self.last_accuracy - accuracy > accuracy_drop:
                src.Log.print_with_color(f"Accuracy drop!", "yellow")
            else:
                self.last_accuracy = accuracy
                # Save to files
                torch.save(self.avg_state_dict, f'{model_name}.pth')
                self.round -= 1
            self.round_result = True

            if self.round > 0:
                self.local_speed = []
                self.training_time = []
                self.selected_client = []
                src.Log.print_with_color(f"Start training round {self.num_round - self.round + 1}", "yellow")
                self.notify_clients()
            else:
                # Stop training
                send_mail(email_config, f"Đã hoàn thành quá trình training")
                self.notify_clients(start=False)
                delete_old_queues()
                sys.exit()
        elif algorithm_name == "haccs":
            self.updated_clients = 0
            src.Log.print_with_color("Collected all parameters.", "yellow")
            # TODO: detect model poisoning with self.all_model_parameters at here
            if save_parameters and self.round_result:
                training_time = max(self.training_time)
                print(f"Training time = {training_time}")
                self.avg_selected_parameters()
                self.all_model_parameters = []
            # Server validation
            accuracy = 0.0
            if save_parameters and validation and self.round_result:
                if  data_name == "DOMAIN2" or data_name == "DOMAIN":
                    self.round_result,loss, accuracy,precision,recall,f1 = self.validation.test(self.avg_state_dict, device)
                    infor = f"Round {self.num_round - self.round + 1}/{self.num_round} with {len(self.selected_client)} client(s). Final Result: loss: {loss:.4f}, accuracy: {accuracy:.4f}, precision: {precision:.4f},recall: {recall:.4f},f1: {f1:.4f} with training time: {max(self.training_time)}"
                    print(infor)
                    self.logger.log_info(infor)

                elif data_name == "CIFAR10":
                    self.round_result,loss, accuracy = self.validation.test(self.avg_state_dict, device)
                    infor = f"Round {self.num_round - self.round + 1}/{self.num_round} with {len(self.selected_client)} client(s). Final Result: loss: {loss:.4f} ,accuracy: {accuracy:.4f} with training time: {max(self.training_time)}"
                    self.logger.log_info(infor)
                    print(infor)


            if not self.round_result:
                src.Log.print_with_color(f"Training failed!", "yellow")
                send_mail(email_config, f"Quá trình training bị lỗi tại round {self.num_round - self.round + 1}")
                if stop_when_false:
                    # Stop training
                    self.notify_clients(start=False)
                    delete_old_queues()
                    sys.exit()
            elif self.last_accuracy - accuracy > accuracy_drop:
                src.Log.print_with_color(f"Accuracy drop!", "yellow")
            else:
                self.last_accuracy = accuracy
                # Save to files
                torch.save(self.avg_state_dict, f'{model_name}.pth')
                self.round -= 1
            self.round_result = True

            if self.round > 0:
                # Start a new training round
                self.local_speed = []
                self.training_time = []
                src.Log.print_with_color(f"Start training round {self.num_round - self.round + 1}", "yellow")
                self.selected_client = []
                self.send_infor()
            else:
                # Stop training
                send_mail(email_config, f"Đã hoàn thành quá trình training")
                self.notify_clients(start=False)
                delete_old_queues()
                sys.exit()
        elif algorithm_name == "hicsfl":
            self.updated_clients = 0
            src.Log.print_with_color("Collected all parameters.", "yellow")
            # TODO: detect model poisoning with self.all_model_parameters at here
            if save_parameters and self.round_result:
                training_time = max(self.training_time)
                print(f"Training time = {training_time}")
                self.avg_selected_parameters()
                self.all_model_parameters = []
            # Server validation
            accuracy = 0.0
            if save_parameters and validation and self.round_result:
                if  data_name == "DOMAIN2" or data_name == "DOMAIN":
                    self.round_result,loss, accuracy,precision,recall,f1 = self.validation.test(self.avg_state_dict, device)
                    infor = f"Round {self.num_round - self.round + 1}/{self.num_round} with {len(self.selected_client)} client(s). Final Result: loss: {loss:.4f}, accuracy: {accuracy:.4f}, precision: {precision:.4f},recall: {recall:.4f},f1: {f1:.4f} with training time: {max(self.training_time)}"
                    print(infor)
                    self.logger.log_info(infor)

                elif data_name == "CIFAR10":
                    self.round_result,loss, accuracy = self.validation.test(self.avg_state_dict, device)
                    infor = f"Round {self.num_round - self.round + 1}/{self.num_round} with {len(self.selected_client)} client(s). Final Result: loss: {loss:.4f} ,accuracy: {accuracy:.4f} with training time: {max(self.training_time)}"
                    self.logger.log_info(infor)
                    print(infor)

            if not self.round_result:
                src.Log.print_with_color(f"Training failed!", "yellow")
                send_mail(email_config, f"Quá trình training bị lỗi tại round {self.num_round - self.round + 1}")
                if stop_when_false:
                    # Stop training
                    self.notify_clients(start=False)
                    delete_old_queues()
                    sys.exit()
            elif self.last_accuracy - accuracy > accuracy_drop:
                src.Log.print_with_color(f"Accuracy drop!", "yellow")
            else:
                self.last_accuracy = accuracy
                # Save to files
                torch.save(self.avg_state_dict, f'{model_name}.pth')
                self.round -= 1
            self.round_result = True

            if self.round > 0:
                # Start a new training round
                self.local_speed = []
                self.training_time = []
                src.Log.print_with_color(f"Start training round {self.num_round - self.round + 1}", "yellow")
                self.selected_client = []
                self.notify_clients()
            else:
                # Stop training
                send_mail(email_config, f"Đã hoàn thành quá trình training")
                self.notify_clients(start=False)
                delete_old_queues()
                sys.exit()
        elif algorithm_name == "feel":
            self.updated_clients = 0
            src.Log.print_with_color("Collected all parameters.", "yellow")
            # TODO: detect model poisoning with self.all_model_parameters at here
            if save_parameters and self.round_result:
                training_time = max(self.training_time)
                print(f"Training time = {training_time}")
                self.avg_selected_parameters()
                self.all_model_parameters = []
            # Server validation
            accuracy = 0.0
            if save_parameters and validation and self.round_result:
                if  data_name == "DOMAIN2" or data_name == "DOMAIN":
                    self.round_result,loss, accuracy,precision,recall,f1 = self.validation.test(self.avg_state_dict, device)
                    infor = f"Round {self.num_round - self.round + 1}/{self.num_round} with {len(self.selected_client)} client(s). Final Result: loss: {loss:.4f}, accuracy: {accuracy:.4f}, precision: {precision:.4f},recall: {recall:.4f},f1: {f1:.4f} with training time: {max(self.training_time)}"
                    print(infor)
                    self.logger.log_info(infor)

                elif data_name == "CIFAR10":
                    self.round_result,loss, accuracy = self.validation.test(self.avg_state_dict, device)
                    infor = f"Round {self.num_round - self.round + 1}/{self.num_round} with {len(self.selected_client)} client(s). Final Result: loss: {loss:.4f} ,accuracy: {accuracy:.4f} with training time: {max(self.training_time)}"
                    self.logger.log_info(infor)
                    print(infor)


            if not self.round_result:
                src.Log.print_with_color(f"Training failed!", "yellow")
                send_mail(email_config, f"Quá trình training bị lỗi tại round {self.num_round - self.round + 1}")
                if stop_when_false:
                    # Stop training
                    self.notify_clients(start=False)
                    delete_old_queues()
                    sys.exit()
            elif self.last_accuracy - accuracy > accuracy_drop:
                src.Log.print_with_color(f"Accuracy drop!", "yellow")
            else:
                self.last_accuracy = accuracy
                # Save to files
                torch.save(self.avg_state_dict, f'{model_name}.pth')
                self.round -= 1
            self.round_result = True

            if self.round > 0:
                # Start a new training round
                src.Log.print_with_color(f"Start training round {self.num_round - self.round + 1}", "yellow")
                self.local_speed = []
                self.training_time = []
                self.size = []
                self.list_staleness = []
                self.list_age = []
                self.stats = []
                self.selected_client = []
                self.notify_clients()
            else:
                # Stop training
                send_mail(email_config, f"Đã hoàn thành quá trình training")
                self.notify_clients(start=False)
                delete_old_queues()
                sys.exit()
            
    def notify_clients(self, start=True):
        """
        Control message to clients
        :param start: If True (default), request clients to start. Else if False, stop training
        :return:
        """
        # Send message to clients when consumed all clients
        if start:
            filepath = f'{model_name}.pth'
            # Read parameters file
            state_dict = None
            if load_parameters:
                if os.path.exists(filepath):
                    state_dict = torch.load(filepath, weights_only=True)
            #count_labels = np.zeros(num_labels)
            if algorithm_name == "our":
                epoch = 1
                if state_dict is None:
                    if data_name == "DOMAIN2":
                        if model_name == "Transformer":
                            model = src.Model.PositionalEncodingTransformer().to(device)
                        elif model_name == "CNN":
                            model = src.Model.CNNClassifier().to(device)
                        else:
                            raise ValueError(f"Model name '{model_name}' is not valid.")
                    elif data_name == "CIFAR10":
                        if model_name == "ResNet":
                            model = src.Model.ResNet18().to(device)
                        elif model_name == "MobileNet":
                            model = src.Model.MobileNetV2().to(device)
                        else:
                            raise ValueError(f"Model name '{model_name}' is not valid.")
                    else:
                        raise ValueError(f"Data name '{data_name}' is not valid.")
                    state_dict = model.state_dict()
                self.state_dict_server = state_dict
                if  (self.round-self.num_round) % cluster_iter == 0:
                    self.cluster_state = True
                    epoch = 5
                    self.selected_client = [_ for _ in range(total_clients)]
                    for i in self.selected_client:
                        client_id = self.list_clients[i]
                        speed = src.Utils.find_speed(client_id,self.id_speed_dict)
                        print(f"speed: {speed}")
                        # Request clients to start training
                        src.Log.print_with_color(f"[>>>] Sent start training request to client {client_id}", "red")
                        response = {"action": "START",
                                    "message": "Server accept the connection!",
                                    "model_name": model_name,
                                    "data_name": data_name,
                                    "parameters": state_dict,
                                    "batch_size": batch_size,
                                    "lr": lr,
                                    "momentum": momentum,
                                    "clip_grad_norm": clip_grad_norm,
                                    "range[0]": data_range[0],
                                    "range[1]": data_range[1],
                                    "data_mode": data_mode,
                                    "epoch": epoch,
                                    "algorithm_name": None,
                                    "speed": speed}
                        #count_labels += self.label_counts[i]
                        self.send_to_response(client_id, pickle.dumps(response))
                else:
                    self.selected_client = self.selected_client_temp
                    for i in self.selected_client:
                        client_id = self.all_model_parameters_temp[i]["client_id"]
                        speed = src.Utils.find_speed(str(client_id),self.id_speed_dict)
                        print(f"speed: {speed}")
                        # Request clients to start training
                        src.Log.print_with_color(f"[>>>] Sent start training request to client {client_id}", "red")
                        response = {"action": "START",
                                    "message": "Server accept the connection!",
                                    "model_name": model_name,
                                    "data_name": data_name,
                                    "parameters": state_dict,
                                    "batch_size": batch_size,
                                    "lr": lr,
                                    "momentum": momentum,
                                    "clip_grad_norm": clip_grad_norm,
                                    "range[0]": data_range[0],
                                    "range[1]": data_range[1],
                                    "data_mode": data_mode,
                                    "epoch": 5,
                                    "algorithm_name": None,
                                    "speed": speed}
                        #count_labels += self.label_counts[i]
                        self.send_to_response(client_id, pickle.dumps(response))
            elif algorithm_name == "csfedavg":
                epoch = 1
                if state_dict is None:
                    if data_name == "DOMAIN2":
                        if model_name == "Transformer":
                            model = src.Model.PositionalEncodingTransformer().to(device)
                        elif model_name == "CNN":
                            model = src.Model.CNNClassifier().to(device)
                        else:
                            raise ValueError(f"Model name '{model_name}' is not valid.")
                    elif data_name == "CIFAR10":
                        if model_name == "ResNet":
                            model = src.Model.ResNet18().to(device)
                        elif model_name == "MobileNet":
                            model = src.Model.MobileNetV2().to(device)
                        else:
                            raise ValueError(f"Model name '{model_name}' is not valid.")
                    else:
                        raise ValueError(f"Data name '{data_name}' is not valid.")
                    state_dict = model.state_dict()
                self.state_dict_server = state_dict
                for i in self.selected_client:
                    client_id = self.list_clients[i]
                    speed = src.Utils.find_speed(client_id,self.id_speed_dict)
                    print(f"speed: {speed}")
                    # Request clients to start training
                    src.Log.print_with_color(f"[>>>] Sent start training request to client {client_id}", "red")
                    response = {"action": "START",
                                "message": "Server accept the connection!",
                                "model_name": model_name,
                                "data_name": data_name,
                                "parameters": state_dict,
                                "batch_size": batch_size,
                                "lr": lr,
                                "momentum": momentum,
                                "clip_grad_norm": clip_grad_norm,
                                "range[0]": data_range[0],
                                "range[1]": data_range[1],
                                "data_mode": data_mode,
                                "epoch": 5,
                                "algorithm_name": None,
                                "speed": speed}
                    self.send_to_response(client_id, pickle.dumps(response))
            elif algorithm_name == "fedcls":
                epoch = 1
                if state_dict is None:
                    if data_name == "DOMAIN2":
                        if model_name == "Transformer":
                            model = src.Model.PositionalEncodingTransformer().to(device)
                        elif model_name == "CNN":
                            model = src.Model.CNNClassifier().to(device)
                        else:
                            raise ValueError(f"Model name '{model_name}' is not valid.")
                    elif data_name == "CIFAR10":
                        if model_name == "ResNet":
                            model = src.Model.ResNet18().to(device)
                        elif model_name == "MobileNet":
                            model = src.Model.MobileNetV2().to(device)
                        else:
                            raise ValueError(f"Model name '{model_name}' is not valid.")
                    else:
                        raise ValueError(f"Data name '{data_name}' is not valid.")
                    state_dict = model.state_dict()
                self.state_dict_server = state_dict
                print(f"List client: {self.list_clients}")
                for i in self.selected_client:
                    print(f"i: {i}")
                    client_id = self.list_clients[i]
                    speed = src.Utils.find_speed(client_id,self.id_speed_dict)
                    print(f"speed: {speed}")
                    src.Log.print_with_color(f"[>>>] Sent start training request to client {client_id}", "red")
                    response = {"action": "START",
                                "message": "Server accept the connection!",
                                "model_name": model_name,
                                "data_name": data_name,
                                "parameters": state_dict,
                                "batch_size": batch_size,
                                "lr": lr,
                                "momentum": momentum,
                                "clip_grad_norm": clip_grad_norm,
                                "range[0]": data_range[0],
                                "range[1]": data_range[1],
                                "data_mode": data_mode,
                                "epoch": 5,
                                "algorithm_name": "fedcls",
                                "speed": speed}
                    self.send_to_response(client_id, pickle.dumps(response))
            elif algorithm_name == "fedrhlp":
                epoch = 1
                if state_dict is None:
                    if data_name == "DOMAIN2":
                        if model_name == "Transformer":
                            model = src.Model.PositionalEncodingTransformer().to(device)
                        elif model_name == "CNN":
                            model = src.Model.CNNClassifier().to(device)
                        else:
                            raise ValueError(f"Model name '{model_name}' is not valid.")
                    elif data_name == "CIFAR10":
                        if model_name == "ResNet":
                            model = src.Model.ResNet18().to(device)
                        elif model_name == "MobileNet":
                            model = src.Model.MobileNetV2().to(device)
                        else:
                            raise ValueError(f"Model name '{model_name}' is not valid.")
                    else:
                        raise ValueError(f"Data name '{data_name}' is not valid.")
                    state_dict = model.state_dict()
                self.state_dict_server = state_dict
                print(f"List client: {self.list_clients}")
                self.selected_client = [_ for _ in range(total_clients)]
                for i in self.selected_client:
                    print(f"i: {i}")
                    client_id = self.list_clients[i]
                    speed = src.Utils.find_speed(client_id,self.id_speed_dict)
                    print(f"speed: {speed}")
                    src.Log.print_with_color(f"[>>>] Sent start training request to client {client_id}", "red")
                    response = {"action": "START",
                                "message": "Server accept the connection!",
                                "model_name": model_name,
                                "data_name": data_name,
                                "parameters": state_dict,
                                "batch_size": batch_size,
                                "lr": lr,
                                "momentum": momentum,
                                "clip_grad_norm": clip_grad_norm,
                                "range[0]": data_range[0],
                                "range[1]": data_range[1],
                                "data_mode": data_mode,
                                "epoch": 5,
                                "algorithm_name": "fedrhlp",
                                "speed": speed}
                    self.send_to_response(client_id, pickle.dumps(response))
                self.list_clients = []
            elif algorithm_name == "haccs":
                epoch = 1
                if state_dict is None:
                    if data_name == "DOMAIN2":
                        if model_name == "Transformer":
                            model = src.Model.PositionalEncodingTransformer().to(device)
                        elif model_name == "CNN":
                            model = src.Model.CNNClassifier().to(device)
                        else:
                            raise ValueError(f"Model name '{model_name}' is not valid.")
                    elif data_name == "CIFAR10":
                        if model_name == "ResNet":
                            model = src.Model.ResNet18().to(device)
                        elif model_name == "MobileNet":
                            model = src.Model.MobileNetV2().to(device)
                        else:
                            raise ValueError(f"Model name '{model_name}' is not valid.")
                    else:
                        raise ValueError(f"Data name '{data_name}' is not valid.")
                    state_dict = model.state_dict()
                self.state_dict_server = state_dict
                print(f"List client: {self.list_clients}")
                for i in self.selected_client:
                    print(f"i: {i}")
                    client_id = self.list_clients[i]
                    speed = src.Utils.find_speed(client_id,self.id_speed_dict)
                    print(f"speed: {speed}")
                    src.Log.print_with_color(f"[>>>] Sent start training request to client {client_id}", "red")
                    response = {"action": "START",
                                "message": "Server accept the connection!",
                                "model_name": model_name,
                                "data_name": data_name,
                                "parameters": state_dict,
                                "batch_size": batch_size,
                                "lr": lr,
                                "momentum": momentum,
                                "clip_grad_norm": clip_grad_norm,
                                "range[0]": data_range[0],
                                "range[1]": data_range[1],
                                "data_mode": data_mode,
                                "epoch": 5,
                                "algorithm_name": "fedcls",
                                "speed": speed}
                    self.send_to_response(client_id, pickle.dumps(response))
            elif algorithm_name == "hicsfl":
                epoch = 1
                if state_dict is None:
                    if data_name == "DOMAIN2":
                        if model_name == "Transformer":
                            model = src.Model.PositionalEncodingTransformer().to(device)
                        elif model_name == "CNN":
                            model = src.Model.CNNClassifier().to(device)
                        else:
                            raise ValueError(f"Model name '{model_name}' is not valid.")
                    elif data_name == "CIFAR10":
                        if model_name == "ResNet":
                            model = src.Model.ResNet18().to(device)
                        elif model_name == "MobileNet":
                            model = src.Model.MobileNetV2().to(device)
                        else:
                            raise ValueError(f"Model name '{model_name}' is not valid.")
                    else:
                        raise ValueError(f"Data name '{data_name}' is not valid.")
                    state_dict = model.state_dict()
                self.state_dict_server = state_dict
                self.selected_client = [_ for _ in range(total_clients)]
                print(f"List client: {self.list_clients}")
                for i in self.selected_client:
                    print(f"i: {i}")
                    client_id = self.list_clients[i]
                    speed = src.Utils.find_speed(client_id,self.id_speed_dict)
                    print(f"speed: {speed}")
                    src.Log.print_with_color(f"[>>>] Sent start training request to client {client_id}", "red")
                    response = {"action": "START",
                                "message": "Server accept the connection!",
                                "model_name": model_name,
                                "data_name": data_name,
                                "parameters": state_dict,
                                "batch_size": batch_size,
                                "lr": lr,
                                "momentum": momentum,
                                "clip_grad_norm": clip_grad_norm,
                                "range[0]": data_range[0],
                                "range[1]": data_range[1],
                                "data_mode": data_mode,
                                "epoch": 5,
                                "algorithm_name": "hicsfl",
                                "speed": speed}
                    self.send_to_response(client_id, pickle.dumps(response))
                self.list_clients = []
            elif algorithm_name == "feel":
                epoch = 1
                if state_dict is None:
                    if data_name == "DOMAIN2":
                        if model_name == "Transformer":
                            model = src.Model.PositionalEncodingTransformer().to(device)
                        elif model_name == "CNN":
                            model = src.Model.CNNClassifier().to(device)
                        else:
                            raise ValueError(f"Model name '{model_name}' is not valid.")
                    elif data_name == "CIFAR10":
                        if model_name == "ResNet":
                            model = src.Model.ResNet18().to(device)
                        elif model_name == "MobileNet":
                            model = src.Model.MobileNetV2().to(device)
                        else:
                            raise ValueError(f"Model name '{model_name}' is not valid.")
                    else:
                        raise ValueError(f"Data name '{data_name}' is not valid.")
                    state_dict = model.state_dict()
                self.state_dict_server = state_dict
                self.selected_client = [_ for _ in range(total_clients)]
                print(f"List client: {self.list_clients}")
                for i in self.selected_client:
                    print(f"i: {i}")
                    client_id = self.list_clients[i]
                    speed = src.Utils.find_speed(client_id,self.id_speed_dict)
                    print(f"speed: {speed}")
                    src.Log.print_with_color(f"[>>>] Sent start training request to client {client_id}", "red")
                    response = {"action": "START",
                                "message": "Server accept the connection!",
                                "model_name": model_name,
                                "data_name": data_name,
                                "parameters": state_dict,
                                "batch_size": batch_size,
                                "lr": lr,
                                "momentum": momentum,
                                "clip_grad_norm": clip_grad_norm,
                                "range[0]": data_range[0],
                                "range[1]": data_range[1],
                                "data_mode": data_mode,
                                "epoch": 5,
                                "algorithm_name": "feel",
                                "speed": speed}
                    self.send_to_response(client_id, pickle.dumps(response))
                self.list_clients = []

        else:
            for client_id in self.list_clients:
                # Request clients to stop process
                src.Log.print_with_color(f"[>>>] Sent stop training request to client {client_id}", "red")
                response = {"action": "STOP",
                            "message": "Stop training!",
                            "parameters": None}
                self.send_to_response(client_id, pickle.dumps(response))

    def client_selection(self):
        """
        Select the specific clients
        :return: The list contain index of active clients: `self.selected_client`.
        E.g. `self.selected_client = [2,3,5]` means client 2, 3 and 5 will train this current round
        """
        if algorithm_name == "our":
            num_datas = [param["num_data"] for param in self.all_model_parameters]

            if client_selection_config['enable']:
                if client_cluster_config['enable']:
                    num_cluster, labels, _ = clustering_algorithm(self.client_vectors, client_cluster_config)
                    print(f"Num cluster = {num_cluster}, labels = {labels}")
                    self.logger.log_info(f"Num cluster = {num_cluster}, labels = {labels}")
                    self.selected_client_temp = []
                    self.all_model_parameters_temp = []
                    self.selected_client = []
                    for i in range(num_cluster):
                        cluster_client = [index for index, label in enumerate(labels) if label == i]
                        print(f"Cluster index{ cluster_client}")
                        if client_selection_config['mode'] == 'speed':
                            self.selected_client += client_selection_speed_base(cluster_client, self.local_speed, num_datas)
                            self.all_model_parameters_temp = self.all_model_parameters
                            print(f"selected_client: {self.selected_client}")
                        elif client_selection_config['mode'] == 'random':
                            self.selected_client += client_selection_random(cluster_client)
                else:
                    if client_selection_config['mode'] == 'speed':
                        self.selected_client = client_selection_speed_base([i for i in range(len(self.list_clients))],
                                                                        self.local_speed, num_datas)
                    elif client_selection_config['mode'] == 'random':
                        self.selected_client = client_selection_random([i for i in range(len(self.list_clients))])
                    elif client_selection_config['mode'] == 'all':
                        print(f"Fedavg")
                        self.selected_client_temp = self.selected_client
                        self.selected_client = [i for i in range(len(self.list_clients))]
                        self.all_model_parameters_temp = self.all_model_parameters
                        
            else:
                print(f"Fedavg")
                self.selected_client_temp = self.selected_client
                self.selected_client = [i for i in range(len(self.list_clients))]
                print("DEBUG: all_model_parameters =", self.all_model_parameters)
                self.all_model_parameters_temp = self.all_model_parameters
            

        
        elif algorithm_name == "csfedavg":
            if client_selection_config['enable']:
                if client_cluster_config['enable']:
                    num_cluster, labels, _ = clustering_algorithm(self.client_vectors, client_cluster_config)
                    print(f"Num cluster = {num_cluster}, labels = {labels}")
                    self.selected_client = []
                    for i in range(num_cluster):
                        cluster_client = [index for index, label in enumerate(labels) if label == i]
                        print(f"Cluster index{ cluster_client}")
                        if client_selection_config['mode'] == 'speed':
                            self.selected_client += client_selection_speed_base(cluster_client, self.local_speed, num_datas)
                            self.all_model_parameters_temp = self.all_model_parameters
                            print(f"selected_client: {self.selected_client}")
                        elif client_selection_config['mode'] == 'random':
                            self.selected_client += client_selection_random(cluster_client)
                else:
                    if client_selection_config['mode'] == 'speed':
                        self.selected_client = client_selection_speed_base([i for i in range(len(self.list_clients))],
                                                                        self.local_speed, num_datas)
                    elif client_selection_config['mode'] == 'random':
                        self.selected_client = random.sample(range(total_clients), int(p * total_clients))
            else:
                self.selected_client = [i for i in range(len(self.list_clients))]


    def send_infor(self):
        for i in range(self.total_clients):
            client_id = self.list_clients[i]
            # Request clients to start training
            src.Log.print_with_color(f"Request label count", "red")
            response = {"action": "INFOR",
                        "message": "I want label count",
                        "algorithm_name": "fedcls",
                        "data_name": data_name,
                        "range[0]": data_range[0],
                        "range[1]": data_range[1]}
            self.send_to_response(client_id, pickle.dumps(response))
        self.list_clients = []
    def avg_selected_parameters(self):
        if algorithm_name == "our":
            print(f"Avg: {len(self.selected_client)} client")
            if not self.selected_client:
                return

            # Danh sách các parameters tương ứng client đã chọn (đã thu thập đúng thứ tự)
            selected_params = [copy.deepcopy(self.all_model_parameters[i]) for i in range(len(self.selected_client))]

            # Lấy bản sao weights từ client đầu tiên trong danh sách selected_params
            avg_state_dict = copy.deepcopy(selected_params[0]["weight"])
            total_size = sum(p["num_data"] for p in selected_params)
            print(f"Total  size = {total_size}")
            for key in avg_state_dict.keys():
                if avg_state_dict[key].dtype != torch.long:
                    avg_state_dict[key] = sum(p["weight"][key] * p["num_data"] for p in selected_params) / total_size
                else:
                    avg_state_dict[key] = sum(p["weight"][key] * p["num_data"] for p in selected_params) // total_size

            self.avg_state_dict = copy.deepcopy(avg_state_dict)
        elif algorithm_name == "csfedavg":
            print(f"Avg: {len(self.selected_client)} client")
            if not self.selected_client:
                return
            print(f"length: len all_model_parameters: {len(self.all_model_parameters)}")
            # Danh sách các parameters tương ứng client đã chọn (đã thu thập đúng thứ tự)
            selected_params = [copy.deepcopy(self.all_model_parameters[i]) for i in range(len(self.selected_client))]
            print(f"length: len selected_params: {len(selected_params)}")
            # Lấy bản sao weights từ client đầu tiên trong danh sách selected_params
            avg_state_dict = copy.deepcopy(selected_params[0]["weight"])
            total_size = sum(p["num_data"] for p in selected_params)
            print(f"Total  size = {total_size}")
            for key in avg_state_dict.keys():
                if avg_state_dict[key].dtype != torch.long:
                    avg_state_dict[key] = sum(
                        p["weight"][key] * p["num_data"] for p in selected_params
                    ) / total_size
                else:
                    avg_state_dict[key] = sum(
                        p["weight"][key] * p["num_data"] for p in selected_params
                    ) // total_size

            self.avg_state_dict = copy.deepcopy(avg_state_dict)
        elif algorithm_name =="fedcls":
            print(f"Avg: {len(self.selected_client)} client")

            if not self.selected_client:
                return
            # Danh sách các parameters tương ứng client đã chọn (đã thu thập đúng thứ tự)
            selected_params = [copy.deepcopy(self.all_model_parameters[i]) for i in range(len(self.selected_client))]

            # Lấy bản sao weights từ client đầu tiên trong danh sách selected_params
            avg_state_dict = copy.deepcopy(selected_params[0]["weight"])
            total_size = sum(p["num_data"] for p in selected_params)
            print(f"Total  size = {total_size}")
            for key in avg_state_dict.keys():
                if avg_state_dict[key].dtype != torch.long:
                    avg_state_dict[key] = sum(
                        p["weight"][key] * p["num_data"] for p in selected_params
                    ) / total_size
                else:
                    avg_state_dict[key] = sum(
                        p["weight"][key] * p["num_data"] for p in selected_params
                    ) // total_size

            self.avg_state_dict = copy.deepcopy(avg_state_dict)
        elif algorithm_name == "fedrhlp":
            selected_params = [copy.deepcopy(self.all_model_parameters[i]) for i in range(len(self.selected_client))]
            print(f"[FedRHLP] Selected {len(selected_params)} params for averaging.")

            # Deep copy state_dict từ client đầu tiên làm mẫu
            avg_state_dict = copy.deepcopy(selected_params[0]["weight"])
            total_size = sum(p["num_data"] for p in selected_params)
            print(f"Total  size = {total_size}")
            for key in avg_state_dict.keys():
                try:
                    if avg_state_dict[key].dtype != torch.long:
                        avg_state_dict[key] = sum(
                            p["weight"][key] * p["num_data"] for p in selected_params
                        ) / total_size
                    else:
                        avg_state_dict[key] = sum(
                            p["weight"][key] * p["num_data"] for p in selected_params
                        ) // total_size
                except Exception as e:
                    print(f"[FedRHLP] Error processing key {key}: {e}")
                    raise

            self.avg_state_dict = copy.deepcopy(avg_state_dict)
            print("[FedRHLP] Averaging complete.")
        elif algorithm_name == "hicsfl": 
            print(f"Avg: {len(self.selected_client)} client")

            if not self.selected_client:
                return
            # Danh sách các parameters tương ứng client đã chọn (đã thu thập đúng thứ tự)
            selected_params = [copy.deepcopy(self.all_model_parameters[i]) for i in range(len(self.selected_client))]
            # Lấy bản sao weights từ client đầu tiên trong danh sách selected_params
            avg_state_dict = copy.deepcopy(selected_params[0]["weight"])
            total_size = sum(p["num_data"] for p in selected_params)
            print(f"Total  size = {total_size}")
            for key in avg_state_dict.keys():
                if avg_state_dict[key].dtype != torch.long:
                    avg_state_dict[key] = sum(
                        p["weight"][key] * p["num_data"] for p in selected_params
                    ) / total_size
                else:
                    avg_state_dict[key] = sum(
                        p["weight"][key] * p["num_data"] for p in selected_params
                    ) // total_size

            self.avg_state_dict = copy.deepcopy(avg_state_dict)
        elif algorithm_name == "haccs":
            print(f"Avg: {len(self.selected_client)} client")

            if not self.selected_client:
                return
            # Danh sách các parameters tương ứng client đã chọn (đã thu thập đúng thứ tự)
            selected_params = [copy.deepcopy(self.all_model_parameters[i]) for i in range(len(self.selected_client))]

            # Lấy bản sao weights từ client đầu tiên trong danh sách selected_params
            avg_state_dict = copy.deepcopy(selected_params[0]["weight"])
            total_size = sum(p["num_data"] for p in selected_params)
            print(f"Total  size = {total_size}")
            for key in avg_state_dict.keys():
                if avg_state_dict[key].dtype != torch.long:
                    avg_state_dict[key] = sum(
                        p["weight"][key] * p["num_data"] for p in selected_params
                    ) / total_size
                else:
                    avg_state_dict[key] = sum(
                        p["weight"][key] * p["num_data"] for p in selected_params
                    ) // total_size

            self.avg_state_dict = copy.deepcopy(avg_state_dict)
        elif algorithm_name == "feel":
            print(f"Avg: {len(self.selected_client)} client")

            if not self.selected_client:
                return
            # Danh sách các parameters tương ứng client đã chọn (đã thu thập đúng thứ tự)
            selected_params = [copy.deepcopy(self.all_model_parameters[i]) for i in range(len(self.selected_client))]

            # Lấy bản sao weights từ client đầu tiên trong danh sách selected_params
            avg_state_dict = copy.deepcopy(selected_params[0]["weight"])
            total_size = sum(p["num_data"] for p in selected_params)
            print(f"Total  size = {total_size}")
            for key in avg_state_dict.keys():
                if avg_state_dict[key].dtype != torch.long:
                    avg_state_dict[key] = sum(
                        p["weight"][key] * p["num_data"] for p in selected_params
                    ) / total_size
                else:
                    avg_state_dict[key] = sum(
                        p["weight"][key] * p["num_data"] for p in selected_params
                    ) // total_size

            self.avg_state_dict = copy.deepcopy(avg_state_dict)
            

def signal_handler(sig, frame):
    print("\nCatch stop signal Ctrl+C. Stop the program.")
    delete_old_queues()
    sys.exit(0)


def delete_old_queues():
    url = f'http://{address}:15672/api/queues'
    response = requests.get(url, auth=HTTPBasicAuth(username, password))

    if response.status_code == 200:
        queues = response.json()

        credentials = pika.PlainCredentials(username, password)
        connection = pika.BlockingConnection(pika.ConnectionParameters(address, 5672, '/', credentials))
        http_channel = connection.channel()

        for queue in queues:
            queue_name = queue['name']
            if queue_name.startswith("reply") or queue_name.startswith("intermediate_queue") or queue_name.startswith(
                    "gradient_queue"):
                try:
                    http_channel.queue_delete(queue=queue_name)
                    src.Log.print_with_color(f"Queue '{queue_name}' deleted.", "green")
                except Exception as e:
                    src.Log.print_with_color(f"Failed to delete queue '{queue_name}': {e}", "yellow")
            else:
                try:
                    http_channel.queue_purge(queue=queue_name)
                    src.Log.print_with_color(f"Queue '{queue_name}' purged.", "green")
                except Exception as e:
                    src.Log.print_with_color(f"Failed to purge queue '{queue_name}': {e}", "yellow")

        connection.close()
        return True
    else:
        src.Log.print_with_color(
            f"Failed to fetch queues from RabbitMQ Management API. Status code: {response.status_code}", "yellow")
        return False


if __name__ == "__main__":
    delete_old_queues()
    signal.signal(signal.SIGINT, signal_handler)
    server = Server()
    server.start()
    src.Log.print_with_color("Ok, ready!", "green")
