import numpy as np
import random


def client_selection_speed_base(indices, all_speeds, all_num_datas):
    speeds = np.array(all_speeds)[indices]
    num_datas = np.array(all_num_datas)[indices]

    def num_active_client(speeds, num_datas, t):
        # number of active client
        n_client = 0
        for i in range(len(speeds)):
            trained_data = speeds[i] * t
            if trained_data <= num_datas[i]:
                n_client += 1
        return n_client

    def sum_velocity(speeds, num_datas, t):
        # sum of velocity
        velocity = 0.0
        for i in range(len(speeds)):
            trained_data = speeds[i] * t
            if trained_data <= num_datas[i]:
                velocity += speeds[i]
        return velocity

    listY = []
    vPerC = 0
    t = 0
    while True:
        active_client = num_active_client(speeds, num_datas, t)
        if active_client != 0:
            a = sum_velocity(speeds, num_datas, t) / active_client
            # print(f"t={t} - n_client={active_client} - {vPerC / (t + 1)}")
            listY.append(vPerC / (t + 1))
            vPerC += a
        else:
            break
        t += 1

    training_times = np.array(num_datas) / np.array(speeds)

    return [indices[i] for i, value in enumerate(training_times) if value < np.argmax(listY) + 1]


def client_selection_random(client_list, num_client=1):
    num_client = min(num_client, len(client_list))
    return random.sample(client_list, num_client)
def client_selection_random_rate(client_list, rate = 0.3, num_client=1):
    num_client = max(1, int(len(client_list) * rate))  
    return random.sample(client_list, num_client)
