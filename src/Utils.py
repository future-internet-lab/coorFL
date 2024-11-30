import numpy as np


def generate_random_array(target_sum, size, max_value=None):
    random_array = np.random.randint(1, target_sum * size, size)

    current_sum = random_array.sum()
    current_sum = (target_sum * size * random_array)/np.sum(current_sum)

    for i in range(size):
        if max_value and current_sum[i] > max_value:
            current_sum[i] = max_value

    return current_sum.astype(int).tolist()
