from sklearn.cluster import AffinityPropagation


def clustering_algorithm(label_counts):
    # Apply Affinity Propagation clustering
    affinity_propagation = AffinityPropagation(random_state=0)
    affinity_propagation.fit(label_counts)

    # Extract cluster centers and labels for each data point
    cluster_centers_indices = affinity_propagation.cluster_centers_indices_
    labels = affinity_propagation.labels_

    return len(cluster_centers_indices), labels
