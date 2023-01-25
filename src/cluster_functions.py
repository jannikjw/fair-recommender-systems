import pandas as pd
import numpy as np

def calculate_cluster_switch(user_to_item_cluster_assignment, final_user_pref_mapping, config):
    data = []
    data.append(user_to_item_cluster_assignment)
    data.append(final_user_pref_mapping)
    data = np.array(data).T
    cluster_changes_df = pd.DataFrame(data, columns=['initial_cluster', 'final_cluster'])
    cluster_changes_df = cluster_changes_df[cluster_changes_df['initial_cluster'] != cluster_changes_df['final_cluster']]
    print(config['model_name'], ': Number of people who changed clusters: ', cluster_changes_df.shape[0])


def average_distance_other_clusters(user_profiles, item_cluster_centers):
    euclidean_distance_matrix = np.empty((len(user_profiles), len(item_cluster_centers)), dtype=float)
    for i, user in enumerate(user_profiles):
        for j, item_cluster in enumerate(item_cluster_centers):
            euclidean_distance_matrix[i, j] = np.linalg.norm(user - item_cluster)

    user_to_item_cluster_assignment = np.argmin(euclidean_distance_matrix, axis=1)
    
    # calculate average of distances to other clusters
    for user, cluster in enumerate(user_to_item_cluster_assignment):
        euclidean_distance_matrix[user, cluster] = np.nan
    
    mean_diff = np.nanmean(euclidean_distance_matrix)

    return mean_diff


# count number of occurences in cluster_ids
def plot_cluster_distributions(binary_ratings_matrix, user_cluster_ids, item_cluster_ids):
    fig, axs = plt.subplots(2,2, figsize=(15, 7))

    # plot bar chart where x axis is cluster_id and height is number of items in a cluster
    cluster_counts = np.bincount(user_cluster_ids)
    axs[0, 0].bar(x=range(len(cluster_counts)), height=cluster_counts)
    axs[0, 0].set_xticks(range(len(cluster_counts)), range(0, len(cluster_counts)))
    axs[0, 0].set_title('Users per cluster')

    # count number of occurences in cluster_ids
    cluster_counts = np.bincount(item_cluster_ids)
    axs[0, 1].bar(x=range(len(cluster_counts)), height=cluster_counts)
    axs[0, 1].set_xticks(range(len(cluster_counts)), range(0, len(cluster_counts)))
    axs[0, 1].set_title('Items per cluster')

    # Interactions per user cluster
    interactions_per_user_cluster = np.empty(len(np.unique(user_cluster_ids)), dtype=int)
    for i, id in enumerate(user_cluster_ids):
        interactions_per_user_cluster[id] += np.sum(binary_ratings_matrix[i, :])

    # Interactions per item cluster
    interactions_per_item_cluster = np.empty(len(np.unique(item_cluster_ids)), dtype=int)
    for i, id in enumerate(item_cluster_ids):
        interactions_per_item_cluster[id] += np.sum(binary_ratings_matrix[:, i])

    axs[1, 0].bar(x=range(len(interactions_per_user_cluster)), height=interactions_per_user_cluster)
    axs[1, 0].set_xticks(range(len(np.unique(user_cluster_ids))), range(0, len(np.unique(user_cluster_ids))))
    axs[1, 0].set_title('Interactions by User Cluster')

    axs[1, 1].bar(x=range(len(interactions_per_item_cluster)), height=interactions_per_item_cluster)
    axs[1, 1].set_xticks(range(len(np.unique(item_cluster_ids))), range(0, len(np.unique(item_cluster_ids))))
    axs[1, 1].set_title('Interactions by Item Cluster')

    plt.show()
