from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
import numpy as np
import pandas as pd
import os
random_state = np.random.seed(42)

def get_topic_clusters(interaction_matrix, n_clusters:int=100, n_attrs:int=100, max_iter:int=100):
    """
    Creates clusters of movies based on their genre.
    Inputs:
        binary_ratings_matrix: a binary matrix of users and movies
        n_attrs: number of attributes to use in NMF
        nmf_solver: solver to use in NMF
    Outputs:
        clusters: a list of cluster assignments
    """
    # Create topic clusters
    #create co-occurence matrix from binary_interaction_matrix
    file_path = f'artefacts/topic_clusters/topic_clusters_{n_clusters}clusters_{n_attrs}attributes_{max_iter}iters.npy'
    if not os.path.exists(file_path):
        co_occurence_matrix = interaction_matrix.T @ interaction_matrix

        co_occurence_matrix = interaction_matrix.T @ interaction_matrix

        # Matrix factorize co_occurence_matrix to get embeddings
        nmf_cooc = NMF(n_components=n_attrs, max_iter=max_iter)
        W_topics = nmf_cooc.fit_transform(co_occurence_matrix)

        # cluster W_topics
        cluster_ids = KMeans(n_clusters=n_clusters, random_state=random_state).fit_predict(W_topics)
    else:
        cluster_ids = np.load(file_path)

    return cluster_ids


def create_embeddings(binary_matrix, n_attrs:int=100, max_iter:int=100):
    """
    Creates embeddings for users and items based on their interactions.
    Inputs:
        binary_matrix: a binary matrix of users and movies
        n_attrs: number of attributes to use in NMF
        max_iter: number of iteration for NMF
    Outputs:
        user_representation: a matrix of user embeddings
        item_representation: a matrix of item embeddings
    """
    user_representation_file_path = f'artefacts/representations/ml_user_representations_{n_attrs}attributes_{max_iter}iters.npy'
    item_representation_file_path = f'artefacts/representations/ml_item_representations_{n_attrs}attributes_{max_iter}iters.npy'
    if not os.path.exists(user_representation_file_path) or not os.path.exists(item_representation_file_path):
        nmf = NMF(n_components=n_attrs, init='random', random_state=random_state, max_iter=max_iter)
        user_representation = nmf.fit_transform(binary_matrix)
        item_representation = nmf.components_
        np.save(user_representation_file_path, user_representation)
        np.save(item_representation_file_path, item_representation)
    else:
        user_representation = np.load(user_representation_file_path)
        item_representation = np.load(item_representation_file_path)

    return user_representation, item_representation


def load_and_process_movielens(file_path):
    ratings_df = pd.read_csv(file_path, sep="\t", names=['UserID', 'MovieID', 'Rating', 'Timestamp'])
    binary_ratings_df = ratings_df.drop(columns=['Timestamp'])
    binary_ratings_df.loc[binary_ratings_df['Rating'] > 0, 'Rating'] = 1

    # turn dataframe into matrix where each movie is a column and each user is a row
    binary_ratings_matrix = binary_ratings_df.pivot(index='UserID', columns='MovieID', values='Rating').fillna(0).to_numpy()
    return binary_ratings_matrix


def load_or_create_measurements_df(model, model_name, file_path):
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, index_col=0)
    else:
        measurements = model.get_measurements()
        df = pd.DataFrame(measurements)
        df['model'] = model_name
    
    return df