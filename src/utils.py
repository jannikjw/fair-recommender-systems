from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
import numpy as np
random_state = np.random.seed(42)

def get_topic_clusters(interaction_matrix, n_clusters:int=100, n_attrs:int=100, max_iter:int=100, nmf_solver:str="mu"):
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
    co_occurence_matrix = interaction_matrix.T @ interaction_matrix
    co_occurence_matrix

    # Matrix factorize co_occurence_matrix to get embeddings
    nmf_cooc = NMF(n_components=n_attrs, solver=nmf_solver, max_iter=max_iter)
    W_topics = nmf_cooc.fit_transform(co_occurence_matrix)

    # cluster W_topics
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(W_topics)

    # assign nearest cluster to observation
    cluster_ids = kmeans.predict(W_topics)

    return cluster_ids


def create_embeddings(binary_matrix, max_iter, n_attrs):
    user_representation_file_path = f'artefacts/ml_user_representations_{max_iter}maxiter_{n_attrs}nAttrs.npy'
    item_representation_file_path = f'artefacts/ml_item_representations_{max_iter}maxiter_{n_attrs}nAttrs.npy'
    if not os.path.exists(user_representation_file_path) or not os.path.exists(item_representation_file_path):
        nmf = NMF(n_components=n_attrs, init='random', random_state=random_state, max_iter=max_iter)
        user_representation = nmf.fit_transform(binary_ratings_matrix)
        item_representation = nmf.components_
        np.save(user_representation_file_path, user_representation)
        np.save(item_representation_file_path, item_representation)
    else:
        user_representation = np.load(user_representation_file_path)
        item_representation = np.load(item_representation_file_path)
        
    return user_representation, item_representation