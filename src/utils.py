import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../t-recs/')
import trecs.matrix_ops as mo

from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
import numpy as np
import pandas as pd
import os
import pickle
import trecs.matrix_ops as mo
import matplotlib.pyplot as plt
import src.globals as globals
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

random_state = np.random.seed(42)

def get_clusters(embeddings, name, n_clusters:int=25, n_attrs:int=20, max_iter:int=100):
    """
    Creates clusters of movies based on their genre.
    Inputs:
        embeddings: Matrix of embeddings, e.g. user representation
        n_attrs: number of attributes to use in NMF
        nmf_solver: solver to use in NMF
    Outputs:
        clusters: a list of cluster assignments
    """
    # Create topic clusters
    #create co-occurence matrix from binary_interaction_matrix
    file_path = f'artefacts/topic_clusters/{name}_clusters_{n_clusters}clusters_{n_attrs}attributes_{max_iter}iters.pkl'
    if not os.path.exists(file_path):
        print('Calculating clusters...')

        kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter, random_state=random_state).fit(embeddings)
        pickle.dump(kmeans, open(file_path, 'wb'))

        print('Calculated clusters.')
    else:         
        # load the model from disk
        kmeans = pickle.load(open(file_path, 'rb'))
        print('Loaded clusters.')

    cluster_ids = kmeans.predict(embeddings)
    centroids = kmeans.cluster_centers_
    return cluster_ids, centroids



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
        print('Calculating embeddings...')
        nmf = NMF(n_components=n_attrs, init='random', random_state=random_state, max_iter=max_iter)
        user_representation = nmf.fit_transform(binary_matrix)
        item_representation = nmf.components_
        np.save(user_representation_file_path, user_representation)
        np.save(item_representation_file_path, item_representation)
        print('Calculated embeddings.')
    else:
        user_representation = np.load(user_representation_file_path)
        item_representation = np.load(item_representation_file_path)
        print('Loaded embeddings.')

    return user_representation, item_representation


def load_and_process_movielens(file_path):
    ratings_df = pd.read_csv(file_path, sep="\t", names=['UserID', 'MovieID', 'Rating', 'Timestamp'])
    binary_ratings_df = ratings_df.drop(columns=['Timestamp'])
    binary_ratings_df.loc[binary_ratings_df['Rating'] > 0, 'Rating'] = 1

    # turn dataframe into matrix where each movie is a column and each user is a row
    binary_ratings_matrix = binary_ratings_df.pivot(index='UserID', columns='MovieID', values='Rating').fillna(0).to_numpy()
    return binary_ratings_matrix


def load_or_create_measurements_df(model, model_name, train_timesteps, file_path):
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, index_col=0)
    else:
        measurements = model.get_measurements()
        df = pd.DataFrame(measurements)
        df['state'] = 'train' # makes it easier to later understand which part was training
        df.loc[df['timesteps'] > train_timesteps, 'state'] = 'run'
        df['model'] = model_name
    
    return df


def user_topic_mapping(user_profiles, item_cluster_centers):
    """
    This function maps users to topics. This mapping can be for either:
        -> actual_user_topic_mapping:
            such that:  user_profiles = actual_user_profiles, and
                        item_attributes = actual_item_attributes
        -> predicted_user_topic_mapping:
            such that:  user_profiles = predicted_user_profiles, and
                        item_attributes = predicted_item_attributes
    
    Parameters
    -----------
        user_profiles: :obj:`numpy.ndarray`, with dims=(#users, #attributes),
                where #attributes is equal to the number of attributes that the algorithm 
                uses to represent each item and user.
            Matrix representation of either predicted or actual user profiles.
            
        item_attributes: :obj:`numpy.ndarray`, with dims=(#attributes, #items),
                where # attributes is equal to the number of attributes that the algorithm 
                uses to represent each item and user.
            Matrix representation of either predicted or actual item attributes
        
        item_topics: array_like, size=(, num_items)
            Represents the topic cluster to which each item belongs
    Returns
    ---------
        :obj:`numpy.ndarray`, with dims=(#users, |set(item_topics)|)
            Histogram of the number of interactions aggregated by items at the given timestep.
    """   
    euclidean_distance_matrix = np.empty((len(user_profiles), len(item_cluster_centers)), dtype=float)
    for i, user in enumerate(user_profiles):
        for j, item_cluster in enumerate(item_cluster_centers):
            euclidean_distance_matrix[i, j] = np.linalg.norm(user - item_cluster)

    user_to_item_cluster_assignment = np.argmin(euclidean_distance_matrix, axis=1)
    return user_to_item_cluster_assignment


def collect_parameters(file, columns):   
    file_name = file[:-4]
    params = file_name.split('_')
    params_start_id = params.index('measurements')
    row = {}
    row['model_name'] = '_'.join(params[:params_start_id])
    for col in columns:
        for param in params:
            if param.endswith(col):
                value = param[:param.find(col)]
                row[col] = value
    return row


def load_measurements(path, numeric_columns):
    dfs = []
    data = []
    columns = ['model_name'] + numeric_columns
    
    for file in os.listdir(path):
        if file.endswith('.csv'):
            row = collect_parameters(file, columns)
            data.append(row)
            df = pd.read_csv(path + '/' + file)
            dfs.append(df)
    
    parameters_df = pd.DataFrame().append(data)
    for col in numeric_columns:
        parameters_df[col] = pd.to_numeric(parameters_df[col])
    return dfs, parameters_df


def create_parameter_string(naming_config):
    parameters_str = ''
    for key, value in naming_config.items():
        parameters_str += f'_{value}{key}'

        
def create_cluster_user_pairs(user_cluster_ids):
    inter_cluster_user_pairs = []
    num_users = len(user_cluster_ids)
    
    for u_idx in range(num_users):
        for v_idx in range(num_users):
            if user_cluster_ids[u_idx] != user_cluster_ids[v_idx]:
                inter_cluster_user_pairs.append((u_idx, v_idx))
    
    intra_cluster_user_pairs = []
    for u_idx in range(num_users):
        for v_idx in range(num_users):
            if user_cluster_ids[u_idx] == user_cluster_ids[v_idx]:
                intra_cluster_user_pairs.append((u_idx, v_idx))

    return inter_cluster_user_pairs, intra_cluster_user_pairs

    
def analyze_user_mse(df_user_mse, train_timesteps=0):
    """
    df_user_mse: pandas.DataFrame
        Each column should represent a timestep and each row is the MSE that corresponds to a given user.
        Additional column `clusterID` must be present which stores the cluster number that each user is assigned to.
    train_timesteps: int
        The number of train_timesteps that are present in the dataframe. 
        If train_timesteps != 0, then the columns 0:train_timesteps will be excluded from the calculations.
    """
    timesteps = np.arange(df_user_mse.shape[1] - train_timesteps)
    print(timesteps[0::10])
    cluster_mse = df_user_mse.iloc[:, train_timesteps:].groupby(['clusterID']).mean()
    worst_user_cluster = cluster_mse.mean(axis=1).idxmax()
    
    plt.plot(cluster_mse.mean(), label = "average MSE")
    plt.plot(cluster_mse.loc[worst_user_cluster], label = f"cluster {worst_user_cluster} MSE")
    plt.title('Avg. MSE vs. Worst-User Cluster MSE')
    plt.xlabel("Timestep")
    plt.ylabel("MSE")
    plt.xticks(timesteps[0::10])
    plt.legend()
    