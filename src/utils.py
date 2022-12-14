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


def load_measurements(path, columns):
    dfs = []
    data = []
    
    for file in os.listdir(path):
        if file.endswith('.csv'):
            row = collect_parameters(file, columns)
            data.append(row)
            df = pd.read_csv(path + '/' + file)
            dfs.append(df)
    
    parameters_df = pd.DataFrame().append(data)
    for col in numeric_cols:
        parameters_df[col] = pd.to_numeric(parameters_df[col])
    return dfs, parameters_df


def plot_measurements(dfs, parameters_df):
    fig, ax = plt.subplots(3, 3, figsize=(15, 15))
    fig.tight_layout(pad=5.0)

    # plot rec_similarity with timesteps on x axis
    legend_lines, legend_names = [], []
    for i, df in enumerate(dfs):
        ts = df['timesteps']
        name = parameters_df.loc[i, 'model_name']
        if not np.isnan(parameters_df.loc[i, 'Lambda']):
             name += f" (Lambda: {parameters_df.loc[i, 'Lambda']})" 
        legend_names.append(name)
        ax[0,0].plot(ts, df['mse'], label=name)
        ax[0,1].plot(ts, df['rec_similarity'], label=name)
        ax[0,2].plot(ts, df['interaction_similarity'], label=name)
        ax[1,0].plot(ts, df['serendipity_metric'], label=name)
        ax[1,1].plot(ts, df['novelty_metric'], label=name)
        line, = ax[1,2].plot(ts, df['diversity_metric'], label=name)
        legend_lines.append(line)

    for a in ax:
        for b in a:
            b.set_xlabel('Timestep')

    ax[0, 0].set_title('Mean Squared Error')
    ax[0, 0].set_ylabel('MSE')
    
    ax[0, 1].set_title('Recommendation similarity')
    ax[0, 1].set_ylabel('Similarity')
    
    ax[0, 2].set_title('Interaction Similarity')
    ax[0, 2].set_ylabel('Jaccard Similarity')
    
    ax[1, 0].set_title('Serendipity')
    ax[1, 0].set_ylabel('Serendipity')
    
    ax[1, 1].set_title('Novelty')
    ax[1, 1].set_ylabel('Novelty')

    ax[1, 2].set_title('Diversity')
    ax[1, 2].set_ylabel('Diversity')

    ax[2, 0].set_title('Recall')
    ax[2, 0].set_ylabel('Recall')
    
    fig.legend(legend_lines, legend_names, loc='upper center', fontsize=14, frameon=False, ncol=5, bbox_to_anchor=(.5, 1.05))
