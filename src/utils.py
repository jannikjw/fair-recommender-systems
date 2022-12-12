from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import src.globals as globals
random_state = np.random.seed(42)

def get_topic_clusters(cooccurence_matrix, n_clusters:int=100, n_attrs:int=100, max_iter:int=100):
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
        print('Calculating clusters...')
        # Matrix factorize co_occurence_matrix to get embeddings
        nmf_cooc = NMF(n_components=n_attrs, max_iter=max_iter)
        W_topics = nmf_cooc.fit_transform(co_occurence_matrix)

        # cluster W_topics
        cluster_ids = KMeans(n_clusters=n_clusters, max_iter=max_iter, random_state=random_state).fit_predict(W_topics)
        np.save(file_path, cluster_ids)

        print('Calculated clusters.')
    else:
        cluster_ids = np.load(file_path)
        print('Loaded clusters.')

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
        df['lambda'] = globals.ALPHA
    
    return df


def collect_parameters(file):
    numeric_cols = ['trainTimesteps', 'runTimesteps', 'nAttrs', 'nClusters', 'Lambda']
    columns = ['model_name'] + numeric_cols
    
    file_name = file[:-4]
    params = file_name.split('_')
    params_start_id = params.index('measurements')
    model_name = '_'.join(params[:params_start_id])
    row = []
    row.append(model_name)
    for col in columns:
        for param in params:
            if param.endswith(col):
                value = param[:param.find(col)]
                row.append(value)
    return row


def load_measurements(path):
    dfs = []
    data = []
    numeric_cols = ['trainTimesteps', 'runTimesteps', 'nAttrs', 'nClusters', 'Drift', 'AttentionExp', 'PairAll', 'Lambda']
    columns = ['model_name'] + numeric_cols
    
    for file in os.listdir(path):
        if file.endswith('.csv'):
            row = collect_parameters(file)
            data.append(row)
            df = pd.read_csv(path + '/' + file)
            dfs.append(df)
    
    parameters_df = pd.DataFrame(data, 
                                 columns=columns)                                
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