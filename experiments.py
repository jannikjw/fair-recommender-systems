import numpy as np
import argparse
import os
import pickle 
from trecs.metrics import MSEMeasurement, InteractionSpread, InteractionSpread, InteractionSimilarity, RecSimilarity, RMSEMeasurement, InteractionMeasurement
from trecs.components import Users

from wrapper.models.bubble import BubbleBurster
from src.utils import *
from src.scoring_functions import cosine_sim, entropy, content_fairness, top_k_reranking
import src.globals as globals
from wrapper.metrics.evaluation_metrics import SerendipityMetric, DiversityMetric, NoveltyMetric, TopicInteractionMeasurement, MeanNumberOfTopics, RecallMeasurement, UserMSEMeasurement

# ignore all future warnings
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

random_state = np.random.seed(42)


def run_experiment(config, measurements, train_timesteps=20, run_timesteps=50):
    model = BubbleBurster(**config)

    # Add Metrics
    model.add_metrics(*measurements)

    print('Training:')
    model.startup_and_train(timesteps=train_timesteps)
    print('Simulating:')
    model.run(run_timesteps, repeated_items=False)

    return model


def create_folder_structure():
    if not os.path.exists('artefacts/'):
        os.mkdir('artefacts/')
    if not os.path.exists('artefacts/topic_clusters/'):
        os.mkdir('artefacts/topic_clusters')
    if not os.path.exists('artefacts/measurements'):
        os.mkdir('artefacts/measurements')
    if not os.path.exists('artefacts/representations/'):
        os.mkdir('artefacts/representations')
    if not os.path.exists('artefacts/models/'):
        os.mkdir('artefacts/models')
    if not os.path.exists('artefacts/final_preferences/'):
        os.mkdir('artefacts/final_preferences')


def main():
    create_folder_structure()
    
    # Initialize parser
    parser = argparse.ArgumentParser()

    # Adding optional argument
    parser.add_argument("-a", "--Attributes", help = "Number of attributes", type=int, default=20)
    parser.add_argument("-c", "--Clusters", help = "Number of clusters", type=int, default=25)
    parser.add_argument("-tt", "--TrainTimesteps", help = "Number of timesteps for training", type=int, default=10)
    parser.add_argument("-rt", "--RunTimesteps", help = "Number of timesteps for simulation", type=int, default=100)
    parser.add_argument("-p", "--Probabilistic", help = "Is model probabilistic?", type=bool, default=False)
    parser.add_argument("-s", "--ScoreFN", help = "Name of the score function of the model", type=str,  default='')
    parser.add_argument("-l", "--Lambda", help = "Weight of regularizer in score function", type=float, default=0.1)
    parser.add_argument("-ud", "--UserDrift", help = "Factor of drift in user preferences. Values in [0,1].", type=float, default=0.05)
    parser.add_argument("-ua", "--UserAttention", help = "Factor of attention to ranking of iems. Values [-1, 0].", type=float, default=-0.8)
    parser.add_argument("-n", "--Name", help = "Name of experiment. Requires folder in artefacts.", type=str, default='general')

    
    # Read arguments from command line
    args = parser.parse_args()

    experiment_name = args.Name
    n_attrs = args.Attributes
    n_clusters = args.Clusters
    train_timesteps = args.TrainTimesteps
    run_timesteps = args.RunTimesteps
    drift = args.UserDrift
    attention_exp = args.UserAttention
    num_items_per_iter = 10
    max_iter = 1000

    globals.initialize()
    globals.ALPHA = float(args.Lambda) if args.Lambda else float(0.2)  
    alpha = globals.ALPHA
        
    config = {
        'num_attributes': n_attrs,
        'num_items_per_iter': num_items_per_iter,
        'seed': 42,
        'record_base_state': True,
    }

    model_name='myopic'
    requires_alpha = False
        
    if args.ScoreFN:
        score_fn = args.ScoreFN
        if score_fn == 'cosine_sim':
            config['score_fn'] = cosine_sim
            requires_alpha = True
        elif score_fn == 'entropy':
            config['score_fn'] = entropy
            requires_alpha = True
        elif score_fn == 'content_fairness':
            config['score_fn'] = content_fairness        
        elif score_fn == 'top_k_reranking':
            config['score_fn'] = top_k_reranking   
        else:
            raise Exception('Given score function does not exist.')
        model_name = args.ScoreFN
    if args.Probabilistic:
        config['probabilistic_recommendations'] = True
        model_name += '_prob'
        
    # Print model configuration
    print('Experiment name: ', experiment_name)
    print("-------------------------Model Parameters-------------------------")
    print("Model name: ", model_name)
    print("Number of Iterations for NMF: ", max_iter)
    print("Number of Attributes: ", n_attrs)
    print("Number of Clusters: ", n_clusters)
    if requires_alpha: 
        print("Lambda: ", globals.ALPHA)
    print("Probabilistic: ", args.Probabilistic)
    print("Number of items recommended at each timesteps: ", num_items_per_iter)
    print("Training Timesteps: ", train_timesteps)
    print("Running Timesteps: ", run_timesteps)

    # Get embeddings
    print("-------------------Get Embeddings and Clusters-------------------")
    interaction_matrix = load_and_process_movielens(file_path='data/ml-100k/u.data')
    user_representation, item_representation = create_embeddings(interaction_matrix, n_attrs=n_attrs, max_iter=max_iter)
            
    # user_representation = user_representation[:100, :]
    # Get item and user clusters
    item_cluster_ids, item_cluster_centers = get_clusters(item_representation.T, name='item', n_clusters=n_clusters, n_attrs=n_attrs, max_iter=max_iter)
    user_cluster_ids, user_cluster_centers = get_clusters(user_representation, name='user', n_clusters=n_clusters, n_attrs=n_attrs, max_iter=max_iter)
    
    # Define users
    users = Users(actual_user_profiles=user_representation, 
                  repeat_interactions=False, 
                  drift=drift,
                  attention_exp=attention_exp)
    
    config['actual_user_representation'] = users
    config['actual_item_representation'] = item_representation
    config['item_topics'] = item_cluster_ids

    user_item_cluster_mapping = user_topic_mapping(user_representation, item_cluster_centers) # TODO: Remove?
    # Create user_pairs by pairing users only with others that are not in the same cluster
    inter_cluster_user_pairs, intra_cluster_user_pairs = create_cluster_user_pairs(user_item_cluster_mapping)

    print("-------------------------User Parameters-------------------------")
    print("Drift: ", drift)
    print("Attention Exponent: ", attention_exp)
    
    print("----------------------------Run Model----------------------------")
    
    # Define model
    measurements = [
        InteractionMeasurement(),
        MSEMeasurement(),
        UserMSEMeasurement(),
        RecallMeasurement(),
        InteractionSpread(),                
        InteractionSimilarity(pairs=inter_cluster_user_pairs, name='inter_cluster_interaction_similarity'), 
        InteractionSimilarity(pairs=intra_cluster_user_pairs, name='intra_cluster_interaction_similarity'), 
        DiversityMetric(),         
        RecSimilarity(pairs=inter_cluster_user_pairs, name='inter_cluster_rec_similarity'), 
        RecSimilarity(pairs=intra_cluster_user_pairs, name='intra_cluster_rec_similarity'), 
        SerendipityMetric(), 
        NoveltyMetric(),
    ]

    model = run_experiment(config, measurements, train_timesteps=train_timesteps, run_timesteps=run_timesteps)
    
    # Determine file name based on parameter values
    parameters = f'_{train_timesteps}trainTimesteps_{run_timesteps}runTimesteps_{n_attrs}nAttrs_{n_clusters}nClusters_{drift}Drift_{attention_exp}AttentionExp'
    if requires_alpha:
        parameters += f'_{alpha}Lambda'
    
    # Save actual user preferences
    final_preferences_dir = f'artefacts/{experiment_name}/final_preferences/'
    file_prefix = f'{model_name}_final_preferences'
    final_preferences_path = final_preferences_dir + file_prefix + parameters + '.npy'
    np.save(final_preferences_path, model.users.actual_user_profiles.value, allow_pickle=True)
    print('User preferences saved.')

    
    # Save measurements
    measurements_dir = f'artefacts/{experiment_name}/measurements/'
    file_prefix = f'{model_name}_measurements'
        
    measurements_path = measurements_dir + file_prefix + parameters + '.csv'
    measurements_df = load_or_create_measurements_df(model, model_name, train_timesteps, measurements_path)
    measurements_df.to_csv(measurements_path)
    print('Measurements saved.')


if __name__=="__main__":
    main()