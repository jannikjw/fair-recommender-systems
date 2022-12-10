import numpy as np
import argparse
import os
import pickle 
from trecs.metrics import MSEMeasurement, InteractionSpread, InteractionSpread, InteractionSimilarity, RecSimilarity, RMSEMeasurement, InteractionMeasurement
from trecs.components import Users

from wrapper.models.bubble import BubbleBurster
from src.utils import get_topic_clusters, create_embeddings, load_and_process_movielens, load_or_create_measurements_df
from src.scoring_functions import cosine_sim, entropy
import src.globals as globals
from wrapper.metrics.evaluation_metrics import SerendipityMetric, DiversityMetric, NoveltyMetric, TopicInteractionMeasurement, MeanNumberOfTopics

# ignore all future warnings
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

random_state = np.random.seed(42)


def run_experiment(config, measurements, train_timesteps=20, run_timesteps=50):
    model = BubbleBurster(**config)

    # Add Metrics
    model.add_metrics(*measurements)

    model.startup_and_train(timesteps=train_timesteps)
    model.run(run_timesteps)

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


def main():
    create_folder_structure()

    # Initialize parser
    parser = argparse.ArgumentParser()

    # Adding optional argument
    parser.add_argument("-a", "--Attributes", help = "Number of attributes")
    parser.add_argument("-c", "--Clusters", help = "Number of clusters")
    parser.add_argument("-tt", "--TrainTimesteps", help = "Number of timesteps for training")
    parser.add_argument("-rt", "--RunTimesteps", help = "Number of timesteps for simulation")
    parser.add_argument("-p", "--Probabilistic", help = "Is model probabilistic?")
    parser.add_argument("-s", "--ScoreFN", help = "Name of the score function of the model")
    parser.add_argument("-l", "--Lambda", help = "Weight of regularizer in score function")

    # Read arguments from command line
    args = parser.parse_args()
    print(args)

    n_attrs = int(args.Attributes) if args.Attributes else 20
    n_clusters = int(args.Clusters) if args.Clusters else 50
    train_timesteps = int(args.TrainTimesteps) if args.TrainTimesteps else 10
    run_timesteps = int(args.RunTimesteps) if args.RunTimesteps else 100
    num_items_per_iter = 10
    max_iter = 1000

    globals.initialize()
    globals.ALPHA = float(args.Lambda) if args.Lambda else 0.2
    alpha = globals.ALPHA

    # print variables above
    print("Number of Iterations for NMF: ", max_iter)
    print("Number of Attributes: ", n_attrs)
    print("Number of Clusters: ", n_clusters)
    print("Lambda: ", globals.ALPHA)
    print("Number of items recommended at each timesteps: ", num_items_per_iter)
    print("Training Timesteps: ", train_timesteps)
    print("Running Timesteps: ", run_timesteps)

    binary_ratings_matrix = load_and_process_movielens(file_path='data/ml-100k/u.data')
    user_representation, item_representation = create_embeddings(binary_ratings_matrix, n_attrs=n_attrs, max_iter=max_iter)
    item_topics = get_topic_clusters(binary_ratings_matrix, n_clusters=n_clusters, n_attrs=n_attrs, max_iter=max_iter)  

    users = Users(actual_user_profiles=user_representation, repeat_interactions=False, attention_exp=1.5, verbose=True)
    
    config = {
        'actual_user_representation': users,
        'actual_item_representation': item_representation,
        'item_topics': item_topics,
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
        else:
            raise Exception('Given score function does not exist.')
        model_name = args.ScoreFN
    if args.Probabilistic == 'True':
        config['probabilistic_recommendations'] = True
        model_name += '_prob'
        
    print(f'Model name: {model_name}')

    user_pairs = [(u_idx, v_idx) for u_idx in range(len(user_representation)) for v_idx in range(len(user_representation))]
    measurements = [
        InteractionMeasurement(), 
        MSEMeasurement(),  
        InteractionSpread(), 
        InteractionSimilarity(pairs=user_pairs), 
        RecSimilarity(pairs=user_pairs), 
        # TopicInteractionMeasurement(),
        # MeanNumberOfTopics(),
        SerendipityMetric(), 
        DiversityMetric(), 
        NoveltyMetric()
    ]

    model = run_experiment(config, measurements, train_timesteps=train_timesteps, run_timesteps=run_timesteps)
    
    # Save measurements
    measurements_path = f'artefacts/measurements/{model_name}_measurements_{train_timesteps}trainTimesteps_{run_timesteps}runTimesteps_{n_attrs}nAttrs_{n_clusters}nClusters'
    if requires_alpha:
        measurements_path += f'_{alpha}Lambda'
    measurements_path += '.csv'
    measurements_df = load_or_create_measurements_df(model, model_name, train_timesteps, measurements_path)
    measurements_df.to_csv(measurements_path)
    print('Measurements saved.')


if __name__=="__main__":
    main()