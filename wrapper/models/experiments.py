import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.decomposition import NMF
from importlib import reload
import argparse
import wrapper
import trecs
import os
from scipy import sparse
from trecs.models import ContentFiltering
from trecs.metrics import MSEMeasurement, InteractionSpread, InteractionSpread, InteractionSimilarity, RecSimilarity, RMSEMeasurement, InteractionMeasurement
from trecs.components import Users

from wrapper.models.bubble import BubbleBurster
from src.utils import get_topic_clusters, create_embeddings, load_and_process_movielens, load_or_create_measurements_df
from wrapper.metrics.evaluation_metrics import SerendipityMetric, DiversityMetric, NoveltyMetric, TopicInteractionMeasurement, MeanNumberOfTopics

random_state = np.random.seed(42)


def run_experiment(config, measurements, train_timesteps=20, run_timesteps=50):
    users = Users(actual_user_profiles=user_representation, repeat_interactions=False, attention_exp=1.5)

    model = BubbleBurster(**config)

    # Add Metrics
    model.add_metrics(*measurements)

    model.startup_and_train(timesteps=train_timesteps)
    model.run(run_timesteps)

    return model

def main():
    # Initialize parser
    parser = argparse.ArgumentParser()

    # Adding optional argument
    parser.add_argument("-m", "--Model", help = "Define RS model")
    parser.add_argument("-a", "--Attributes", help = "Number of attributes")
    parser.add_argument("-c", "--Clusters", help = "Number of clusters")
    parser.add_argument("-tt", "--TrainTimesteps", help = "Number of timesteps for training")
    parser.add_argument("-rt", "--RunTimesteps", help = "Number of timesteps for simulation")
    parser.add_argument("-p", "--Probablistic", help = "Is model probabilistic?")
    parser.add_argument("-s", "--ScoreFN", help = "Name of the score function of the model")

    # Read arguments from command line
    args = parser.parse_args()
    print(args)

    n_attrs = int(args.Attributes) if args.Attributes else 100
    n_clusters = int(args.Clusters) if args.Clusters else 20
    train_timesteps = int(args.TrainTimesteps) if args.TrainTimesteps else 20
    run_timesteps = int(args.RunTimesteps) if args.RunTimesteps else 50
    max_iter = 500

    binary_ratings_matrix = load_and_process_movielens(file_path='data/ml-100k/u.data')
    user_representation, item_representation = create_embeddings(binary_ratings_matrix, n_attrs=n_attrs, max_iter=max_iter)
    item_topics = get_topic_clusters(binary_ratings_matrix, n_clusters=n_clusters, n_attrs=n_attrs, max_iter=max_iter)  

    config = {
        'actual_user_representation': user_representation,
        'actual_item_representation': item_representation,
        'item_topics': item_topics,
        'num_attributes': n_attrs,
    }

    model_name='myopic'

    if args.ScoreFN:
        config['score_fn'] = args.ScoreFN
        model_name = args.ScoreFN
    if args.Probabilistic == 'True':
        config['probabilistic_recommendations'] = True
        model_name += '_prob'

    user_pairs = [(u_idx, v_idx) for u_idx in range(len(user_representation)) for v_idx in range(len(user_representation))]
    measurements = [
        InteractionMeasurement(), 
        MSEMeasurement(),  
        InteractionSpread(), 
        InteractionSimilarity(pairs=user_pairs), 
        RecSimilarity(pairs=user_pairs), 
        # TopicInteractionMeasurement(),
        MeanNumberOfTopics(),
        SerendipityMetric(), 
        DiversityMetric(), 
        NoveltyMetric()
    ]

    model = run_experiment(config, measurements, train_timesteps=train_timesteps, run_timesteps=run_timesteps)
    path = f'artefacts/measurements/{model_name}_measurements_{train_timesteps}trainTimesteps_{run_timesteps}runTimesteps_{n_attrs}nAttrs_{n_clusters}nClusters.csv'
    measurements_df = load_or_create_measurements_df(model, model_name, path)
    measurements_df.to_csv(path)


if __name__=="__main__":
    main()