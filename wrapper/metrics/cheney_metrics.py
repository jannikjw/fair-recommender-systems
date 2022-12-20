import numpy as np
from collections import defaultdict
import os
import pickle as pkl

import sys
sys.path.insert(1, '../t-recs/')
from trecs.metrics import Measurement
from trecs.models import ContentFiltering
from trecs.matrix_ops import normalize_matrix, inner_product
from trecs.random import Generator

from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import nnls
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

def calculate_avg_jaccard(pairs, interactions):
    """ Calculates average Jaccard index over specified pairs of users.
    """
    similarity = 0
    num_pairs = len(pairs)
    for user1, user2 in pairs:
        itemset_1 = set(interactions[user1, :])
        itemset_2 = set(interactions[user2, :])
        common = len(itemset_1.intersection(itemset_2))
        union = len(itemset_1.union(itemset_2))
        similarity += common / union / num_pairs
    return similarity

# Calculate homogenization by the average Euclidean distance of the interaction set

def avg_interaction_distance(items1, items2, item_attributes):
    """
    Assumes items are provided in timestep order;
    averages the euclidean distance over timesteps.

    Assume items matrix is |A| x |I|
    """
    num_steps = len(items1)
    assert len(items1) == len(items2) # should have interacted with same # of itesm
    total_distance = 0
    for i in range(num_steps):
        item1 = item_attributes[:, items1[i]]
        item2 = item_attributes[:, items2[i]]
        total_distance += np.linalg.norm(item1 - item2)
    return total_distance / num_steps

def distance_of_mean_items(items1, items2, item_attributes):
    """
    Returns the difference between the average vector of the items
    in set 1 and the average vector of the items in set 2.

    Assume items matrix is |A| x |I|
    """
    mean1 = item_attributes[:, items1].mean(axis=1)
    mean2 = item_attributes[:, items2].mean(axis=1)
    return np.linalg.norm(mean1 - mean2)

def mean_item_dist_pairs(pairs, interaction_history, item_attributes):
    """
    For each pair, calculates the distance between the mean item
    interacted with by each member of the pair. Then averages these
    distances across all pairs.
    """
    dist = 0
    for pair in pairs:
        itemset_1 = interaction_history[pair[0], :].flatten()
        itemset_2 = interaction_history[pair[1], :].flatten()
        dist += distance_of_mean_items(itemset_1, itemset_2, item_attributes) / len(pairs)
    return dist

class MeanInteractionDistance(Measurement):
    """
    Cacluates the mean distance between items in each users' recommendation list based on their item attributes
    This class inherits from :class:`.Measurement`.
    Parameters
    -----------
        verbose: bool (optional, default: False)
            If True, enables verbose mode. Disabled by default.
    Attributes
    -----------
        Inherited by Measurement: :class:`.Measurement`
        name: str (optional, default: "mean_rec_distance")
            Name of the measurement component.
    """
    def __init__(self, pairs, name="mean_interaction_dist", verbose=False):
        Measurement.__init__(self, name, verbose)
        self.pairs = pairs
        self.interaction_hist = None

    def measure(self, recommender):
        """
        Based on the pairings provided by the user, calculates the distance between
        the average item interacted by user 1 and user 2. These distances are averaged
        over all pairs. See mean_item_dist_pairs for more details.

        Parameters
        ------------
            recommender: :class:`~models.recommender.BaseRecommender`
                Model that inherits from
                :class:`~models.recommender.BaseRecommender`.
        """
        interactions = recommender.interactions
        if recommender.interactions.size == 0:
            # at beginning of simulation, there are no interactions
            self.observe(None)
            return
        if self.interaction_hist is None:
            self.interaction_hist = np.copy(interactions).reshape((-1, 1))
        else:
            self.interaction_hist = np.hstack([self.interaction_hist, interactions.reshape((-1, 1))])

        avg_dist = mean_item_dist_pairs(self.pairs, self.interaction_hist, recommender.actual_item_attributes)
        self.observe(avg_dist)

class MeanDistanceSimUsers(Measurement):
    """
    Cacluates the mean distance between items in each users' interaction list based on their item attributes
    This class inherits from :class:`.Measurement`.
    Parameters
    -----------
        verbose: bool (optional, default: False)
            If True, enables verbose mode. Disabled by default.
    Attributes
    -----------
        Inherited by Measurement: :class:`.Measurement`
        name: str (optional, default: "mean_rec_distance")
            Name of the measurement component.
    """
    def __init__(self, ideal_interaction_hist, ideal_item_attrs, seed=None, name="sim_user_dist", verbose=False):
        self.ideal_hist = ideal_interaction_hist
        self.ideal_item_attrs = ideal_item_attrs
        self.interaction_hist = None
        self.timestep = 0
        self.rng = Generator(seed)
        Measurement.__init__(self, name, verbose)

    def measure(self, recommender):
        """
        Based on pairs generated by finding the most similar user to each user (by cosine
        similarity of algorithmic representation), calculates the distance between
        the average item interacted by user 1 and user 2. These distances are averaged
        over all pairs. Finally, we subtract the same metric for the pairings provided from
        the ideal algorithmic simulation.

        See mean_item_dist_pairs for more details.

        Parameters
        ------------
            recommender: :class:`~models.recommender.BaseRecommender`
                Model that inherits from
                :class:`~models.recommender.BaseRecommender`.
        """
        if recommender.interactions.size == 0:
            # at beginning of simulation, there are no interactions
            self.observe(None)
            return
        interactions = recommender.interactions
        if self.interaction_hist is None:
            self.interaction_hist = np.copy(interactions).reshape((-1, 1))
        else:
            self.interaction_hist = np.hstack([self.interaction_hist, interactions.reshape((-1, 1))])

        # get value of user matrix
        user_representation = recommender.users_hat.state_history[-1]
        # find most similar users
        sim_matrix = cosine_similarity(user_representation, user_representation)
        # set diagonal entries to zero
        num_users = sim_matrix.shape[0]
        sim_matrix[np.arange(num_users), np.arange(num_users)] = 0
        # add random perturbation to break ties
        sim_tiebreak = np.zeros(
            sim_matrix.shape, dtype=[("score", "f8"), ("random", "f8")]
        )
        sim_tiebreak["score"] = sim_matrix
        sim_tiebreak["random"] = self.rng.random(sim_matrix.shape)
        # array where element x at index i represents the "most similar" user to user i
        closest_users = np.argsort(sim_tiebreak, axis=1, order=["score", "random"])[:, -1]
        pairs = list(enumerate(closest_users))
        # calculate average jaccard similarity
        ideal_hist = self.ideal_hist[:, :(self.timestep + 1)]
        ideal_dist = mean_item_dist_pairs(pairs, ideal_hist, self.ideal_item_attrs)
        this_dist = mean_item_dist_pairs(pairs, self.interaction_hist, recommender.actual_item_attributes)
        self.observe(this_dist - ideal_dist)
        self.timestep += 1 # increment timestep