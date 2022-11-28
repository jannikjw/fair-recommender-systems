import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../t-recs/')
from trecs.metrics import Measurement

import math
import numpy as np
from itertools import combinations

class NoveltyMetric(Measurement):
    def __init__(self, name="novlety_metric", verbose=False):
        Measurement.__init__(self, name, verbose)
        
    def measure(self, recommender):
        """
        The purpose of this metric is to capture the global popularity-based measurements
        - computing the average novelty of all slates that are presented to users at the current timestep
        
        This metric is based on the item diversity measure used in :
        Minmin Chen, Yuyan Wang, Can Xu, Ya Le, Mohit Sharma, Lee Richardson, Su-Lin Wu, and Ed Chi. 
        Values of user exploration in recommender systems. 
        In Proceedings of the 15th ACM Conference on Recommender Systems, 
        RecSys ’21, page 85–95, New York, NY, USA, 2021. Association for Computing Machinery.
        
        Parameters
        ------------
            recommender: :class:`~models.recommender.BaseRecommender`
                Model that inherits from
                :class:`~models.recommender.BaseRecommender`.
        """
        interactions = recommender.interactions
        if interactions.size == 0:
            self.observe(None) # no interactions yet
            return
        # Indices for the items shown
        items_shown = recommender.items_shown.flatten()
        """
        Need to implement it such that it subtracts the number of users who consumed
        the item in the current timestep (if we are following that part of Chen et. al.'s
        implementation).
        """
        # total number of users that have seen each of the items shown for all previous iterations
        num_users_for_items_shown = recommender.item_count[items_shown]
        # calculate novelty between each user and their presented item slate
        novelty = sum((-1) * math.log((num_users_for_items_shown*1.0) / recommender.num_users))
        # to complete the measurement, call `self.observe(metric_value)`
        self.observe(novelty.mean())
        

class SerendipityMetric(Measurement):
    def __init__(self, name="novlety_metric", verbose=False):
        Measurement.__init__(self, name, verbose)
        
    def measure(self, recommender):
        """
        Metric to capture the unexpectedness/surprise of the recommendation to a specific user.
        Item-wise serendipity is equal to 1 if, for that specific user, that item is associated 
        with a score greater than 0 and the topic cluster of that item is not present in the 
        topic history for that user. Otherwise, item-wise serendipity is equal to 0.
        Global serendipity for an interation is computed as the summation of all item-wise serendipity
        scores divided by the number of users.
        
        This metric is based on the item diversity measure used in :
        Minmin Chen, Yuyan Wang, Can Xu, Ya Le, Mohit Sharma, Lee Richardson, Su-Lin Wu, and Ed Chi. 
        Values of user exploration in recommender systems. 
        In Proceedings of the 15th ACM Conference on Recommender Systems, 
        RecSys ’21, page 85–95, New York, NY, USA, 2021. Association for Computing Machinery.
        
        Parameters
        ------------
            recommender: :class:`~models.recommender.BaseRecommender`
                Model that inherits from
                :class:`~models.recommender.BaseRecommender`.
        """
        interactions = recommender.interactions
        if interactions.size == 0:
            self.observe(None) # no interactions yet
            return
        # Indices for the items shown
        items_shown = recommender.items_shown
        # Scores for the items shown
        user_scores = recommender.users.actual_user_scores.value
        # Scores for just the shown items that have a score greater than 0
        user_scores_items_shown = np.take_along_axis(user_scores, items_shown, axis=1) > 0
        # Topics that correspond to each item shown
        topics_shown = np.take_along_axis(np.broadcast_to(recommender.item_topics, (recommender.num_users, recommender.num_items)), items_shown, axis=1)
        """
        Need to update the below 2 lines depending on how user_topic_history is implemented in the wrapper class
        """
        # Boolean matrix where value=1 if the topic shown is not in the user history, otherwise value=0
        new_topics = np.apply_along_axis(np.isin, 1, topics_shown, recommender.user_topic_history, invert=True)
        # calculate serendipity for all items presented to each user
        serendipity = np.sum(np.multiply(new_topics, user_scores_items_shown)) / recommender.num_users
        # to complete the measurement, call `self.observe(metric_value)`
        self.observe(serendipity)
        
class DiversityMetric(Measurement):
    def __init__(self, name="diversity_metric", verbose=False):
        Measurement.__init__(self, name, verbose)
        
    def measure(self, recommender):
        """
        Measures the number of distinct faucets the recommendation set contains, which is measured as 
        the average dissimilarity of all pairs of items in the set is a popular choice.,
        such that: similarity(i, j) = 1 if i and j belongs to the same topic cluster, and 0 otherwise.
        
        This metric is based on the item diversity measure used in :
        Minmin Chen, Yuyan Wang, Can Xu, Ya Le, Mohit Sharma, Lee Richardson, Su-Lin Wu, and Ed Chi. 
        Values of user exploration in recommender systems. 
        In Proceedings of the 15th ACM Conference on Recommender Systems, 
        RecSys ’21, page 85–95, New York, NY, USA, 2021. Association for Computing Machinery.
        
        Parameters
        ------------
            recommender: :class:`~models.recommender.BaseRecommender`
                Model that inherits from
                :class:`~models.recommender.BaseRecommender`.
        """

        combos = combinations(np.arange(recommender.num_items_per_iter), 2)
        items_shown = recommender.items_shown

        stop = 0
        slate_diversity = np.zeros(recommender.num_users)
        for i in combos:
            item_pair = items_shown[:, i]
            topic_pair = recommender.item_topics[item_pair]
            topic_similarity = (topic_pair[:,0] != topic_pair[:,1])
            slate_diversity += topic_similarity
        
        diversity = 1 - (1 / (recommender.num_items_per_iter) * (recommender.num_items_per_iter-1))
        diversity *= np.sum(slate_diversity)
        self.observe(diversity)