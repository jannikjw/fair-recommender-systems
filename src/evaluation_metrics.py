from trecs.metrics import Measurement

import math
import numpy as np

class NoveltyMetric(Measurement):
    def __init__(self, name="novlety_metric", verbose=False):
        Measurement.__init__(self, name, verbose)
        
    def measure(self, recommender):
        """
        The purpose of this metric is to capture the global popularity-based measurements
        - computing the average novelty of all slates that are presented to users at the current timestep

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
        # total number of users that have seen each of the items shown for all previous iterations
        num_users_for_items_shown = recommender.item_count[items_shown]
        # calculate novelty between each user and their presented item slate
        novelty = sum((-1) * math.log((num_users_for_items_shown*1.0) / self.num_users))
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
        topics_shown = np.take_along_axis(np.broadcast_to(recommender.topics, (self.num_users, self.num_items)), items_shown, axis=1)
        # Boolean matrix where value=1 if the topic shown is not in the user history, otherwise value=0
        new_topics = np.apply_along_axis(np.isin, 1, topics_shown, self.user_topic_history, invert=True)
        # calculate serendipity for all items presented to each user
        serendipity = np.sum(np.multiply(new_topics, user_scores_items_shown)) / self.num_users
        # to complete the measurement, call `self.observe(metric_value)`
        self.observe(serendipity)