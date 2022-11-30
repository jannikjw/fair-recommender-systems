import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../t-recs/')
from trecs.metrics.measurement import Measurement

# import math
import numpy as np
from itertools import combinations

class NoveltyMetric(Measurement):
    def __init__(self, name="novelty_metric", verbose=False):
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
        if recommender.interactions.size == 0:
            self.observe(None) # no interactions yet
            return
        # Indices for the items shown
        # items_shown = recommender.items_shown.flatten()
        """
        Need to implement it such that it subtracts the number of users who consumed
        the item in the current timestep (if we are following that part of Chen et. al.'s
        implementation).
        """
        """
        OLD
        # total number of users that have seen each of the items shown for all previous iterations
        num_users_for_items_shown = recommender.item_count[np.unique(recommender.items_shown)]
        # calculate novelty between each user and their presented item slate
        novelty = sum((-1) * np.log((num_users_for_items_shown*1.0) / recommender.num_users))
        # to complete the measurement, call `self.observe(metric_value)`
        """
        slate_items_self_info = recommender.item_count[recommender.items_shown]
        slate_items_self_info = (-1) * np.log(np.divide(slate_items_self_info, recommender.num_users))
        slate_items_pred_score = np.take_along_axis(recommender.users.actual_user_scores.value, recommender.items_shown, axis=1)
        slate_novelty = np.multiply(slate_items_self_info, slate_items_pred_score)
        slate_novelty = np.sum(slate_novelty, axis=1)
        self.observe(np.mean(slate_novelty))
        

class SerendipityMetric(Measurement):
    def __init__(self, name="serendipity_metric", verbose=False):
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
        if recommender.interactions.size == 0:
            self.observe(None) # no interactions yet
            return
        """
        OLD
        # Indices for the items shown
        items_shown = recommender.items_shown
        # Scores for the items shown
        user_scores = recommender.users.actual_user_scores.value
        # Scores for just the shown items that have a score greater than 0
        user_scores_items_shown = np.take_along_axis(user_scores, items_shown, axis=1) > 0
        # Topics that correspond to each item shown
        topics_shown = np.take_along_axis(np.broadcast_to(recommender.item_topics, (recommender.num_users, recommender.num_items)), items_shown, axis=1)

        # Need to update the below 2 lines depending on how user_topic_history is implemented in the wrapper class
        # Boolean matrix where value=1 if the topic shown is not in the user history, otherwise value=0
        new_topics = np.apply_along_axis(np.isin, 1, topics_shown, recommender.user_topic_history, invert=True)
        # calculate serendipity for all items presented to each user
        serendipity = np.sum(np.multiply(new_topics, user_scores_items_shown)) / recommender.num_users
        # to complete the measurement, call `self.observe(metric_value)`
        self.observe(serendipity)
        """
        # Scores for just the shown items that have a score greater than 0
        user_scores_items_shown = np.take_along_axis(recommender.users.actual_user_scores.value, recommender.items_shown, axis=1) > 0
        # Topics that correspond to each item shown
        topics_shown = recommender.item_topics[recommender.items_shown]
        # Boolean matrix where value=1 if the topic shown is not in the user history, otherwise value=0
        new_topics = np.apply_along_axis(np.isin, 1, topics_shown, recommender.user_topic_history, invert=True)
        # calculate serendipity for all items presented to each user
        items_shown_serendipity = np.multiply(new_topics, user_scores_items_shown)
        # Calculate average serendipity - average serendipity by slate AND users
        self.observe(np.mean(items_shown_serendipity))
        
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
        interactions = recommender.interactions
        if interactions.size == 0:
            self.observe(None) # no interactions yet
            return
        # Getting all possible 2-item combinations (the indices) for the items in a slate
        combos = combinations(np.arange(recommender.num_items_per_iter), 2)
        topic_similarity = np.zeros(recommender.num_users)
        for i in combos:
            # topic_similarity is equal to the number of 2-item combinations in which the items' topics are the same
            item_pair = recommender.items_shown[:, i]
            topic_pair = recommender.item_topics[item_pair]
            topic_similarity += (topic_pair[:,0] == topic_pair[:,1])

        slate_diversity = 1 - ((1 / (recommender.num_items_per_iter * (recommender.num_items_per_iter-1))) * topic_similarity)
        self.observe(np.mean(slate_diversity))

class TopicInteractionMeasurement(Measurement):
    """
    Keeps track of the interactions between users and topics.

    Specifically, at each timestep, it stores a histogram of length
    :math:`|I|`, where element :math:`i` is the number of interactions
    received by topic :math:`i`.

    Parameters
    -----------

        verbose: bool, default False
            If ``True``, enables verbose mode. Disabled by default.

    Attributes
    -----------
        Inherited by Measurement: :class:`.Measurement`

        name: str, default ``"topic_interaction_histogram"``
            Name of the measurement component.
    """

    def __init__(self, name="topic_interaction_histogram", verbose=False):
        Measurement.__init__(self, name, verbose)


    @staticmethod
    def _generate_interaction_histogram(interactions, num_users, num_topics):
        """
        Generates a histogram of the number of interactions per topics at the
        given timestep.

        Parameters
        -----------
            interactions : :obj:`numpy.ndarray`
                Array of user interactions.

            num_users : int
                Number of users in the system

            num_topics : int
                Number of topics in the system

        Returns
        ---------
            :obj:`numpy.ndarray`:
                Histogram of the number of interactions aggregated by items at the given timestep.
        """
        histogram = np.zeros(num_topics)
        print(interactions)
        np.add.at(histogram, interactions, 1)
        # Check that there's one interaction per user
        if histogram.sum() != num_users:
            raise ValueError("The sum of interactions must be equal to the number of users")
        return histogram


    def measure(self, recommender):
        """
        Measures and stores a histogram of the number of interactions per
        item at the given timestep.

        Parameters
        ------------
            recommender: :class:`~models.recommender.BaseRecommender`
                Model that inherits from :class:`~models.recommender.BaseRecommender`.
        """
        if recommender.interactions.size == 0:
            # at beginning of simulation, there are no interactions
            self.observe(None)
            return

        histogram = self._generate_interaction_histogram(
            recommender.user_topic_history, recommender.num_users, recommender.num_topics
        )
        self.observe(histogram, copy=True)
