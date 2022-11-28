import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../t-recs/')
from trecs.validate import validate_user_item_inputs
from trecs.models import BaseRecommender
from trecs.models import ContentFiltering
from trecs.random import Generator
import trecs.matrix_ops as mo

import numpy as np
from numpy.linalg import norm
import scipy.sparse as sp

class BubbleBurster(ContentFiltering):
    """
    
    Parameters
    -----------
            
        item_topics: array_like, size=(1 x num_items)
            Represents the topic cluster to which each item belongs
            
        num_topics: int, optional
            The number of topic clusters that the items are clustered into.
            If a value is supplied, it must be equal to the length of set(item_topics)
            
        user_topic_history: :obj:`numpy.ndarray`, optional
            A :math:`|U|\\times(num_topics)` matrix that represents whether a user has been shown an item that 
            belongs to topic for all 'topic' in 'set(item_topics)', up to and including the current timestep
            Elements consist of boolean, 0 or 1, values, such that: 
                user_topic_history[i,j] = 1: if topic has been presented in any past-present slate
                user_topic_history[i,j] = 0: if topic has not been presented in any past-present slate
                
        item_count: array_like, size=(1 x num_items), optional
            Represents the total number of users that have consumed each of the items 
            up to and including the current timestep
    
    Attributes
    -----------
    
        Inherited from ContentFiltering: :class:`~models.content.ContentFiltering`
        
        Inherited from BaseRecommender: :class:`~models.recommender.BaseRecommender`
        
        item_topics: same as parameter 'item_topics'
        
        num_topics: same as parameter 'num_topics'
        
        user_topic_history: same as parameter 'user_topic_history'
        
        item_count: same as parameter 'item_count'
    
    """
    # We define default values in the signature so we can call the constructor with no argument

    def __init__(
        self, 
        num_topics=None,
        item_topics=None,
        user_topic_history=None,
        item_count=None,
        **kwargs
    ):
        
        super().__init__(**kwargs)

        # Initializing 'item_topics' attribute
        if (not isinstance(item_topics, (list, np.ndarray, sp.spmatrix))):
            raise TypeError("Must supply array_like object for 'item_topics'")
        elif type(item_topics) == None:
            raise TypeError("'item_topics' cannot be None")
        elif len(item_topics) != self.num_items:
            raise TypeError("number of item-topic values must be equal to num_items for 'item_topics'")
        else:
            self.item_topics = item_topics
            
        # Initializing 'num_topics' attribute
        if num_topics == None:
            self.num_topics = len(set(item_topics))
        elif num_topics == len(set(item_topics)):
            num_topics = num_topics
        else:
            raise TypeError("Must supply an int value for 'num_topics'")

        # Initializing 'user_topic_history' attribute
        if user_topic_history == None:
            self.user_topic_history = np.zeros((self.num_users, self.num_topics))
        elif user_topic_history.shape != (self.num_users, self.num_topics) or ~((user_topic_history!=0) & (user_topic_history!=1)).any():
            raise TypeError("'user_topic_history' must be a binary ndarray with shape=(num_users, num_topics)")
        else:
            self.user_topic_history = user_topic_history
        
        # Initializing 'item_count' attribute
        if item_count == None:
            self.item_count = np.zeros((1,self.num_items))
        elif item_count.shape != (1, self.num_items):
            raise TypeError("item_count must have shape=(1, recommender.num_items)")
        else:
            self.item_count = item_count
        
    def _calculate_cosine_similarities(self, slate):
        """
        Calculates cosine similarity for a set of recommendations.
        Inputs:
            slate: a list of recommendations
            item_representation: a matrix of item representations
        Outputs:
            cosine_similarity: mean average precision
        """
        cosine_similarities = [] #TODO: Vecotrize (select item representation vectors from item represetnations and then multiply transposes)
        for item_id in slate:
            cosine_similarity = 0
            for item_id_2 in slate:
                if item_id != item_id_2:
                    vec_1 = self.items_hat[:, item_id]
                    vec_2 = self.items_hat[:, item_id_2]
                    vec_prod = np.dot(vec_1, vec_2) / (norm(vec_1) * norm(vec_2))
                    cosine_similarity += vec_prod
            cosine_similarities.append(cosine_similarity)
        return cosine_similarities


    def _re_rank_scores(self, recommendations):
        """
        Re-ranks scores for a set of recommendations.
        Inputs:
            item_representation: a matrix of item representations
            recommendations: a list of recommendations
        Outputs:
            re_ranked_recommendations: a list of re-ranked recommendations
        """
        exps = [np.round(x * 0.1, 1) for x in range(0, len(recommendations[0]))][::-1]
        eps = 0.00001
        initial_scores = np.exp(exps)
        re_ranked_recommendations = np.zeros_like(recommendations)
        
        for i, slate in enumerate(recommendations):
            # print(f"Slate:\t\t\t{slate}")
            cosine_similarities = self._calculate_cosine_similarities(slate)
            # multiply cosine_similarities with each list in recommendations
            re_ranked_scores = initial_scores * 1/(eps + np.power(cosine_similarities,2))
            # print(f'Initial Scores:\t\t{np.round(initial_scores, 2)}')
            # print(f'Re-ranked scores:\t{np.round(re_ranked_scores, 2)}')
            tup = list(zip(slate, re_ranked_scores))
            tup.sort(key = lambda x: x[1], reverse=True)
            # create list from second element in each tuple in tup
            re_ranked_slate = np.array([x[0] for x in tup])
            # print(f"Re-ranked Slate:\t{re_ranked_slate}")
            re_ranked_recommendations[i] = re_ranked_slate

        return re_ranked_recommendations