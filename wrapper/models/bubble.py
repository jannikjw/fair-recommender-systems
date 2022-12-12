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
from src.scoring_functions import content_fairness
from pulp import *

class BubbleBurster(ContentFiltering):
    """
    
    Parameters
    -----------
            
        item_topics: array_like, size=(, num_items)
            Represents the topic cluster to which each item belongs
            
        num_topics: int, optional
            The number of topic clusters that the items are clustered into.
            If a value is supplied, it must be equal to the length of set(item_topics)
            
        user_topic_history: :obj:`numpy.ndarray`, optional
            A :math:`|U|\\times(num_topics)` matrix that represents the number of times 
            each topic has been interacted with/consumed by each user over all timesteps 
            up to the present timestep
                
        item_count: array_like, size=(, num_items), optional
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
            self.num_topics = np.unique(item_topics).size #len(set(item_topics))
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
            self.item_count = np.zeros((self.num_items)).astype(int)
        elif item_count.shape != (1, self.num_items):
            raise TypeError("item_count must have shape=(1, recommender.num_items)")
        else:
            self.item_count = item_count
        
    def _update_internal_state(self, interactions):
        ContentFiltering._update_internal_state(self, interactions)
        interacted_items = self.interactions
        # Updating `item_count`
        interacted_item_val, interacted_item_count = np.unique(interacted_items, return_counts=True)
        self.item_count[interacted_item_val] += interacted_item_count
        # Updating `user_topic_history`
        interacted_topics = np.expand_dims(self.item_topics[interacted_items], axis=1)
        self.topic_interactions = interacted_topics
        old_topic_count = np.take_along_axis(self.user_topic_history, interacted_topics, axis=1)
        np.put_along_axis(self.user_topic_history, interacted_topics, old_topic_count+1, axis=1)


    def generate_recommendations(self, k=1, item_indices=None):
        """
        Generate recommendations for each user.

        Parameters
        -----------

            k : int, default 1
                Number of items to recommend.

            item_indices : :obj:`numpy.ndarray`, optional
                A matrix containing the indices of the items each user has not yet
                interacted with. It is used to ensure that the user is presented
                with items they have not already interacted with. If `None`,
                then the user may be recommended items that they have already
                interacted with.

        Returns
        ---------
            Recommendations: :obj:`numpy.ndarray`
        """
        if item_indices is not None:
            if item_indices.size < self.num_users:
                raise ValueError(
                    "At least one user has interacted with all items!"
                    "To avoid this problem, you may want to allow repeated items."
                )
            if k > item_indices.shape[1]:
                raise ValueError(
                    f"There are not enough items left to recommend {k} items to each user."
                )
        if k == 0:
            return np.array([]).reshape((self.num_users, 0)).astype(int)
        # convert to dense because scipy does not yet support argsort - consider
        # implementing our own fast sparse version? see
        # https://stackoverflow.com/questions/31790819/scipy-sparse-csr
        # -matrix-how-to-get-top-ten-values-and-indices
        s_filtered = mo.to_dense(self.predicted_scores.filter_by_index(item_indices))
        row = np.repeat(self.users.user_vector, item_indices.shape[1])
        row = row.reshape((self.num_users, -1))
        if self.probabilistic_recommendations:
            permutation = s_filtered.argsort()
            rec = item_indices[row, permutation]
            # the recommended items will not be exactly determined by
            # predicted score; instead, we will sample from the sorted list
            # such that higher-preference items get more probability mass
            num_items_unseen = rec.shape[1]  # number of items unseen per user
            probabilities = np.logspace(0.0, num_items_unseen / 10.0, num=num_items_unseen, base=2)
            probabilities = probabilities / probabilities.sum()
            picks = np.random.choice(num_items_unseen, k, replace=False, p=probabilities)
            return rec[:, picks]

        elif self.score_fn == content_fairness:
            permutation = s_filtered.argsort()
            rec = item_indices[row, permutation]
            num_items_unseen = rec.shape[1]  # number of items unseen per user
            probabilities = np.logspace(0.0, num_items_unseen / 10.0, num=num_items_unseen, base=2)
            probabilities = probabilities / probabilities.sum()

            slate_size = k
            upper_bound = 0.75
            rec = np.empty((self.num_users, slate_size), dtype=int)
            for i in range(self.num_users):
                items = list(range(self.num_items))
                sizes = dict(zip(items, [1] * len(items)))
                weights = dict(zip(items, gw))
                probs = dict(zip(items, probabilities[i]))

                picked_vars = LpVariable.dicts("", items, lowBound=0, upBound=1, cat='Integer')

                total_score = LpProblem("Fair_Recs_Problem", LpMaximize)
                total_score += lpSum([probs[i] * picked_vars[i] for i in picked_vars])

                total_score += lpSum([sizes[i] * picked_vars[i] for i in picked_vars]) == slate_size
                total_score += lpSum([weights[i] * picked_vars[i] for i in picked_vars]) <= [upper_bound] * num_topics
                total_score.solve()

                rec[i] = [int(v.name[1:]) for v in total_score.variables() if v.varValue > 0]

            return rec
        else:
            # returns top k indices, sorted from greatest to smallest
            sort_top_k = mo.top_k_indices(s_filtered, k, self.random_state)
            # convert top k indices into actual item IDs
            rec = item_indices[row[:, :k], sort_top_k]
            if self.is_verbose():
                self.log(f"Item indices:\n{str(item_indices)}")
                self.log(
                    f"Top-k items ordered by preference (high to low) for each user:\n{str(rec)}"
                )
            print(rec.shape)
            return rec