import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../t-recs/')
from trecs.validate import validate_user_item_inputs
from trecs.models import BaseRecommender
from trecs.models import ContentFiltering
from trecs.random import Generator
import trecs.matrix_ops as mo

from models.bubble import BubbleBurster

import numpy as np
from numpy.linalg import norm
import scipy.sparse as sp

class MyopicExcludeK(BubbleBurster):
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
        
        excludeK: int
            The top `k` recommendations (i.e., the `k` recommendations with the highest score for a user)
            that should be excluded from generated slate of recommendations
    
    Attributes
    -----------
    
        Inherited from BubbleBurster: :class:`~models.bubble.BubbleBurster`
        
        Inherited from ContentFiltering: :class:`~models.content.ContentFiltering`
        
        Inherited from BaseRecommender: :class:`~models.recommender.BaseRecommender`
        
        excludeK: same as parameter 'excludeK'
    
    """
    # We define default values in the signature so we can call the constructor with no argument

    def __init__(
        self, 
        excludeK=None,
        **kwargs
    ):
        super().__init__(**kwargs)
            
        # Initializing 'num_topics' attribute
        if excludeK == None:
            raise TypeError("Must supply a value for 'excludeK' - that's the entire point of this model, dummy!")
        elif excludeK >= self.num_items:
            raise TypeError("Value supplied for 'excludeK' must be less than or equal to num_items")
        elif excludeK >= self.num_items/2:
            print("WARNING: Excluding a lot of items there, bud")
        
        self.excludeK = excludeK
        
        
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
        print(self.excludeK)
        super.generate_recommendations()
    #     if item_indices is not None:
    #         if item_indices.size < self.num_users:
    #             raise ValueError(
    #                 "At least one user has interacted with all items!"
    #                 "To avoid this problem, you may want to allow repeated items."
    #             )
    #         if k > item_indices.shape[1]:
    #             raise ValueError(
    #                 f"There are not enough items left to recommend {k} items to each user."
    #             )
    #     if k == 0:
    #         return np.array([]).reshape((self.num_users, 0)).astype(int)
    #     # convert to dense because scipy does not yet support argsort - consider
    #     # implementing our own fast sparse version? see
    #     # https://stackoverflow.com/questions/31790819/scipy-sparse-csr
    #     # -matrix-how-to-get-top-ten-values-and-indices
    #     s_filtered = mo.to_dense(self.predicted_scores.filter_by_index(item_indices))
    #     row = np.repeat(self.users.user_vector, item_indices.shape[1])
    #     row = row.reshape((self.num_users, -1))
    #     if self.probabilistic_recommendations:
    #         permutation = s_filtered.argsort()
    #         rec = item_indices[row, permutation]
    #         # the recommended items will not be exactly determined by
    #         # predicted score; instead, we will sample from the sorted list
    #         # such that higher-preference items get more probability mass
    #         num_items_unseen = rec.shape[1]  # number of items unseen per user
    #         probabilities = np.logspace(0.0, num_items_unseen / 10.0, num=num_items_unseen, base=2)
    #         probabilities = probabilities / probabilities.sum()
    #         picks = np.random.choice(num_items_unseen, k, replace=False, p=probabilities)
    #         return rec[:, picks]
    #     else:
    #         # returns top k indices, sorted from greatest to smallest
    #         sort_top_k = mo.top_k_indices(s_filtered, k, self.random_state)
    #         # convert top k indices into actual item IDs
    #         rec = item_indices[row[:, :k], sort_top_k]
    #         if self.is_verbose():
    #             self.log(f"Item indices:\n{str(item_indices)}")
    #             self.log(
    #                 f"Top-k items ordered by preference (high to low) for each user:\n{str(rec)}"
    #             )
    #         return rec
        
    # # def next_k_myopic_scoring(predicted_user_profiles, predicted_item_attributes):
    # #     # alpha is equal to k, the number of items that we wish to exclude from recommending
    # #     alpha = 10 #ALPHA
    # #     pred_scores = mo.inner_product(predicted_user_profiles, predicted_item_attributes)
    # #     sorted_index_array = np.argsort(predicted_user_profiles)
    # #     rs = RandomState(42)
    # #     top_k_idxs = mo.top_k_indices(matrix=pred_scores, k=alpha, random_state=rs)
    # #     re_ranked_scores = pred_scores
    # #     np.put_along_axis(arr=re_ranked_scores, indices=top_k_idxs, values=0, axis=1)
    # #     return re_ranked_scores