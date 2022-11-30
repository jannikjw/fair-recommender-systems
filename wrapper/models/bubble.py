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
            self.user_topic_history = np.zeros((self.num_users, self.num_topics), dtype=int)
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
        old_topic_count = np.take_along_axis(self.user_topic_history, interacted_topics, axis=1)
        np.put_along_axis(self.user_topic_history, interacted_topics, old_topic_count+1, axis=1)