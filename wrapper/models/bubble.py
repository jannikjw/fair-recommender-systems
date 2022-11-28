import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../t-recs/')
from trecs.validate import validate_user_item_inputs
from trecs.models import BaseRecommender
from trecs.models import ContentFiltering
from trecs.random import Generator

import numpy as np
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
        
    def _update_internal_state(self, interactions):
        ContentFiltering._update_internal_state(self, interactions)
        items_shown = self.items_shown
        for i in range(items_shown.shape[0]):
            items_shown_val, items_shown_count = np.unique(items_shown[i,:], return_counts=True)
            self.item_count[0, items_shown_val] += 1
            topics_shown = self.item_topics[items_shown_val]
            topics_shown_val, topics_shown_count = np.unique(topics_shown, return_counts=True)
            self.user_topic_history[i, topics_shown_val] += topics_shown_count
            if (sum(items_shown_count) != 10):
                print("DUPLICATE ITEMS IN SLATE", items_shown_count)
                break
        # return item_count, user_topic_history