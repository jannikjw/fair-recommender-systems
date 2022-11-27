import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../t-recs/')
from trecs.validate import validate_user_item_inputs
from trecs.models import BaseRecommender
from trecs.models import ContentFiltering
from trecs.random import Generator

import numpy as np

class BubbleBurster(ContentFiltering):
    # We define default values in the signature so we can call the constructor with no argument
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    # def _update_internal_state(self, interactions):
    #     # ...