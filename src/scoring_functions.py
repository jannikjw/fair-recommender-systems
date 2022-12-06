import trecs.matrix_ops as mo
import scipy.spatial as sp
import numpy as np
from numpy.linalg import norm
import src.globals as globals

def cosine_sim(predicted_user_profiles, predicted_item_attributes):
    # Reranking
    alpha = globals.ALPHA
    predicted_scores = mo.inner_product(predicted_user_profiles, predicted_item_attributes)
    # create a vector that contains the norms of all row vectors in predicted_user_profiles
    user_norms = norm(predicted_user_profiles, axis=1)
    item_norms = norm(predicted_item_attributes, axis=0)

    # create a matrix that contains the outer product aAf user_norms and item_norms
    norms = np.outer(user_norms, item_norms)
    cosine_similarities = predicted_scores / norms
    cosine_similarities = np.nan_to_num(cosine_similarities)
    # print max value of norms
    re_ranked_scores = predicted_scores - alpha * cosine_similarities
    return re_ranked_scores

def entropy(predicted_user_profiles, predicted_item_attributes):
    # Reranking
    alpha = globals.ALPHA
    predicted_scores = mo.inner_product(predicted_user_profiles, predicted_item_attributes)
    
    entropy = - predicted_scores * np.log(predicted_scores + globals.EPS)
    
    re_ranked_scores = predicted_scores + alpha * entropy
    return re_ranked_scores