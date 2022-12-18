import trecs.matrix_ops as mo
import scipy.spatial as sp
import numpy as np
from numpy.linalg import norm
from scipy import stats
import src.globals as globals

from numpy.random import RandomState

rs = RandomState(42)


def cosine_sim(predicted_user_profiles, predicted_item_attributes):
    # Reranking
    alpha = globals.ALPHA

    predicted_scores = mo.inner_product(predicted_user_profiles,
                                        predicted_item_attributes)
    user_norms = norm(predicted_user_profiles, axis=1)
    item_norms = norm(predicted_item_attributes, axis=0)

    # create a matrix that contains the outer product af user_norms and item_norms
    norms = np.outer(user_norms, item_norms)
    if (norms == 0).any():
        return predicted_scores

    cosine_similarities = predicted_scores / norms
    re_ranked_scores = predicted_scores - alpha * cosine_similarities
    # add minimum value of each row to all item scores to ensure scores are positive.
    # re_ranked_scores += np.abs(np.min(re_ranked_scores, axis=1))[:, np.newaxis]
    re_ranked_scores = np.exp(re_ranked_scores)

    assert (re_ranked_scores >= 0).all(), "Some scores are negative."
    return re_ranked_scores


def entropy(predicted_user_profiles, predicted_item_attributes):
    # Reranking
    alpha = globals.ALPHA

    predicted_scores = mo.inner_product(predicted_user_profiles,
                                        predicted_item_attributes)
    if np.sum(predicted_scores) == 0:
        return predicted_scores

    predicted_probs = predicted_scores / np.sum(predicted_scores,
                                                 axis=1)[:, np.newaxis]
    
    # entropy = -np.sum(predicted_probs * np.log(predicted_probs + globals.EPS), axis=1)[:, np.newaxis]
    # re_ranked_scores = predicted_scores + alpha * entropy

    entropy_deltas = np.zeros_like(predicted_probs)
    set_entropy = stats.entropy(predicted_probs, axis=1)
    for i in range(predicted_probs.shape[1]):
        entropy_deltas[:, i] = np.max(set_entropy - stats.entropy(np.delete(predicted_probs, i, axis=1), axis=1), 0)
    
    re_ranked_scores = predicted_scores + alpha * entropy_deltas
    
    assert (entropy_deltas >= 0).all(), "Some entropy values are negative."
    assert (re_ranked_scores >= 0).all(), "Some scores are negative."
    
    return re_ranked_scores


def content_fairness(predicted_user_profiles, predicted_item_attributes):
    """
    A score function that ensures that no topic is overrepresented. 
    1. Scores are predicted using the inner product (myopic).
    2. Probabilities are created based on these scores to ensure common value range
    3. The weight of an item to a topic is determined based on its relative value in its embeddings
    4. A slate is created based on myopic scores.
    5. If adding a new item to the slate would exceed the upper bound in a topic dimension, the item is not added.
       Instead, the function proceeds to the next available item until one is found that does not exceed any weights
       in the topic dimensions.
    6. To ensure that the items appear in the top-k slate picked by t-recs, the scores of the slate items are
       manually increased to be in the top-k.
    
    Caveats: Upper_bound and slate_size are set manually because default t-recs implementation does not allow
    passing additional parameters to a score function. The current implementation only works with a top-k slate (non-probabilistic). Also scores are manually altered which reduces interpretability of the scores.
    
    Based on "Controlling Polarization in Personalization: An Algorithmic Framework" by Celis et. al.
    
    Inputs:
        - predicted_user_profiles
        - pretedicted_item_attributes
    Outputs:
        - predicted_scores_reranked (Shape: U x I)
    """
    slate_size = 10
    upper_bound = 0.75

    # 1. Calculate myopic scores
    predicted_scores = mo.inner_product(predicted_user_profiles,
                                        predicted_item_attributes)


    return predicted_scores


def top_k_reranking(predicted_user_profiles, predicted_item_attributes):
    """
    A score function that reverses the scores assigned for the number of items in the iteration,
    such that: the item with the 10th highest score is assigned the score of the item with the first highest score, etc.
        i.e., item_score[0] -> item_score[9] && item_score[9] -> item_score[0]
    
    Inputs:
        - predicted_user_profiles
        - pretedicted_item_attributes
    Outputs:
        - predicted_scores_reranked (Shape: U x I)
    """
    k = 10

    pred_scores = mo.inner_product(predicted_user_profiles,
                                   predicted_item_attributes)
    top_k_idxs = mo.top_k_indices(matrix=pred_scores, k=k, random_state=rs)

    top_k_re_ranked_idxs = top_k_idxs[:, -1::-1]
    top_k_re_ranked_scores = np.take_along_axis(pred_scores,
                                                top_k_idxs,
                                                axis=1)
    top_k_re_ranked_scores = top_k_re_ranked_scores[:, -1::-1]

    re_ranked_scores = np.copy(pred_scores)
    np.put_along_axis(re_ranked_scores,
                      top_k_idxs,
                      top_k_re_ranked_scores,
                      axis=1)

    return re_ranked_scores
