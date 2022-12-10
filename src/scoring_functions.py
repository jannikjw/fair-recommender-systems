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
    # add minimum value of each row to all item scores to ensure scores are positive.
    re_ranked_scores += np.abs(np.min(re_ranked_scores, axis=1))[:, np.newaxis]

    assert (re_ranked_scores >= 0).all(), "Some scores are negative."
    return re_ranked_scores


def entropy(predicted_user_profiles, predicted_item_attributes):
    # Reranking
    alpha = globals.ALPHA
    predicted_scores = mo.inner_product(predicted_user_profiles, predicted_item_attributes)
    
    entropy = - predicted_scores * np.log(predicted_scores + globals.EPS)
    
    re_ranked_scores = predicted_scores + alpha * entropy
    assert (re_ranked_scores >= 0).all(), "Some scores are negative."

    return re_ranked_scores


def content_fairness(predicted_user_profiles, predicted_item_attributes):
    slate_size = 10
    upper_bound = 0.75


    predicted_scores =  mo.inner_product(predicted_user_profiles, predicted_item_attributes)
    probs = (predicted_scores.T / np.sum(predicted_scores, axis=1)).T

    probs_sorted = np.flip(np.argsort(probs, axis=1), axis=1)
    
    gw = predicted_item_attributes.T / np.sum(predicted_item_attributes.T, axis=1)[:, np.newaxis]

    num_user = len(predicted_user_profiles)
    recs = np.empty((num_user, slate_size))
    for user in range(len(probs_sorted)):
        agg_weight_per_cluster = np.zeros((len(gw[0])))
        i = 0
        for item in probs_sorted[user]:
            weight_item = gw[item]
            # print(f'{item}: {weight_item}')
            proposed_weights = weight_item + agg_weight_per_cluster
            if (proposed_weights <= upper_bound).all() and i < slate_size:
                agg_weight_per_cluster = proposed_weights
                recs[user, i] = item
                i += 1

    predicted_scores_reranked = np.copy(predicted_scores)
    for i, user in enumerate(recs):
        for item in np.flip(user):
            predicted_scores_reranked[int(i), int(item)] = np.max(predicted_scores_reranked[int(i)]) + 1

    return predicted_scores_reranked
