import math
import numpy as np

def topic_similarity(topics, x, y):
    """
    Assumptions:
    -   Item object has an attribute num_users, which is a count of the number of users that have 
        consumed this piece of content
    """
    if topics[x] == topics[y]:
        return 1
    else:
        return 0

def calculate_diversity(topics, slate): #TODO: Vectorize this
    sum_similarity = 0
    for i in range(len(slate)):
        for j in range(len(slate)):
            if i != j:
                sum_similarity += topic_similarity(topics, slate[i], slate[j])
    
    return 1 - 1 / (len(slate)**2 - len(slate)) * sum_similarity


def self_information(item_users, item, num_users):
    """
    Item-level self-information - measures the unexpectedness of a recommended item relative to its global popularity
    *  Assume *  Item object has an attribute num_users, which is a count of the number of users that have consumed this piece of content
            Note that num_users is an attribute defined in the ActualUserProfiles class
    """
    return (-1) * math.log((item_users*1.0) / num_users)    


def calculate_novelty(slate, num_users, interaction_matrix):
    """
    Metric to capture the global popularity-based measurements. Computing the novelty of a slate that is presented to a user.
    """
    return sum([self_information(sum(interaction_matrix[item]), item, num_users) for item in slate])


def item_serendipity(item, user):
    """
    Item-level serendipity
    TO DO:  Define some metric that evaluates item-wise serendipity. This could incorporate our previous implementations for 
            self_information and similarity/dissimilarity (the latter of which could be w.r.t. user.state_history)
    """
    return 0


def calculate_serendipity(slate, user):
    """
    Metric to capture the unexpectedness/surprise of the recommendation to a specific user
    """
    return sum([item_serendipity(item, user) for item in slate])


def leaning_distance(item1, item2):
    """
    Function to compute the distance in leaning between two items
    """
    dist = math.abs(item1.leaning - item2.leaning)
    return dist**2


def calculate_spread(slate):
    """
    Metric to capture the spread of *leanings* in a given slate of items
    """
    sorted = np.sort(slate)
    return sum([leaning_distance(slate[i-1], slate[i]) for i in slate[1:]])