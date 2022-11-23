import math

"""
Assumptions:
-   Item object has an attribute n_users, which is a count of the number of users that have 
    consumed this piece of content
"""

def similarity(x, y):
    if x == y:
        return 1
    else:
        return 0

def slate_diversity(slate):
    sum_similarity = 0
    for i in range(len(slate)):
        for j in range(len(slate)):
            if i != j:
                sum_similarity += similarity(slate[i], slate[j])
    
    return 1 - 1 / (len(slate)**2 - len(slate)) * sum_similarity

"""
Item-level self-information - measures the unexpectedness of a recommended item relative to its global popularity
*Assume Item object has an attribute n_users, which is a count of the number of users that have consumed this piece of content
"""
def self_information(item, n_users):
    return (-1) * math.log((item.n_users*1.0) / n_users)    

"""
Metric to capture the global popularity-based measurements. Computing the novelty of a slate that is presented to a user.
"""
def novelty(slate, n_users):
    return sum([self_information(item, n_users) for item in slate])

def item_serendipity(item, user):
    return

"""
Metric to capture the unexpectedness/surprise of the recommendation to a specific user
"""
def serendipity(slate, user):
    return sum([item_serendipity(item, user) for item in slate])

def spread():
    return