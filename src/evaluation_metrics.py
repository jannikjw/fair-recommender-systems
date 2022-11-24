import math

def topic_similarity(x, y):
    """
    Assumptions:
    -   Item object has an attribute num_users, which is a count of the number of users that have 
        consumed this piece of content
    """
    if topic[x] == topic[y]:
        return 1
    else:
        return 0

def slate_diversity(slate):
    sum_similarity = 0
    for i in range(len(slate)):
        for j in range(len(slate)):
            if i != j:
                sum_similarity += topic_similarity(slate[i], slate[j])
    
    return 1 - 1 / (len(slate)**2 - len(slate)) * sum_similarity


def self_information(item, num_users):
    """
    Item-level self-information - measures the unexpectedness of a recommended item relative to its global popularity
    *  Assume *  Item object has an attribute num_users, which is a count of the number of users that have consumed this piece of content
            Note that num_users is an attribute defined in the ActualUserProfiles class
    """
    return (-1) * math.log((item.num_users*1.0) / num_users)    


def novelty(slate, num_users):
    """
    Metric to capture the global popularity-based measurements. Computing the novelty of a slate that is presented to a user.
    """
    return sum([self_information(item, num_users) for item in slate])


def item_serendipity(item, user):
    """
    Item-level serendipity
    TO DO:  Define some metric that evaluates item-wise serendipity. This could incorporate our previous implementations for 
            self_information and similarity/dissimilarity (the latter of which could be w.r.t. user.state_history)
    """
    return


def serendipity(slate, user):
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


def spread(slate):
    """
    Metric to capture the spread of *leanings* in a given slate of items
    """
    sorted = slate.sorted()
    return sum([leaning_distance(slate[i-1], slate[i]) for i in slate[1:]])