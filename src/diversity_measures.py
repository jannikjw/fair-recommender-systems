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