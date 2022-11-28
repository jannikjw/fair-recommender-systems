from sklearn.cluster import KMeans

def get_topic_clusters(interaction_matrix, n_clusters:int=100, n_attrs:int=100, max_iter:int=100, nmf_solver:str="mu"):
    """
    Creates clusters of movies based on their genre.
    Inputs:
        binary_ratings_matrix: a binary matrix of users and movies
        n_attrs: number of attributes to use in NMF
        nmf_solver: solver to use in NMF
    Outputs:
        clusters: a list of cluster assignments
    """
    # Create topic clusters
    #create co-occurence matrix from binary_interaction_matrix
    co_occurence_matrix = interaction_matrix.T @ interaction_matrix
    co_occurence_matrix

    # Matrix factorize co_occurence_matrix to get embeddings
    nmf_cooc = NMF(n_components=n_attrs, solver=nmf_solver, max_iter=max_iter)
    W_topics = nmf_cooc.fit_transform(co_occurence_matrix)

    # cluster W_topics
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(W_topics)

    # assign nearest cluster to observation
    cluster_ids = kmeans.predict(W_topics)

    return cluster_ids