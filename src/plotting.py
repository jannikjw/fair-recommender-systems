import matplotlib.pyplot as plt

def plot_measurements(measurements_df):

    fig, ax = plt.subplots(2, 3, figsize=(15, 6))
    fig.tight_layout(h_pad=3)

    measurements_df['rec_similarity'].plot(ax=ax[0, 0])
    measurements_df['interaction_similarity'].plot(ax=ax[0, 1])
    measurements_df['mean_num_topics'].plot(ax=ax[0, 2])
    measurements_df['serendipity_metric'].plot(ax=ax[1, 0])
    measurements_df['novelty_metric'].plot(ax=ax[1, 1])
    measurements_df['diversity_metric'].plot(ax=ax[1, 2])

    for a in ax:
        for b in a:
            b.set_xlabel('Timestep')

    ax[0, 0].set_title('Recommendation similarity')
    ax[0, 0].set_ylabel('Similarity')
    
    ax[0, 1].set_title('Interaction Similarity')
    ax[0, 1].set_ylabel('Jaccard Similarity')

    ax[0, 2].set_title('Mean Number of Topics per User')
    ax[0, 2].set_ylabel('Mean Number of Topics per User')
    
    ax[1, 0].set_title('Serendipity')
    ax[1, 0].set_ylabel('Serendipity')
    
    ax[1, 1].set_title('Novelty')
    ax[1, 1].set_ylabel('Novelty')

    ax[1, 2].set_title('Diversity')
    ax[1, 2].set_ylabel('Diversity')