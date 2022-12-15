import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from numpy import reshape
import seaborn as sns
import colorcet as cc
from scipy import interpolate
from scipy.spatial import ConvexHull
import numpy as np
import pandas as pd


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


def apply_tsne_2d(x, y, perplexity=50):
    """
    Apply t-SNE to reduce the dimensionality of the data to 2 dimensions.
    Inputs:
        x: numpy array of shape (n_samples, n_features)
        y: numpy array of shape (n_samples,)
    Outputs:
        df: pandas dataframe with columns "y", "comp-1", "comp-2"
    """
    tsne = TSNE(perplexity=perplexity,
                n_components=2,
                verbose=0,
                random_state=42)
    z = tsne.fit_transform(x)

    df = pd.DataFrame()
    df["y"] = y
    df["comp-1"] = z[:, 0]
    df["comp-2"] = z[:, 1]

    return df


def plot_clusters(df, axis, palette):
    sns.scatterplot(x="comp-1",
                    y="comp-2",
                    hue=df.y.tolist(),
                    ax=axis,
                    palette=palette,
                    alpha=1,
                    data=df).set(title="")

    # Label points
    for ind in df.index:
        axis.text(x=df['comp-1'][ind],
                  y=df['comp-2'][ind],
                  s=df['y'][ind],
                  color='black',
                  fontsize=6,
                  horizontalalignment='center',
                  verticalalignment='center')

    # Create hulls
    for i in df.y.unique():
        points = df[df.y == i][['comp-1', 'comp-2']].values
        if len(points) >= 3:
            # get convex hull
            hull = ConvexHull(points)
            # get x and y coordinates
            # repeat last point to close the polygon
            x_hull = np.append(points[hull.vertices, 0], points[hull.vertices,
                                                                0][0])
            y_hull = np.append(points[hull.vertices, 1], points[hull.vertices,
                                                                1][0])
            # interpolate
            dist = np.sqrt((x_hull[:-1] - x_hull[1:])**2 +
                           (y_hull[:-1] - y_hull[1:])**2)
            dist_along = np.concatenate(([0], dist.cumsum()))
            spline, u = interpolate.splprep([x_hull, y_hull],
                                            u=dist_along,
                                            s=0,
                                            per=1)
            interp_d = np.linspace(dist_along[0], dist_along[-1], 50)
            interp_x, interp_y = interpolate.splev(interp_d, spline)
            # plot shape
            axis.fill(interp_x, interp_y, '--', c=palette[i], alpha=0.2)


def plot_tsne(df, perplexity n_clusters):
    """
    Plots tsne with convex hulls.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with columns 'comp-1', 'comp-2' and 'y'
    perplexity : int
        Perplexity for tsne
    """

    # plot tsne
    fig, axs = plt.subplots(1, 1, figsize=(15, 5))

    palette = sns.color_palette(cc.glasbey, n_colors=n_clusters)

    plot_clusters(df, axs, palette)

    plt.title(f'TSNE with perplexity={perplexity}')
    plt.suptitle(f'Opaque points are items. Others are users.')
    plt.show()


def plot_tsne_comparison(df1, df2, n_clusters):
    """
    Plots two tsne plots (before and after simulation) with convex hulls.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with columns 'comp-1', 'comp-2' and 'y'
    perplexity : int
        Perplexity for tsne
    """
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))

    palette = sns.color_palette(cc.glasbey, n_colors=n_clusters)

    plot_clusters(df1, axs[0], palette)
    axs[0].set_title('Before Simulation')
    plot_clusters(df2, axs[1], palette)
    axs[1].set_title('After Simulation')

    plt.suptitle(f'TSNE')
    plt.show()