import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy

from keras.datasets import mnist
# from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def circles_example():
    """
    an example function for generating and plotting synthetic data.
    """

    t = np.arange(0, 2 * np.pi, 0.03)
    length = np.shape(t)
    length = length[0]
    circle1 = np.matrix([np.cos(t) + 0.1 * np.random.randn(length),
                         np.sin(t) + 0.1 * np.random.randn(length)])
    circle2 = np.matrix([2 * np.cos(t) + 0.1 * np.random.randn(length),
                         2 * np.sin(t) + 0.1 * np.random.randn(length)])
    circle3 = np.matrix([3 * np.cos(t) + 0.1 * np.random.randn(length),
                         3 * np.sin(t) + 0.1 * np.random.randn(length)])
    circle4 = np.matrix([4 * np.cos(t) + 0.1 * np.random.randn(length),
                         4 * np.sin(t) + 0.1 * np.random.randn(length)])
    circles = np.hstack((circle1, circle2, circle3, circle4))
    circles = np.array(circles.T)

    return circles


def apml_pic_example(path='APML_pic.pickle'):
    """
    an example function for loading and plotting the APML_pic data.

    :param path: the path to the APML_pic.pickle file
    """

    with open(path, 'rb') as f:
        apml = pickle.load(f)

    return apml


def four_gaussians_example():
    """
    an example function for generating and plotting synthetic data.
    """
    N = 840
    params = {0: {'mean': [1, -1],
                  'cov': [[1, 0], [0, 1]]},
              1: {'mean': [5, -5],
                  'cov': [[1, 0], [0, 1]]},
              2: {'mean': [10, -10],
                  'cov': [[2, 0], [0, 2]]},
              3: {'mean': [10, 10],
                  'cov': [[2, 0], [0, 2]]}
              }

    mixture_idx = np.random.choice(a=4, size=N, replace=True)

    x_cord, y_cord = [], []
    for i in mixture_idx:
        x, y = np.random.multivariate_normal(params[i]['mean'], params[i]['cov'], 1).T
        x_cord.append(x[0])
        y_cord.append(y[0])

    data = np.array((x_cord, y_cord)).T

    return data


def microarray_exploration(data_path='microarray_data.pickle',
                            genes_path='microarray_genes.pickle',
                            conds_path='microarray_conds.pickle'):
    """
    an example function for loading and plotting the microarray data.
    Specific explanations of the data set are written in comments inside the
    function.

    :param data_path: path to the data file.
    :param genes_path: path to the genes file.
    :param conds_path: path to the conds file.
    """

    # loading the data:
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    with open(genes_path, 'rb') as f:
        genes = pickle.load(f)
    with open(conds_path, 'rb') as f:
        conds = pickle.load(f)

    # look at a single value of the data matrix:
    i = 128
    j = 63
    print("gene: " + str(genes[i]) + ", cond: " + str(
        conds[0, j]) + ", value: " + str(data[i, j]))

    # make a single bar plot:
    plt.figure()
    plt.bar(np.arange(len(data[i, :])) + 1, data[i, :])
    plt.show()

    # look at two conditions and their correlation:
    plt.figure()
    plt.scatter(data[:, 27], data[:, 29])
    plt.plot([-5,5],[-5,5],'r')
    plt.show()

    # see correlations between conds:
    correlation = np.corrcoef(np.transpose(data))
    plt.figure()
    plt.imshow(correlation, extent=[0, 1, 0, 1])
    plt.colorbar()
    plt.show()

    # look at the entire data set:
    plt.figure()
    plt.imshow(data, extent=[0, 1, 0, 1], cmap="hot", vmin=-3, vmax=3)
    plt.colorbar()
    plt.show()


def euclid(X, Y):
    """
    return the pair-wise euclidean distance between two data matrices.
    :param X: NxD matrix.
    :param Y: MxD matrix.
    :return: NxM euclidean distance matrix.
    """

    dist = np.zeros((X.shape[0], Y.shape[0]))
    for i in range(0, X.shape[0]):
        for j in range(0, Y.shape[0]):
            dist[i,j] = np.sqrt(np.dot(X[i], X[i].T) - 2 * np.dot(X[i], Y[j].T) + np.dot(Y[j], Y[j].T))
    return dist


def euclidean_centroid(X):
    """
    return the center of mass of data points of X.
    :param X: a sub-matrix of the NxD data matrix that defines a cluster.
    :return: the centroid of the cluster.
    """
    c = X.sum(axis=0)/X.shape[0]
    return c


def kmeans_pp_init(X, k, metric):
    """
    The initialization function of kmeans++, returning k centroids.
    :param X: The data matrix.
    :param k: The number of clusters.
    :param metric: a metric function like specified in the kmeans documentation.
    :return: kxD matrix with rows containing the centroids.
    """

    # step 1: pic random data point as first center
    idx = np.random.choice(range(0, X.shape[0]), 1)
    center = X[idx]

    # step 2: choose new center with probability ~ D(x)**2
    for _ in range(1,k):
        t = metric(center, X) # distance between data points and center that already were chosen
        t = np.nan_to_num(t)
        t = t.min(axis=0)          # we are only interested in the distance to the closest center
        t_sq = np.multiply(t, t)   #
        w = t_sq / t_sq.sum()      # probability (weight) for the choice of the data point as new center
        idx = np.random.choice(X.shape[0], 1, p=w)  # randomly choose index of new center (with weights)
        center_i = X[idx]
        center = np.vstack((center, center_i))
    return center


def kmeans(X, k, iterations=10, metric=euclid, center=euclidean_centroid, init=kmeans_pp_init):
    """
    The K-Means function, clustering the data X into k clusters.
    :param X: A NxD data matrix.
    :param k: The number of desired clusters.
    :param iterations: The number of iterations.
    :param metric: A function that accepts two data matrices and returns their
            pair-wise distance. For a NxD and KxD matrices for instance, return
            a NxK distance matrix.
    :param center: A function that accepts a sub-matrix of X where the rows are
            points in a cluster, and returns the cluster centroid.
    :param init: A function that accepts a data matrix and k, and returns k initial centroids.
    :param stat: A function for calculating the statistics we want to extract about
                the result (for K selection, for example).
    :return: a tuple of (clustering, centroids, statistics)
    clustering - A N-dimensional vector with indices from 0 to k-1, defining the clusters.
    centroids - The kxD centroid matrix.
    """

    # step 0: use kmeans++ to initialize k centers
    centers = init(X, k, metric)
    print('init over - now kmeans')

    for _ in range(0, iterations):
        # step 1: measure distance of points to centers - assign points to the closest center
        points2cluster = metric(X, centers).argmin(axis=1)

        # step 2: calculate new centers
        for i in range(0,k):
            idx = (points2cluster == i)
            c_i = center(X[idx])
            centers[i] = c_i

    points2cluster = metric(X, centers).argmin(axis=1)  # finial assignment of points

    return points2cluster, centers



def gaussian_kernel(X, sigma):
    """
    calculate the gaussian kernel similarity of the given distance matrix.
    :param X: A NxN distance matrix.
    :param sigma: The width of the Gaussian in the heat kernel.
    :return: NxN similarity matrix.
    """

    W = np.exp(-(X**2)/(2*(sigma**2)))
    return W


def mnn(X, m):
    """
    calculate the mutual nearest neighbors similarity of the given distance matrix.
    :param X: A NxN distance matrix.
    :param m: The number of nearest neighbors. in every direction (point A is neighbor of point B) .
              is reduced to n<m so that points are nearest neighbors in both directions
              (only S_ij = 1 if A is nearest neighbor of B AND B is nearest neighbor of A)
    :return: NxN similarity matrix.
    """

    dist_idx_1 = np.argsort(X, axis=1)
    nearest_idx_1 = dist_idx_1[:, :(m+1)]  # column wise m nearest neighbors

    dist_idx_0 = np.argsort(X, axis=0)
    nearest_idx_0 = dist_idx_0[:(m+1), :]  # row wise m nearest neighbors

    NN = np.zeros(X.shape)
    for i in range(0, X.shape[0]):
        NN[i, nearest_idx_1[i]] += 1
    for j in range(0, X.shape[0]):
        NN[nearest_idx_0[:,j], j] += 1

    NN = NN - 2*np.diag(np.ones(X.shape[0]))  # substract diagonal
    S = (NN==2).astype(int)  # only keep entries where both points are neighbors of each other

    return S


def spectral(X, k, similarity_param, similarity=gaussian_kernel):
    """
    Cluster the data into k clusters using the spectral clustering algorithm.
    :param X: A NxD data matrix.
    :param k: The number of desired clusters.
    :param similarity_param: m for mnn, sigma for the Gaussian kernel.
    :param similarity: The similarity transformation of the data.
    :return: a tuple of (clustering, centroids, eigenvectors, eigenvalues)
    clustering - A N-dimensional vector with indices from 0 to k-1, defining the clusters.
    centroids - The kxD centroid matrix.
    eigenvectors - only k smallest ones, row wise normalized so that every data point has norm=1
    eigencalues - 30 smallest ones
    """

    Distance = euclid(X, X)
    Adjacency = similarity(Distance, similarity_param)
    diag = Adjacency.sum(axis=0)  # degree
    diag[(diag==0)] = 1  # for a point that was no mutual nearest neighbor assign 1 because would break with 0
                         # and this cheat still works well enough

    Deg_inv = np.diag(1/np.sqrt(diag))
    L_sym = np.diag(np.ones(Adjacency.shape[0])) - np.matmul(Deg_inv.T, np.matmul(Adjacency, Deg_inv))
    v, U = scipy.linalg.eigh(L_sym)

    T = U[:, :k]  # use k smallest eigenvectors for kmeans clustering
    row_sums = np.linalg.norm(T, axis=1)  # normalize rows
    t = T / row_sums[:, np.newaxis]

    # k-means clustering
    points2cluster, center = kmeans(t, k)

    return points2cluster, center, t, v


def plot_similarity(X, similarity, similarity_param, points2cluster):
    """
    plots the similarity matrix for data in random order and sorted according to clusters
    :param X: A NxD data matrix.
    :param similarity: The similarity transformation of the data.
    :param similarity_param: m for mnn, sigma for the Gaussian kernel.
    :param points2cluster: A N-dimensional vector with indices from 0 to k-1, defining the clusters.
    :return: plot of similarity matrix of unsorted and sorted data
    """

    # bring data in random order
    idx = np.random.choice(range(0,X.shape[0]), X.shape[0])
    X_rand = X[idx]

    # compute similarity
    Distance = euclid(X_rand, X_rand)
    Adjacency_mix = similarity(Distance, similarity_param)

    # use points2cluster assignment to sort data according to clusters
    p2c = np.matrix((np.arange(points2cluster.shape[0]), points2cluster))
    p2c_sort = p2c.T.tolist()

    p2c_sort.sort(key=lambda x: x[1])

    p2c_sort = np.matrix(p2c_sort)
    p2c_idx = p2c_sort[:,0].flatten().tolist()[0]

    X_sort = X[p2c_idx]  # data is sorted now

    # compute similarity again
    Distance = euclid(X_sort, X_sort)
    Adjacency_sort = similarity(Distance, similarity_param)

    # plot comparison
    fig1, (ax0, ax1) = plt.subplots(ncols=2, figsize=(8, 5),
                             subplot_kw={'xticks': [], 'yticks': []})

    ax0.set_title('Data in random order')
    ax0.imshow(Adjacency_mix)

    ax1.set_title('Data sorted according to clusters')
    ax1.imshow(Adjacency_sort)

    return fig1


def silhouette(X, points2cluster):
    """
    Uses the silhouette method to determine quality of the clustering
    :param X: A NxD data matrix.
    :param points2cluster: A N-dimensional vector with indices from 0 to k-1, defining the clusters.
    :return: silhouette score value between -1 and 1, where -1 is bad performance, 1 perfect performance
    """

    k = points2cluster.max()  # amount of clusters

    assigned = np.vstack((X.T, points2cluster.T))
    a_list, b_list = [], []
    clusters = np.array(range(0,k+1))

    for i in clusters:
        # calculate mean distance of data point to the other points in his cluster
        mask = (assigned[2, :] == i).tolist()
        a_i = assigned[:2, mask].T

        a = euclid(a_i, a_i).mean(axis=1).sum()
        a_list = a_list + [a]

        b_j_list = []
        clusters_without_i = np.delete(clusters, i)

        # b = 0

        for j in clusters_without_i:
            # calculate mean distance of datapoint to the closest data point of an other cluster
            mask_is_j = (assigned[2, :] == j).tolist()
            b_ij = assigned[:2, mask_is_j].T

            b_j = euclid(a_i, b_ij).mean(axis=1).sum()

            b_j_list = b_j_list + [b_j]
            b = np.min(np.array(b_j_list))

        b_list = b_list + [b]

    # calculate score
    S = 1/(k+1) * ((np.array(b_list) - np.array(a_list))/np.array([a_list, b_list]).max(axis=0)).sum()

    return S


def choose_k_spectral(X, values, similarity_param, similarity):  # TODO: vielleicht brauche ich das gar nicht, weil ich es mit four_gaussians mache
    """
    Uses visual method and silhouette method to choose k
    :param X:
    :param values:
    :param similarity_param:
    :param similarity:
    :return:
    """
    runs = {}
    for k in values:

        print(str(k) + '  clusters')
        points2cluster, centers, t, v = spectral(X=X, k=k,
                                                 similarity_param=similarity_param,
                                                 similarity=similarity)

        dist = euclid(X, X)
        S = silhouette(X, points2cluster)

        runs.update({k: {'p2c': points2cluster,
                         'silhouette': S}
                     })

    # plot points2cluster for different k
    fig, axes = plt.subplots(3, 3, figsize=(12, 12),
                             subplot_kw={'xticks': [], 'yticks': []})

    fig.subplots_adjust(hspace=0.3, wspace=0.05)

    for ax, k in zip(axes.flat, sorted(runs)):
        points2cluster = runs[k]['p2c']
        ax.scatter(X[:, 0], X[:, 1], c=points2cluster, cmap=plt.get_cmap('Paired'))
        ax.set_title(str(k) + ' clusters')

    fig2, (ax0) = plt.subplots(1, 1, figsize=(8,5))

    silh = [runs[k]['silhouette'] for k in runs]
    ax0.plot(values, silh)
    ax0.set_title('Silhouette')

    return fig, fig2


def choose_m_mnn(X, values, k):
    """
    Uses visual method to find m
    :param X: data
    :param values: range of values for m to try
    :param k: number of desired clusters
    :return: scatterplot of data for different m values
    """
    runs = {}
    for m in values:

        print(str(m) + '  neighbors')
        points2cluster, centers, t, v = spectral(X=X, k=k,
                                                 similarity_param=m,
                                                 similarity=mnn)


        runs.update({m: {'p2c': points2cluster}})

    # plot points2cluster for different m
    fig, axes = plt.subplots(3, 3, figsize=(12, 12),
                             subplot_kw={'xticks': [], 'yticks': []})

    fig.subplots_adjust(hspace=0.3, wspace=0.05)

    for ax, m in zip(axes.flat, sorted(runs)):
        points2cluster = runs[m]['p2c']
        ax.scatter(X[:, 0], X[:, 1], c=points2cluster, cmap=plt.get_cmap('Paired'))
        ax.set_title(str(m) + ' neighbors')

    return fig


def choose_k_kmean(X, range):
    """
    performs k-means++ clustering on data for different clusters k and uses
    elbow and silhouette method to determine amount of clusters k
    :param X: data
    :param range: range of k clusters
    :return: plots for camparison of different k (fig, fig1)
    fig1: visualisation of data with k labels according to clustering
    fig2: loss (elbow method) and silhouette for different k
    """
    runs = {}
    for k in range:
        # k-means++ clustering
        points2cluster, centers = kmeans(X=X, k=k)

        dist = euclid(X, centers)
        dist = dist.min(axis=1)
        loss = dist.sum()

        S = silhouette(X, points2cluster)

        runs.update({k: {'p2c': points2cluster,
                         'loss': loss,
                         'silhouette': S}
                     })

    # plot points2cluster for different k
    fig1, axes = plt.subplots(3, 3, figsize=(12, 12),
                             subplot_kw={'xticks': [], 'yticks': []})

    fig1.subplots_adjust(hspace=0.3, wspace=0.05)

    for ax, k in zip(axes.flat, sorted(runs)):
        points2cluster = runs[k]['p2c']
        ax.scatter(X[:, 0], X[:, 1], c=points2cluster, cmap=plt.get_cmap('Paired'))
        ax.set_title(str(k) + ' clusters')

    # plot silhouette and loss (elbow method)
    fig2, (ax0, ax1) = plt.subplots(1, 2, figsize=(8,5))

    silh = [runs[k]['silhouette'] for k in runs]
    ax0.plot(range, silh)
    ax0.set_title('Silhouette')

    loss = [runs[k]['loss'] for k in runs]
    ax1.plot(range, loss)
    ax1.set_title('Loss')

    return fig1, fig2


def load_data():
    """
    load and store synthetic datasets and microarray
    also assign their k and similarity parameters
    :return: data in dictionary
    """
    circles = circles_example()

    apml = apml_pic_example()

    idx = np.random.choice(apml.shape[0], 840)
    apml_small = apml[idx]

    data_path = 'microarray_data.pickle'
    with open(data_path, 'rb') as f:
        microarray = pickle.load(f)


    idx = np.random.choice(microarray.shape[0], 1040)
    microarray_small = microarray[idx]

    four_gaussians = four_gaussians_example()

    data = {'circles': {'similarity_param': {'gaussian': 0.17,
                                       'mnn': 10},
                        'k': 4,
                        'data': circles},
            'apml': {'similarity_param': {'gaussian': 5.,
                                    'mnn': 27},
                     'k': 9,
                     'data': apml},
            'apml_small': {'similarity_param': {'gaussian': 5.,
                                    'mnn': 15},
                           'k': 9,
                           'data': apml_small},
            'microarray': {'similarity_param': {'gaussian': 7.,
                                    'mnn': 106},
                           'k': 10,
                           'data': microarray},
            'microarray_small': {'similarity_param': {'gaussian': 7.,
                                                'mnn': 80},
                           'k': 10,
                           'data': microarray_small},
            'four_gaussians': {'similarity_param': {'gaussian': 0.35,
                                                'mnn': 24},
                           'k': 4,
                           'data': four_gaussians}}
    return data


def get_sigma(X, nn):
    """
    estimate sigma (variance for gaussian kernel on distance matrix)
    :param X: data points
    :param nn: how many nearest neighbors
    :return: sigma: mean of nn nearest neighbors
    """

    dist = euclid(X, X) # calculate distance between points
    dist.sort()  # sort ascending
    dist = dist[:, 1:]  # drop zeros of diagonal
    sigma = dist[:, :nn].mean(axis=1).mean()  # mean of nn nearest neighbors
    # dist = dist.flatten()
    # plt.hist(dist, bins=300)  # plot distance histogram
    # plt.show
    return sigma


def tSNE_vs_PCA(data, tags, n_plot):
    """
    compare t-SNE to PCA. Which method captures the structure of the data better?
    :param data: dataset to analyse
    :param tags: known tags to this dataset
    :param n_plot: number of data points to plot
    :return: fig1: plot comparison of t-SNE to PCA
    """

    # PCA
    mu = data.mean(axis=0)
    U, s, V = np.linalg.svd(data - mu, full_matrices=False)
    pca_results = np.dot(data - mu, V.transpose())

    # t-SNE
    tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(data)

    # plot comparison
    fig1, (ax0, ax1) = plt.subplots(ncols=2, figsize=(8, 5))
    ax0.set_title('PCA')
    ax0.scatter(pca_results[:n_plot, 0], pca_results[:n_plot, 1], c=tags[:n_plot], s=5, cmap='tab10')
    ax0.set_xticks([])
    ax0.set_yticks([])

    ax1.set_title('t-SNE')
    im = ax1.scatter(tsne_results[:n_plot, 0], tsne_results[:n_plot, 1], c=tags[:n_plot], s=5, cmap='tab10')
    ax1.set_xticks([])
    ax1.set_yticks([])

    axins = inset_axes(ax1,         # for colorbar
                       width="5%",
                       height="100%",
                       loc='lower left',
                       bbox_to_anchor=(1.05, 0., 1, 1),
                       bbox_transform=ax1.transAxes,
                       borderpad=0,
                       )
    fig1.colorbar(im, cax=axins)
    fig1.tight_layout()
    return fig1


def choose_m_similarity(X, values):
    """
    tries different values m for mutual nearest neighbor similarity and plots similarity matrix and eigenvalues
    :param X: data set
    :param values: range for m mutual nearest neighbors and their best amount of cluster k
    :return: plots to compare different m
    fig: plot similarity matrix
    fig1: plot eigenvalues
    """
    runs = {}
    for val in values:
        # spectral clustering for (m,k)
        m = val.tolist()[0][0]
        k = val.tolist()[0][1]

        print(str(m) + '  neighbors,  ' + str(k) + '  clusters')
        points2cluster, centers, t, v = spectral(X=X, k=k,
                                                 similarity_param=m,
                                                 similarity=mnn)

        runs.update({m: {'p2c': points2cluster,
                         'k': v[:20]}})

    # plot similarities for different m
    fig, axes = plt.subplots(3, 3, figsize=(12, 12),
                             subplot_kw={'xticks': [], 'yticks': []})

    fig.subplots_adjust(hspace=0.3, wspace=0.05)

    for ax, m in zip(axes.flat, sorted(runs)):
        # order datapoints according to their points2cluster assignment from clustering
        points2cluster = runs[m]['p2c']
        p2c = np.matrix((np.arange(points2cluster.shape[0]), points2cluster))
        p2c_sort = p2c.T.tolist()

        p2c_sort.sort(key=lambda x: x[1])

        p2c_sort = np.matrix(p2c_sort)
        p2c_idx = p2c_sort[:, 0].flatten().tolist()[0]

        X_sort = X[p2c_idx]

        dist_sort = euclid(X_sort, X_sort)
        sim_sort = mnn(dist_sort, m)
        ax.imshow(sim_sort, extent=[0, 1, 0, 1], vmin=0, vmax=1)
        ax.set_title(str(m) + ' neighbors')

    # plot eigenvalues
    fig1, axes = plt.subplots(3, 3, figsize=(12, 12),
                                 subplot_kw={'yticks': []})


    fig1.subplots_adjust(hspace=0.3, wspace=0.05)

    for ax, m in zip(axes.flat, sorted(runs)):
        ax.bar(range(0,20), runs[m]['k'])
        ax.set_title(str(m) + ' neighbors')

    return fig, fig1


def choose_sigma_similarity(X, values, k):
    """
    like choose_m_similarity but for sigma and gaussian_kernel instead of m and mnn
    """
    runs = {}
    for val in values:
        sigma = val.tolist()[0][0]
        k = int(val.tolist()[0][1])

        print(str(sigma) + '  sigma,  ' + str(k) + '  clusters')
        points2cluster, centers, t, v = spectral(X=X, k=k,
                                                 similarity_param=sigma,
                                                 similarity=gaussian_kernel)


        runs.update({sigma: {'p2c': points2cluster,
                         'k': v[:20]}})

    # plot points2cluster for different m
    fig, axes = plt.subplots(3, 3, figsize=(12, 12),
                             subplot_kw={'xticks': [], 'yticks': []})

    fig.subplots_adjust(hspace=0.3, wspace=0.05)

    for ax, m in zip(axes.flat, sorted(runs)):
        print('plot ' + str(m))
        points2cluster = runs[m]['p2c']
        p2c = np.matrix((np.arange(points2cluster.shape[0]), points2cluster))
        p2c_sort = p2c.T.tolist()

        p2c_sort.sort(key=lambda x: x[1])

        p2c_sort = np.matrix(p2c_sort)
        p2c_idx = p2c_sort[:, 0].flatten().tolist()[0]

        X_sort = X[p2c_idx]

        dist_sort = euclid(X_sort, X_sort)
        sim_sort = gaussian_kernel(dist_sort, m)
        ax.imshow(sim_sort, extent=[0, 1, 0, 1], vmin=0, vmax=1)
        ax.set_title('sigma = ' + str(m))

    fig1, axes = plt.subplots(3, 3, figsize=(12, 12),
                                 subplot_kw={'yticks': []})


    fig1.subplots_adjust(hspace=0.3, wspace=0.05)

    for ax, m in zip(axes.flat, sorted(runs)):
        ax.bar(range(0,20), runs[m]['k'])
        ax.set_title('sigma = ' + str(m))

    return fig, fig1



if __name__ == '__main__':

    data = load_data()

    ####################################################################################
    #                 K-means++ Clustering on Synthetic Data                           #
    ####################################################################################

    vals = {}

    for name in ['apml', 'circles', 'four_gaussians']:

        print(name)
        dataset = name
        X = data[dataset]['data']
        k = data[dataset]['k']
        sigma = data[dataset]['similarity_param']['gaussian']
        m = data[dataset]['similarity_param']['mnn']

        p2c_kmeans, centers = kmeans(X, k)
        p2c_spec_gaussian, centers, t, v = spectral(X, k, similarity_param=sigma, similarity=gaussian_kernel)
        p2c_spec_mnn, centers, t, v = spectral(X, k, similarity_param=m, similarity=mnn)

        vals.update({'kmeans_' + name: {'X': X,
                                        'p2c': p2c_kmeans},
                    'gaussian_' + name: {'X': X,
                                         'p2c': p2c_spec_gaussian},
                     'mnn_' + name: {'X':X,
                                     'p2c': p2c_spec_mnn}})

        fig, axes = plt.subplots(3, 3, figsize=(12, 12),
                                 subplot_kw={'xticks': [], 'yticks': []})

        fig.subplots_adjust(hspace=0.3, wspace=0.05)

        for ax, k in zip(axes.flat, sorted(vals)):
            print(k)
            X = vals[k]['X']
            points2cluster = vals[k]['p2c']
            ax.scatter(X[:, 0], X[:, 1], c=points2cluster, cmap=plt.get_cmap('Paired'))
            ax.set_title(str(k))

    fig.savefig('synthetic_data_clustering.png')

    ####################################################################################
    #                 choose m  and  choose k silhouette                               #
    ####################################################################################
    for name in ['apml_small', 'circles', 'four_gaussians']:
        print(name)
        dataset = name
        X = data[dataset]['data']
        k = data[dataset]['k']
        similarity_param = data[dataset]['similarity_param']['mnn']

        similarity = mnn

        fig1 = choose_m_mnn(X=X, values=[6, 9, 12, 15, 18, 27, 40, 55, 70], k=k)

    ####################################################################################
    #                           Similarity Plot                                        #
    ####################################################################################

    dataset = 'apml_small'
    X = data[dataset]['data']
    k = data[dataset]['k']
    similarity_param = data[dataset]['similarity_param']['mnn']
    similarity = mnn

    points2cluster, centers, t, v = spectral(X, k, similarity_param=similarity_param, similarity=similarity)
    fig4 = plot_similarity(X=X, similarity=similarity, similarity_param=similarity_param, points2cluster=points2cluster)
    fig4.savefig('similarity.png')

    ####################################################################################
    #                        Choose k k-mean elbow                                     #
    ####################################################################################

    dataset = 'four_gaussians'
    X = data[dataset]['data']
    k = data[dataset]['k']

    points2cluster, centers = kmeans(X=X, k=k)
    plt.scatter(X[:,0], X[:,1], c=points2cluster)

    fig5, fig6 = choose_k_kmean(X, range(2,11))

    fig5.savefig('4gaussians_choose_k.png')
    fig6.savefig('4gaussians_choose_k_loss.png')

    ####################################################################################
    #                          find sigma for microarray                               #
    ####################################################################################

    # calculate mean percentile of sigmas for synthetic data - because here we know that it worked
    for name in ['apml', 'circles', 'four_gaussians']:
        print(name)
        dataset = name
        X = data[dataset]['data']
        k = data[dataset]['k']
        sigma = data[dataset]['similarity_param']['gaussian']

        dist = euclid(X, X)
        dist.sort()
        dist = dist[:, 1:]
        dist = dist.flatten()
        dist.sort()
        smaller_sigma = (dist <= sigma)
        percentile = smaller_sigma.sum()/smaller_sigma.shape[0]
        print(percentile)

    mean_percentile = 1/3* (0.002639105022069362 + 0.0039502809467052615 + 0.005749474998581077)
    print(mean_percentile)

    # now find the equivalent sigma from microarray
    # therefore use smaller subset to speed things up, randomly drawn to protect distribution
    X = data['microarray_small']['data']

    dist = euclid(X, X)

    dist.sort()
    dist = dist[:, 1:]
    dist = dist.flatten()
    dist.sort()
    dist = dist[307470:] # deleted more zeros

    idx = dist.shape[0]*mean_percentile
    sigma = dist[int(idx)]  # 4.77798

    smaller_sigma = (dist <= sigma)
    percentile = smaller_sigma.sum()/smaller_sigma.shape[0]
    print(sigma)

    print(percentile, mean_percentile)


    ####################################################################################
    #                                 microarray                                       #
    ####################################################################################

    dataset = 'microarray_small'
    X = data[dataset]['data']
    k = data[dataset]['k']

    ####################################################################################
    #                                  k-means++                                       #
    ####################################################################################

    vals = {}
    for k in range(5, 17):
        points2cluster, centers = kmeans(X, k)
        vals.update({k: points2cluster})


    # plot points2cluster for different k
    fig8, axes = plt.subplots(4, 3, figsize=(12, 16),
                             subplot_kw={'xticks': [], 'yticks': []})

    fig8.subplots_adjust(hspace=0.3, wspace=0.05)

    for ax, k in zip(axes.flat, sorted(vals)):
        points2cluster = vals[k]
        p2c = np.matrix((np.arange(points2cluster.shape[0]), points2cluster))
        p2c_sort = p2c.T.tolist()

        p2c_sort.sort(key=lambda x: x[1])

        p2c_sort = np.matrix(p2c_sort)
        p2c_idx = p2c_sort[:, 0].flatten().tolist()[0]

        X_sort = X[p2c_idx]

        ax.imshow(X_sort, extent=[0, 1, 0, 1], cmap='hot', vmin=-3, vmax=3)
        ax.set_title(str(k) + ' clusters')

    ####################################################################################
    #                        spectral gaussian_kernel                                  #
    ####################################################################################


    fig9, fig10 = choose_sigma_similarity(X,
                                 np.matrix([[4., 5., 6., 7., 8., 9., 10., 11., 12.],
                                              [9, 9, 10, 10, 9, 8, 8, 8, 8]]).T,
                                 k)

    ####################################################################################
    #                        spectral mnn                                              #
    ####################################################################################

    fig11, fig12 = choose_m_similarity(X,
                                 np.matrix([[91, 94, 97, 100, 103, 106, 109, 112, 115],
                                              [9, 9, 10, 10, 9, 8, 8, 8, 8]]).T)

    ####################################################################################
    #                               t-SNE vs. PCA                                      #
    ####################################################################################


    # digits, tags = load_digits(return_X_y=True)   # small
    # data = digits / 16

    (data, tags), (x_test, y_test) = mnist.load_data()   #big
    data = data.reshape(60000, 784) / 255

    fig7 = tSNE_vs_PCA(data, tags, 3000)
    fig7.savefig('tSNE_vs_PCA.png')


    plt.show()

