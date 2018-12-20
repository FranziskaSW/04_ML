import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import silhouette_score  # TODO: delete
import scipy

from keras.datasets import mnist
# from sklearn.datasets import load_digits
import time
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

    points2cluster = np.array([0]*length + [1]*length + [2]*length + [3]*length)

    return circles


def apml_pic_example(path='APML_pic.pickle'):
    """
    an example function for loading and plotting the APML_pic data.

    :param path: the path to the APML_pic.pickle file
    """

    with open(path, 'rb') as f:
        apml = pickle.load(f)

    # plt.plot(apml[:, 0], apml[:, 1], '.')
    # plt.show()
    return apml


def four_gaussians_example():
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

    # TODO: is it already euclidean centroid or do I need to square it somewhere?
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

    # TODO: YOUR CODE HERE
    # step 0: use kmeans++ to initialize k centers
    centers = init(X, k, metric)
    print('init over - now kmeans')

    for _ in range(0, iterations):
        # step 1: measure distance of points to centers - assign points to the closest center
        points2cluster = metric(X, centers).argmin(axis=1)

        # step 2: calculate new centers
        for i in range(0,k):
            idx = (points2cluster == i)
            c_i = center(X[idx])       # TODO: right now it's some float number, should it better be the closest data point?
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

    # TODO: check
    W = np.exp(-(X**2)/(2*(sigma**2)))
    return W


def mnn(X, m):
    """
    calculate the m nearest neighbors similarity of the given distance matrix.
    :param X: A NxN distance matrix.
    :param m: The number of nearest neighbors.
    :return: NxN similarity matrix.
    """

    # TODO: YOUR CODE HERE
    dist_idx_1 = np.argsort(X, axis=1)
    nearest_idx_1 = dist_idx_1[:, :(m+1)]

    dist_idx_0 = np.argsort(X, axis=0)
    nearest_idx_0 = dist_idx_0[:(m+1), :]

    NN = np.zeros(X.shape)
    for i in range(0, X.shape[0]):
        NN[i, nearest_idx_1[i]] += 1
    for j in range(0, X.shape[0]):
        NN[nearest_idx_0[:,j], j] += 1

    NN = NN - 2*np.diag(np.ones(X.shape[0]))
    t = (NN==2).astype(int)

    return t


def spectral(X, k, similarity_param, similarity=gaussian_kernel):
    """
    Cluster the data into k clusters using the spectral clustering algorithm.
    :param X: A NxD data matrix.
    :param k: The number of desired clusters.
    :param similarity_param: m for mnn, sigma for the Gaussian kernel.
    :param similarity: The similarity transformation of the data.
    :return: clustering, as in the kmeans implementation.
    """

    Distance = euclid(X, X)

    Adjacency = similarity(Distance, similarity_param)
    diag = Adjacency.sum(axis=0)
    diag[(diag==0)] = 1   # for a point that was no mutual nearest neighbor assign 1 because would break with 0
    Degree = np.diag(diag)

    Deg_inv = np.diag(1/np.sqrt(diag))
    L_sym = np.diag(np.ones(Degree.shape[0])) - np.matmul(Deg_inv.T, np.matmul(Adjacency, Deg_inv))
    v, U = scipy.linalg.eigh(L_sym)

    T = U[:, :k]
    row_sums = np.linalg.norm(T, axis=1)
    t = T / row_sums[:, np.newaxis]
    print('now kmeans clustering')
    points2cluster, center = kmeans(t, k)

    return points2cluster, center, t, v

'''
similarity = gaussian_kernel
similarity_param = 5 # 0.17 #0.35358367870405927
k = 9

similarity = mnn
similarity_param = 15
k = 9

plt.scatter(X[:,0], X[:,1], c=points2cluster)
plt.bar(range(0,30), v[:30])
'''

def plot_similarity(X, similarity, similarity_param, points2cluster):

    idx = np.random.choice(range(0,X.shape[0]), X.shape[0])
    X_rand = X[idx]

    Distance = euclid(X_rand, X_rand)
    Adjacency_mix = similarity(Distance, similarity_param)

    p2c = np.matrix((np.arange(points2cluster.shape[0]), points2cluster))
    p2c_sort = p2c.T.tolist()

    p2c_sort.sort(key=lambda x: x[1])

    p2c_sort = np.matrix(p2c_sort)
    p2c_idx = p2c_sort[:,0].flatten().tolist()[0]

    X_sort = X[p2c_idx]
    Distance = euclid(X_sort, X_sort)
    Adjacency_sort = similarity(Distance, similarity_param)

    fig1, (ax0, ax1) = plt.subplots(ncols=2, figsize=(8, 5),
                             subplot_kw={'xticks': [], 'yticks': []})

    ax0.set_title('Data in random order')
    ax0.imshow(Adjacency_mix)

    ax1.set_title('Data sorted according to clusters')
    ax1.imshow(Adjacency_sort)

    return fig1

def translate(p2c):  # TODO: do I even need this?

    p2c_new = []
    translate = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    translate_idx = -1
    words = {}

    for i in range(0, p2c.shape[0]):
        val = p2c[i]

        try:
            p2c_new = p2c_new + [words[val]]
        except KeyError:
            translate_idx += 1
            p2c_new = p2c_new + [translate[translate_idx]]
            words.update({val : translate[translate_idx]})

    return np.array(p2c_new)


def silhouette(X, points2cluster):

    k = points2cluster.max()

    assigned = np.vstack((X.T, points2cluster.T))
    a_list, b_list = [], []
    clusters = np.array(range(0,k+1))

    for i in clusters:
        mask = (assigned[2, :] == i).tolist()
        a_i = assigned[:2, mask].T

        a = euclid(a_i, a_i).mean(axis=1).sum()
        a_list = a_list + [a]

        b_j_list = []
        clusters_without_i = np.delete(clusters, i)

        # b = 0

        for j in clusters_without_i:
            mask_is_j = (assigned[2, :] == j).tolist()
            b_ij = assigned[:2, mask_is_j].T

            b_j = euclid(a_i, b_ij).mean(axis=1).sum()

            b_j_list = b_j_list + [b_j]
            b = np.min(np.array(b_j_list))

        b_list = b_list + [b]

    S = 1/(k+1) * ((np.array(b_list) - np.array(a_list))/np.array([a_list, b_list]).max(axis=0)).sum()

    return S


def choose_k_spectral(X, values, similarity_param, similarity):
    runs = {}
    for k in values:

        print(str(k) + '  clusters')
        points2cluster, centers, t, v = spectral(X=X, k=k,
                                                 similarity_param=similarity_param,
                                                 similarity=similarity)


        S = 0 # silhouette(X, points2cluster)
        S2 = silhouette_score(X, points2cluster)  # TODO: delete

        runs.update({k: {'p2c': points2cluster,
                         'silhouette': S,
                         'S2': S2}
                     })

    # plot points2cluster for different k
    fig, axes = plt.subplots(3, 3, figsize=(12, 12),
                             subplot_kw={'xticks': [], 'yticks': []})

    fig.subplots_adjust(hspace=0.3, wspace=0.05)

    for ax, k in zip(axes.flat, sorted(runs)):
        points2cluster = runs[k]['p2c']
        ax.scatter(X[:, 0], X[:, 1], c=points2cluster, cmap=plt.get_cmap('Set1'))
        ax.set_title(str(k) + ' clusters')

    fig2, (ax0) = plt.subplots(1, 1, figsize=(8,5))

    # silh = [runs[k]['silhouette'] for k in runs]
    silh2 = [runs[k]['S2'] for k in runs]
    # ax0.plot(values, silh)
    ax0.plot(values, silh2)
    ax0.set_title('Silhouette')

    return fig, fig2


def choose_m_mnn(X, values, k):
    runs = {}
    for m in values:

        print(str(m) + '  neighbors')
        points2cluster, centers, t, v = spectral(X=X, k=k,
                                                 similarity_param=m,
                                                 similarity=mnn)


        runs.update({m: {'p2c': points2cluster}
                         #'silhouette': S,
                         #'S2': S2}
                     })

    # plot points2cluster for different m
    fig, axes = plt.subplots(3, 3, figsize=(12, 12),
                             subplot_kw={'xticks': [], 'yticks': []})

    fig.subplots_adjust(hspace=0.3, wspace=0.05)

    for ax, m in zip(axes.flat, sorted(runs)):
        points2cluster = runs[m]['p2c']
        ax.scatter(X[:, 0], X[:, 1], c=points2cluster, cmap=plt.get_cmap('Set1'))
        ax.set_title(str(m) + ' neighbors')

    return fig


def choose_k_kmean(X, range):
    runs = {}
    for k in range:

        print(str(k) + '  clusters')
        points2cluster, centers = kmeans(X=X, k=k)

        dist = euclid(X, centers)
        dist = dist.min(axis=1)
        loss = dist.sum()

        S = 0 # silhouette(X, points2cluster)
        S2 = silhouette_score(X, points2cluster)  # TODO: delete

        runs.update({k: {'p2c': points2cluster,
                         'loss': loss,
                         'silhouette': S,
                         'S2': S2}
                     })

    # plot points2cluster for different k
    fig, axes = plt.subplots(3, 3, figsize=(12, 12),
                             subplot_kw={'xticks': [], 'yticks': []})

    fig.subplots_adjust(hspace=0.3, wspace=0.05)

    for ax, k in zip(axes.flat, sorted(runs)):
        points2cluster = runs[k]['p2c']
        ax.scatter(X[:, 0], X[:, 1], c=points2cluster, cmap=plt.get_cmap('Set1'))
        ax.set_title(str(k) + ' clusters')

    fig2, (ax0, ax1) = plt.subplots(1, 2, figsize=(8,5))

    #silh = [runs[k]['silhouette'] for k in runs]
    silh2 = [runs[k]['S2'] for k in runs]
    #ax0.plot(range, silh)
    ax0.plot(range, silh2)
    ax0.set_title('Silhouette')

    loss = [runs[k]['loss'] for k in runs]
    ax1.plot(range, loss)
    ax1.set_title('Loss')

    return fig, fig2


def load_data():
    circles = circles_example()

    apml = apml_pic_example()

    idx = np.random.choice(apml.shape[0], 840)
    apml_small = apml[idx]

    data_path = 'microarray_data.pickle'
    with open(data_path, 'rb') as f:
        microarray = pickle.load(f)

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
            'apml_small': {'similarity_param': {'gaussian': 5.,   #
                                    'mnn': 15},
                           'k': 9,
                           'data': apml_small},
            'microarray': {'similarity_param': {'gaussian': 4.77798,
                                    'mnn': 27},
                           'k': 13,
                           'data': microarray},
            'microarray_small': {'similarity_param': {'gaussian': 4.77798,
                                                'mnn': 15},
                           'k': 13,
                           'data': microarray_small},
            'four_gaussians': {'similarity_param': {'gaussian': 0.35,
                                                'mnn': 24},
                           'k': 4,
                           'data': four_gaussians}}
    return data


def get_sigma(X, nn):

    dist = euclid(X, X)
    dist.sort()
    dist = dist[:, 1:]
    sigma = dist[:, :nn].mean(axis=1).mean()
    # dist = dist.flatten()
    # plt.hist(dist, bins=300)
    print(sigma)

    return sigma


def tSNE_vs_PCA(data, tags, n_plot):
    ## PCA

    mu = data.mean(axis=0)
    U, s, V = np.linalg.svd(data - mu, full_matrices=False)
    Zpca = np.dot(data - mu, V.transpose())

    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(data)

    print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))

    fig1, (ax0, ax1) = plt.subplots(ncols=2, figsize=(8, 5))
    ax0.set_title('PCA')
    ax0.scatter(Zpca[:n_plot, 0], Zpca[:n_plot, 1], c=tags[:n_plot], s=5, cmap='tab10')
    ax0.set_xticks([])
    ax0.set_yticks([])

    ax1.set_title('t-SNE')
    im = ax1.scatter(tsne_results[:n_plot, 0], tsne_results[:n_plot, 1], c=tags[:n_plot], s=5, cmap='tab10')
    ax1.set_xticks([])
    ax1.set_yticks([])

    axins = inset_axes(ax1,
                       width="5%",  # width = 10% of parent_bbox width
                       height="100%",  # height : 50%
                       loc='lower left',
                       bbox_to_anchor=(1.05, 0., 1, 1),
                       bbox_transform=ax1.transAxes,
                       borderpad=0,
                       )

    fig1.colorbar(im, cax=axins)  # , ax=ax1)
    fig1.tight_layout()

    return fig1


if __name__ == '__main__':

    data = load_data()

    # ####################################################################################
    # #                 K-means++ Clustering on Synthetic Data                           #
    # ####################################################################################
    #
    # vals = {}
    #
    # for name in ['apml', 'circles', 'four_gaussians']:
    #
    #     print(name)
    #     dataset = name
    #     X = data[dataset]['data']
    #     k = data[dataset]['k']
    #     sigma = data[dataset]['similarity_param']['gaussian']
    #     m = data[dataset]['similarity_param']['mnn']
    #
    #     p2c_kmeans, centers = kmeans(X, k)
    #     p2c_spec_gaussian, centers, t, v = spectral(X, k, similarity_param=sigma, similarity=gaussian_kernel)
    #     p2c_spec_mnn, centers, t, v = spectral(X, k, similarity_param=m, similarity=mnn)
    #
    #     vals.update({'kmeans_' + name: {'X': X,
    #                                     'p2c': p2c_kmeans},
    #                 'gaussian_' + name: {'X': X,
    #                                      'p2c': p2c_spec_gaussian},
    #                  'mnn_' + name: {'X':X,
    #                                  'p2c': p2c_spec_mnn}})
    #
    #     fig, axes = plt.subplots(3, 3, figsize=(12, 12),
    #                              subplot_kw={'xticks': [], 'yticks': []})
    #
    #     fig.subplots_adjust(hspace=0.3, wspace=0.05)
    #
    #     for ax, k in zip(axes.flat, sorted(vals)):
    #         print(k)
    #         X = vals[k]['X']
    #         points2cluster = vals[k]['p2c']
    #         ax.scatter(X[:, 0], X[:, 1], c=points2cluster, cmap=plt.get_cmap('Set1'))
    #         ax.set_title(str(k))
    #
    # fig.savefig('synthetic_data_clustering.png')
    #
    # ####################################################################################
    # #                 choose m  and  choose k silhouette                               #
    # ####################################################################################
    # # TODO: silhouette does not work :-(
    # for name in ['apml_small', 'circles', 'four_gaussians']:
    #     print(name)
    #     dataset = name
    #     X = data[dataset]['data']
    #     k = data[dataset]['k']
    #     similarity_param = data[dataset]['similarity_param']['mnn']
    #
    #     similarity = mnn
    #
    #     # fig1 = choose_m_mnn(X=X, values=[6, 9, 12, 15, 18, 21, 24, 27, 30], k=k)
    #
    #     fig2, fig3 = choose_k_spectral(X, values=range(3,12),
    #                                    similarity_param=similarity_param,
    #                                    similarity=similarity)
    #
    #     # fig1.savefig(name + '_choose_m.png')
    #     fig2.savefig(name + '_choose_k.png')
    #     fig3.savefig(name + '_choose_k_sil.png')
    #
    # ####################################################################################
    # #                           Similarity Plot                                        #
    # ####################################################################################
    #
    # dataset = 'apml_small'
    # X = data[dataset]['data']
    # k = data[dataset]['k']
    # similarity_param = data[dataset]['similarity_param']['mnn']
    # similarity = mnn
    #
    # points2cluster, cdfa, t, v = spectral(X, k, similarity_param=similarity_param, similarity=similarity)
    # fig4 = plot_similarity(X=X, similarity=similarity, similarity_param=similarity_param, points2cluster=points2cluster)
    # fig4.savefig('similarity.png')
    #
    # ####################################################################################
    # #                        Choose k k-mean elbow                                     #
    # ####################################################################################
    #
    # dataset = 'four_gaussians'
    # X = data[dataset]['data']
    # k = data[dataset]['k']
    #
    # points2cluster, centers = kmeans(X=X, k=k)
    # plt.scatter(X[:,0], X[:,1], c=points2cluster)
    #
    # fig5, fig6 = choose_k_kmean(X, range(2,11))
    #
    # fig5.savefig('4gaussians_choose_k.png')
    # fig6.savefig('4gaussians_choose_k_loss.png')
    #
    # ####################################################################################
    # #                               t-SNE vs. PCA                                      #
    # ####################################################################################
    #
    #
    # # digits, tags = load_digits(return_X_y=True)   # small
    # # data = digits / 16
    #
    # (data, tags), (x_test, y_test) = mnist.load_data()   #big
    # data = data.reshape(60000, 784) / 255
    #
    # fig7 = tSNE_vs_PCA(data, tags, 3000)
    # fig7.savefig('tSNE_vs_PCA.png')
    #

    ####################################################################################
    #                          find sigma for microarray                               #
    ####################################################################################
    #
    # for name in ['apml_small', 'apml', 'circles', 'four_gaussians']:
    #     print(name)
    #     dataset = name
    #     X = data[dataset]['data']
    #     k = data[dataset]['k']
    #     sigma = data[dataset]['similarity_param']['gaussian']
    #
    #     dist = euclid(X, X)
    #     dist.sort()
    #     dist = dist[:, 1:]
    #     dist = dist.flatten()
    #     dist.sort()
    #     smaller_sigma = (dist <= sigma)
    #     percentile = smaller_sigma.sum()/smaller_sigma.shape[0]
    #     print(percentile)
    #
    # mean_percentile = 0.0038190077669538578
    #
    # X = data['microarray']['data']
    #
    # dist = euclid(X, X)
    #
    # dist.sort()
    # dist = dist[:, 1:]
    # dist = dist.flatten()
    # dist.sort()
    # dist = dist[307470:] # deleted more zeros
    #
    # idx = dist.shape[0]*mean_percentile
    # sigma = dist[int(idx)]  # 4.77798
    #
    # smaller_sigma = (dist <= sigma)
    # percentile = smaller_sigma.sum()/smaller_sigma.shape[0]
    #
    # print(percentile, mean_percentile)
    #

    ####################################################################################
    #                                 microarray                                       #
    ####################################################################################

    dataset = 'microarray'
    X = data[dataset]['data']
    # k = 8 # data[dataset]['k']
    # similarity_param = data[dataset]['similarity_param']['gaussian']
    # similarity = gaussian_kernel

    # points2cluster, cdfa, t, v = spectral(X, k, similarity_param=similarity_param, similarity=similarity)
    # points2cluster, center = kmeans(X, k)

    # plt.scatter(X[:, 0], X[:, 1], c=points2cluster)
    # plt.bar(range(0, 30), v[:30])

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

        dist_sort = euclid(X_sort, X_sort)
        sim_sort = mnn(dist_sort, 30)
        ax.imshow(sim_sort, extent=[0, 1, 0, 1], vmin=0, vmax=1)
        ax.set_title(str(k) + ' clusters')

    fig8.savefig('microarray_choose_k_sim_big.png')

    dist = euclid(X, X)
    sim = mnn(dist, 30)
    plt.figure()
    plt.imshow(sim, extent=[0, 1, 0, 1], vmin=0, vmax=1)
    plt.colorbar()


    plt.show()

