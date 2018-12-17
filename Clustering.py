import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import silhouette_score  # TODO: delete

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

    return circles, points2cluster


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
            # dist[i,j] = np.sqrt(np.dot(X[i], X[i].T) - 2 * np.dot(X[i], Y[j].T) + np.dot(Y[j], Y[j].T))
            dist[i,j] = np.linalg.norm(X[i] - Y[j])

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
        t = metric(center, X)      # distance between data points and center that already were chosen
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

    for i in range(0, iterations):
        print('iteration ' + str(i))
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
    W = np.exp(-X/(2*sigma**2))
    return W


def mnn(X, m):
    """
    calculate the m nearest neighbors similarity of the given distance matrix.
    :param X: A NxN distance matrix.
    :param m: The number of nearest neighbors.
    :return: NxN similarity matrix.
    """

    # TODO: YOUR CODE HERE
    dist_idx = np.argsort(X, axis=1)
    nearest_idx = dist_idx[:, :(m+1)]

    NN = np.zeros(X.shape)
    for i in range(0, X.shape[0]):
        NN[i, nearest_idx[i]] += 1 # so far only one directional nn, not mutual nn

    return NN


def spectral(X, k, similarity_param, similarity=gaussian_kernel):
    """
    Cluster the data into k clusters using the spectral clustering algorithm.
    :param X: A NxD data matrix.
    :param k: The number of desired clusters.
    :param similarity_param: m for mnn, sigma for the Gaussian kernel.
    :param similarity: The similarity transformation of the data.
    :return: clustering, as in the kmeans implementation.
    """

    # TODO: YOUR CODE HERE
    # X.sort()
    Distance = euclid(X, X)

    # Distance = (Distance - Distance.mean(axis=0)) / Distance.std(axis=0)

    Adjacency = similarity(Distance, similarity_param)
    diag = Adjacency.sum(axis=0)
    Degree = np.diag(diag)

    L = Degree - Adjacency
    L_sym = np.multiply(np.sqrt(diag)**(-1), np.multiply(L, np.sqrt(diag)**(-1)))
    #L_sym = np.sqrt(np.linalg.inv(Degree)) * L * np.sqrt(np.linalg.inv(Degree))

    v, U = np.linalg.eig(L_sym)
    v_real = v.real
    U_real = U.real

    v_idx = np.matrix((np.arange(v.shape[0]), v_real))
    v_sort = v_idx.T.tolist()

    v_sort.sort(key=lambda x: x[1])
    v_sort = np.matrix(v_sort)

    mask = (~(v_sort[:,1]==0)).flatten().tolist()[0]
    # mask = (v_sort[:, 1] > 0).flatten().tolist()[0]

    v_sort = v_sort[mask, :]
    uk_idx = v_sort[:k,0].flatten().tolist()[0]
    uk_idx = list(map(int, uk_idx))

    T = U_real[:, uk_idx] # no = np.linalg.norm(t, axis=1) T[~(no<=1)] idx[~(no<=1)] some rows are = 0... why? other eigenvectors? only the ones that are bigge tahn 0?

    # plt.plot(range(0,840), u2)  # when X.sort(axis=0) this is Fiedler Vector - maybe does not have to be sorted
    # plt.bar(range(0,10), v_sort[:10, 1].flatten().tolist()[0])

    row_sums = np.linalg.norm(T, axis=1)
    t = T / row_sums[:, np.newaxis]
    print('now kmeans clustering')

    points2cluster, center = kmeans(t, k)

    return points2cluster, center, t, v_sort

similarity = gaussian_kernel
similarity_param = 0.35358367870405927
k = 4

similarity = mnn
similarity_param = 50
k = 4

plt.scatter(X[:,0], X[:,1], c=points2cluster)
plt.bar(range(0,30), v_sort[:30, 1].flatten().tolist()[0])

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

    fig1, (ax0, ax1) = plt.subplots(ncols=2, figsize=(8, 5))

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

        mask_not_i = ~(assigned[2, :] == i)
        b_j_list = []
        clusters_without_i = np.delete(clusters, i)

        for j in clusters_without_i:
            mask_is_j = (assigned[2, :] == j).tolist()
            b_ij = assigned[:2, mask_is_j].T

            b_j = euclid(a_i, b_ij).mean(axis=1).sum()

            b_j_list = b_j_list + [b_j]
            b = np.min(np.array(b_j_list))

        b_list = b_list + [b]

    S = 1/(k+1) * ((np.array(b_list) - np.array(a_list))/np.array([a_list, b_list]).max(axis=0)).sum()

    return S


def choose_k(X, range):
    runs = {}
    for k in range:

        print(str(k) + '  clusters')
        points2cluster, centers = kmeans(X=X, k=k)

        #tr = translate(points2cluster)

        dist = euclid(X, centers)
        dist = dist.min(axis=1)
        loss = dist.sum()

        S = silhouette(X, points2cluster)
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

    for ax, k in zip(axes.flat, runs):
        points2cluster = runs[k]['p2c']
        ax.scatter(X[:, 0], X[:, 1], c=points2cluster, cmap=plt.get_cmap('Set1'))
        ax.set_title(str(k) + ' clusters')

    fig2, (ax0, ax1) = plt.subplots(1, 2, figsize=(8,5))

    silh = [runs[k]['silhouette'] for k in runs]
    silh2 = [runs[k]['S2'] for k in runs]
    ax0.plot(range, silh)
    ax0.plot(range, silh2)
    ax0.set_title('Silhouette')

    loss = [runs[k]['loss'] for k in runs]
    ax1.plot(range, loss)
    ax1.set_title('Loss')
    plt.show()


if __name__ == '__main__':

    # TODO: YOUR CODE HERE
    X = four_gaussians_example() # gk nn 5 # mnn 50

    X, p2c_ref = circles_example() # gk nn 5 # mnn 10, 50 wie kmeans

    # X = apml_pic_example() # gk nn  # mnn 100... maybe even more
    # idx = np.random.choice(X.shape[0], 840)
    # X = X[idx]

    similarity = mnn
    similarity_param = 50

    similarity = gaussian_kernel
    similarity_param = 0.37

    #points2cluster, centers = kmeans(X, 4)

    nn = 4
    #dist = euclid(X,X)
    #dist.sort()
    #dist = dist[:,1:]
    #sigma = dist[:,1:nn].mean(axis=1).mean()
    #print(sigma)
    #points2cluster, centers = spectral(X=X, k=4,
     #                                  similarity_param=sigma,
      #                                 similarity=gaussian_kernel)

    #plt.scatter(X[:,0], X[:,1], c=points2cluster)

    #plot_similarity(X=X, similarity=gaussian_kernel,
     #               similarity_param=sigma,
      #              points2cluster=points2cluster)



    nn = 11
    points2cluster, centers = spectral(X=X, k=4,
                                       similarity_param=nn,
                                       similarity=mnn)


    plt.scatter(X[:,0], X[:,1], c=points2cluster)

    plot_similarity(X=X, similarity=mnn,
                    similarity_param=nn,
                    points2cluster=points2cluster)


    #plt.plot(centers[:, 0], centers[:, 1], 'ob')
    plt.show()

##################### Choosing K ###########################

    with open('circles_data.pickle', 'rb') as handle:
        X = pickle.load(handle)

    with open('circles_p2c.pickle', 'rb') as handle:
        p2c_ref = pickle.load(handle)

    choose_k(X, range(2,6))

    nn = 5
    dist = euclid(X,X)
    dist.sort()
    dist = dist[:,1:]
    sigma = dist[:,:nn].mean(axis=1).mean()
    dist = dist.flatten()
    plt.hist(dist, bins=300)
    print(sigma)

    points2cluster, centers, t, v_sort = spectral(X=X, k=4,
                                          similarity_param=sigma,
                                          similarity=gaussian_kernel)

    plt.scatter(X[:,0], X[:,1], c=points2cluster)
    plt.bar(range(0,30), v_sort[:30, 1].flatten().tolist()[0])


    ###########################microarray###################################


    data_path = 'microarray_data.pickle'
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    idx = np.random.choice(data.shape[0], 840)  # mnn nn 50 for 840 data
    X = data[idx]
