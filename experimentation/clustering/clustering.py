import numpy as np
import geopy.distance
from scipy.cluster.vq import kmeans2
import json
from sklearn.cluster import KMeans, MiniBatchKMeans
import matplotlib.pyplot as plt
from scipy.spatial import distance


class Result(object):
    clusters = [];
    coordinates = [];
    labels = [];
    distance = [];
    dict = {};
    mean = 0;
    median = 0;
    variance = 0;

    # The class "constructor" - It's actually an initializer
    def __init__(self, clusters, coordinates, labels, distance, dict, mean, median, variance):
        self.clusters = clusters;
        self.coordinates = coordinates;
        self.labels = labels;
        self.dict = dict;
        self.distance = distance;
        self.mean = mean;
        self.median = median;
        self.variance = variance;

    def make_result(clusters, coordinates, labels, distance, dict, mean, median, variance):
        result = Result(clusters, coordinates, labels, dict, distance, mean, median, variance);
        return result;

    def __repr__(self):
        #return jsonpickle.encode(self);
        return json.dumps(self.__dict__);


def make_clustering(coordinates, clusters=int, iter=int):
    centroids, labels = kmeans2(coordinates, 200, iter=iter, minit="points");

    #Calculate distance between centroids and points
    dist = [None] * len(coordinates);
    dict = {};
    key = [];
    value = [];
    for i in range(len(coordinates)):
        key = coordinates[i];
        value = centroids[labels[i]];
        dict[','.join(map(str, key))] = value;
        dist[i] = geopy.distance.vincenty((key[0], key[1]), (value[0], value[1])).meters;

    # Calculate mean
    mean = np.mean(dist, dtype=np.float64);

    # Calculate median
    median = np.median(dist);

    # Calculate variance
    variance = np.var(dist, dtype=np.float64);

    return Result(centroids.tolist(), coordinates, labels.tolist(), dict, dist, mean.tolist(), median.tolist(), variance.tolist());


def make_clustering_2(coordinates, clusters=int):
    #kmeans_model = KMeans(n_clusters=clusters, random_state=1).fit(coordinates);
    kmeans_model = MiniBatchKMeans(n_clusters=clusters, random_state=1, init_size=3*clusters).fit(coordinates);

    # These are our fitted labels for clusters -- the first cluster has label 0, and the second has label 1.
    labels = kmeans_model.labels_;

    centroids = kmeans_model.cluster_centers_;

    # Calculate distance between centroids and points
    dist = [None] * len(coordinates);
    dict = {};
    key = [];
    value = [];
    for i in range(len(coordinates)):
        key = coordinates[i];
        value = centroids[labels[i]];
        dict[','.join(map(str, key))] = value.tolist();
        dist[i] = geopy.distance.vincenty((key[0], key[1]), (value[0], value[1])).meters;

    # Calculate mean
    mean = np.mean(dist, dtype=np.float64);

    # Calculate median
    median = np.median(dist);

    # Calculate variance
    variance = np.var(dist, dtype=np.float64);

    return Result(centroids.tolist(), [], labels.tolist(), dict, dist, mean, median, variance);


"""result = make_clustering(coordinates, 200, 200);
print(result);

result = make_clustering_2(coordinates, 200);
print(result);

sum_of_squared_distances = [];
K = range(1, 500);

for k in K:
    result = make_clustering_2(coordinates, k);
    sum_of_squared_distances.append(result.mean);

plt.plot(K, sum_of_squared_distances, 'bx-');
plt.xlabel('k');
plt.ylabel('Sum_of_squared_distances');
plt.title('Elbow Method For Optimal k');
plt.show();"""