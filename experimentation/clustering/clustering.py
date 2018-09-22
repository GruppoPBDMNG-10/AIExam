import numpy as np
import geopy.distance
import json
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.externals import joblib

class Result(object):
    mean = 0
    median = 0
    variance = 0
    model = object

    # The class "constructor" - It's actually an initializer
    def __init__(self, mean, median, variance, model):
        self.mean = mean
        self.median = median
        self.variance = variance
        self.model = model

    def make_result(self, mean, median, variance, model):
        result = Result(self, mean, median, variance, model)
        return result

    def __repr__(self):
        return json.dumps(self.__dict__)


def make_clustering_2(coordinates, clusters=int) -> Result:
    # model = KMeans(n_clusters=clusters, random_state=1).fit(coordinates)
    model = MiniBatchKMeans(n_clusters=clusters, random_state=1, init_size=3*clusters).fit(coordinates)

    # These are our fitted labels for clusters -- the first cluster has label 0, and the second has label 1.
    labels = model.labels_
    centroids = model.cluster_centers_

    # Calculate distance between centroids and points
    dist = [None] * len(coordinates)
    for i in range(len(coordinates)):
        key = coordinates[i]
        value = centroids[labels[i]]
        dist[i] = geopy.distance.vincenty((key[0], key[1]), (value[0], value[1])).meters

    # Calculate mean
    mean = np.mean(dist, dtype=np.float64)

    # Calculate median
    median = np.median(dist)

    # Calculate variance
    variance = np.var(dist, dtype=np.float64)

    return Result(mean, median, variance, model)


def predict(model, coordinates):
    return KMeans.predict(model, [(float(coordinates[0]), float(coordinates[1]))])[0]


def dump_model(model, file_path=str):
    joblib.dump(model, file_path)


def model_from_dump(file_path=str):
    return joblib.load(file_path)