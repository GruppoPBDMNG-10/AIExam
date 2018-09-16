import common as common
import pandas as pd

import experimentation.clustering.clustering as clustering
import experimentation.common.common as common

result_folder = 'result/';
file = 'dataset/go_track_trackspoints.csv';
result_csv = result_folder + 'go_track_trackpoints_cleaned.csv';
result_coordinates = result_folder + 'coordinates.csv';

chunksize = 100000;
dist = 0;
coordinates = [];

for df in pd.read_csv(file, chunksize=chunksize, iterator=True, dtype={'latitude':float, 'longitude':float}):
    #Get coordinates as touple
    for latitude, longitude in zip(df['latitude'], df['longitude']):
        coordinates.append((latitude, longitude))

dist = common.calculate_distance_array(coordinates)/len(coordinates);
print("Dataset lenght: ", len(coordinates))
print("Dataset average distance: ", dist);

print("Start clustering process");
clusterResult = clustering.make_clustering_2(coordinates, 500);

print("Cluster result: ", clusterResult)
