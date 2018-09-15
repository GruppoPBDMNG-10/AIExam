import pandas as pd
import itertools
from common import common
from clustering import clustering
import geopy.distance
import numpy as np
from sqlalchemy import create_engine


def polyline_to_coordinates(polyline):
    """It transforms the polyline column in coordinates array"""
    coordinates = [];
    list = polyline.replace("[", "").replace("]", "").split(",");
    if len(list) > 1 & len(list) % 2 == 0:
        for i in range(0, len(list) - 1, 2):
            coordinates.append((float(list[i]),float(list[i+1])));
    return coordinates;


def drop_columns(df=pd.DataFrame, columsName=list):
    """Drop specified colums from the required dataframe"""
    return df.drop(columsName, axis=1);

#csv_database = create_engine('sqlite:///taxi.db');
result_folder = 'result/';
file = 'dataset/taxi.zip';
result_csv = result_folder + 'taxi_cleaned.csv';
result_coordinates = result_folder + 'coordinates.csv';

columns_to_remove = ["CALL_TYPE","ORIGIN_CALL","ORIGIN_STAND","DAY_TYPE","MISSING_DATA"];

chunksize = 100000;
dist = 0;
print_header = True;
write_mode = 'w';

coordinates = [];

for df in pd.read_csv(file, chunksize=chunksize, iterator=True, dtype={'POLYLINE': str}):
    #Remove undesired colums from dataset
    print("Start drop colums");
    df = drop_columns(df, columns_to_remove);

    #Change coordinates format to be python compliant
    print("Start polyline refactoring");
    df['POLYLINE'] = df['POLYLINE'].apply(lambda x: polyline_to_coordinates(x));

    print("Start calculating coordinates list");
    coordinates = itertools.chain(coordinates, list(itertools.chain.from_iterable(df['POLYLINE'])));

    #TODO: Esegui rimozione delle tuple senza coordinate
    #df = df[df['POLYLINE'].size > 0];

    # Write chunk to result csv file
    print("Start file writing");
    df.to_csv(result_csv, mode=write_mode, header=print_header, index=False);

    #Print CSV header only the first time
    print_header = False;

    #Change write mode to append. In this way, if the file already exists, it is rewrited only the first time.
    write_mode = 'a';

#With itertool, coordinates needs to be stored in memory after the execution.
print("Start coordinates file creation process.");
coordinates = list(coordinates);
coordinates_df = pd.DataFrame.from_records(coordinates);
coordinates_df.to_csv(result_coordinates, mode='w', header=False, index=False);

dist = common.calculate_distance_array(coordinates)/len(coordinates);
print("Dataset average distance: ", dist);

print("Start clustering process");
clusterResult = clustering.make_clustering_2(coordinates, 500);

#TODO: non funziona, capire come mai
#print("Saving cluster to file");
#common.to_json_file(result_folder+'cluster.json', clusterResult.__repr__());








