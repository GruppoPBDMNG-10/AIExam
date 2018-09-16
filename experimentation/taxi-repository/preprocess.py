import re
import pandas as pd
import itertools
from common import common
from clustering import clustering
import geopy.distance
import numpy as np
from sqlalchemy import create_engine
from sklearn.externals import joblib


def polyline_to_coordinates(polyline, regex):
    """It transforms the polyline column in coordinates array"""
    coordinates_array = []
    matches = re.finditer(regex, polyline)
    # matches = re.finditer(r'\[(-?[0-9]{1,2}\.[0-9]+),\s(-?[0-9]{1,2}\.[0-9]+)\]', polyline)
    for match in matches:
        coordinates_array.append((float(match.group(1)), float(match.group(2))))

    return coordinates_array


def drop_columns(data_frame=pd.DataFrame, columns_name=list):
    """Drop specified columns from the required data frame"""
    return data_frame.drop(columns_name, axis=1)


# csv_database = create_engine('sqlite:///taxi.db')
result_folder = 'result/'
file = 'dataset/taxi_small.zip'
result_csv = result_folder + 'taxi_cleaned.csv'
result_coordinates = result_folder + 'coordinates.csv'
result_model = result_folder + 'model.pkl'
columns_to_remove = ["CALL_TYPE", "ORIGIN_CALL", "ORIGIN_STAND", "DAY_TYPE", "MISSING_DATA"]
chunk_size = 100000
dist = 0
print_header = True
write_mode = 'w'
coordinates = []

for df in pd.read_csv(file, chunksize=chunk_size, iterator=True, dtype={'POLYLINE': str}):
    # Remove undesired columns from data set
    print("Start drop columns")
    df = drop_columns(df, columns_to_remove)

    # Change coordinates format to be python compliant
    print("Start polyline refactoring")
    df['POLYLINE'] = df['POLYLINE'].\
        apply(lambda x: polyline_to_coordinates(x, r'\[(-?[0-9]{1,2}\.[0-9]+),\s(-?[0-9]{1,2}\.[0-9]+)\]'))

    print("Start calculating coordinates list")
    coordinates = itertools.chain(coordinates, list(itertools.chain.from_iterable(df['POLYLINE'])))

    # TODO: Esegui rimozione delle tuple senza coordinate
    # df = df[df['POLYLINE'].size > 0]

    # Write chunk to result csv file
    print("Start file writing")
    df.to_csv(result_csv, mode=write_mode, header=print_header, index=False)

    # rint CSV header only the first time
    print_header = False

    # Change write mode to append. In this way, if the file already exists, it is rewritten only the first time.
    write_mode = 'a'

# With itertool, coordinates needs to be stored in memory after the execution.
print("Start coordinates file creation process.")
coordinates = list(coordinates)
coordinates_df = pd.DataFrame.from_records(coordinates)
coordinates_df.to_csv(result_coordinates, mode='w', header=False, index=False)

dist = common.calculate_distance_array(coordinates)/len(coordinates)
print("Data set average distance: ", dist)

print("Start clustering process")
clusterResult = clustering.make_clustering_2(coordinates, 500)
print("Cluster completed")

print("Dump model")
joblib.dump(clusterResult.model, result_model)

print_header = True
write_mode = 'w'
for df in pd.read_csv(result_csv, chunksize=chunk_size, iterator=True, dtype={'POLYLINE': str}):

    df['POLYLINE'] = df['POLYLINE'].\
        apply(lambda x: [clustering.predict(clusterResult.model, point) for point in
                         polyline_to_coordinates(x, r'\((-?[0-9]{1,2}\.[0-9]+),\s(-?[0-9]{1,2}\.[0-9]+)\)')])
    df.to_csv(result_csv, mode=write_mode, header=print_header, index=False)

    # Print CSV header only the first time
    print_header = False

    # Change write mode to append. In this way, if the file already exists, it is rewrited only the first time.
    write_mode = 'a'







