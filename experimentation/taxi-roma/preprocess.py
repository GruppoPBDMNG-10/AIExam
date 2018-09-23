from pathlib import Path
import re
import pandas as pd
import numpy as np
import itertools
import experimentation.clustering.clustering as clustering
import experimentation.common.common as common

driver_map = {}
timestamp_map = {}
last_trip_index = 0;
four_hours = 4*60*60

def calculate_trip_id(df):
    global last_trip_index
    timestamp = None
    trip_id = driver_map.get(df['TAXI_ID'])
    if trip_id:
        #The current driver is already running in a trip
        timestamp = timestamp_map.get(df['TAXI_ID'])
        delta = df['TIMESTAMP'] - timestamp
        if delta > four_hours:
            #the trip is ended. Increment trip id and save it.
            last_trip_index += 1
            trip_id = last_trip_index
            driver_map[df['TAXI_ID']] = trip_id
    else:
        last_trip_index += 1
        trip_id = last_trip_index
        driver_map[df['TAXI_ID']] = trip_id

    # Update timestamp map with new timestamp recorded
    timestamp_map[df['TAXI_ID']] = df['TIMESTAMP']

    return trip_id


result_folder = 'result/'
file = 'dataset/taxi_rome.csv'
result_csv = result_folder + 'taxi_cleaned.csv'
result_coordinates = result_folder + 'coordinates.csv'
result_model = result_folder + 'model.pkl'
chunk_size = 500000
dist = 0
print_header = True
write_mode = 'w'
coordinates = []

resultFile = Path(result_csv)
coordinatesFile = Path(result_coordinates)

if resultFile.is_file() & resultFile.exists() & coordinatesFile.is_file() & coordinatesFile.exists():
    print("Loading coordinates from file")

    for df in pd.read_csv(coordinatesFile, chunksize=chunk_size, iterator=True, header=None):
        df = pd.DataFrame(df)
        coordinates = itertools.chain(coordinates, zip(df.iloc[:, 0], df.iloc[:, 1]))

    print("Start coordinates list creation")
    coordinates = list(coordinates)
    print("Coordinates list created")

else:
    print("Start loading csv file")
    for df in pd.read_csv(file, chunksize=chunk_size, iterator=True, names=["TAXI_ID", "TIMESTAMP", "POINT"], sep=";",
                          parse_dates=['TIMESTAMP'], infer_datetime_format=True, dtype={'TAXI_ID': int, 'POINT': str}):
        df = pd.DataFrame(df)

        # Convert datetime to unix timestamp
        df['TIMESTAMP'] = df['TIMESTAMP'].apply(lambda x: common.datetime_to_unix(x))

        # Change coordinates format to be python compliant
        print("Start point refactoring")
        df['POINT'] = df['POINT']. \
            apply(
            lambda x: common.polyline_to_coordinates(x, r'POINT\((-?[0-9]{1,2}\.[0-9]+) (-?[0-9]{1,2}\.[0-9]+)\)'))

        print("Start calculating coordinates list")
        coordinates = itertools.chain(coordinates, list(itertools.chain.from_iterable(df['POINT'])))

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
    del coordinates_df

dist = common.calculate_average_distance(coordinates)
print("Data set average distance: ", dist)

clusterFile = Path(result_model)
clusterResult = None
clusterModel = None

if clusterFile.is_file() & clusterFile.exists():
    print("Loading cluster model from file")
    clusterModel = clustering.model_from_dump(clusterFile)
    clusterResult = clustering.Result(0, 0, 0, clusterModel)
    print("Cluster model loaded: ", clusterModel)
else:
    print("Start clustering process")
    clusterResult = clustering.make_clustering_2(coordinates, 500)
    print("Cluster completed")
    print("Dump model")
    clustering.dump_model(clusterResult.model, result_model)
    print("Dump completed")

del coordinates

print_header = True
write_mode = 'w'

for df in pd.read_csv(result_csv, chunksize=chunk_size, iterator=True, dtype={'TAXI_ID': int, 'POINT': str}):
    df = pd.DataFrame(df)

    df['POINT'] = df['POINT'].apply(lambda x: [clustering.predict(clusterResult.model, point) for point in
                                               common.polyline_to_coordinates(x,
                                                                              r'\((-?[0-9]{1,2}\.[0-9]+),\s(-?[0-9]{1,2}\.[0-9]+)\)')]).astype(
        str)

    df.rename(columns={df.columns[2]: "GATE"}, inplace=True)

    df.drop_duplicates(['TAXI_ID', 'GATE'], inplace=True)

    # Define a trip
    df['TRIP_ID'] = df.apply(lambda x: calculate_trip_id(x), axis=1)
    df = df.reindex(columns=['TRIP_ID'] + df.columns[:-1].tolist())

    df.to_csv(result_folder + 'taxi_rome_gate.csv', mode=write_mode, header=print_header, index=False)

    # Print CSV header only the first time
    print_header = False

    # Change write mode to append. In this way, if the file already exists, it is rewrited only the first time.
    write_mode = 'a'