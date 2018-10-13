import glob
import os
import pandas as pd
from pathlib import Path
import experimentation.common.common as common
import numpy as np
from numba import jit

PATH = 'dataset/'
RESULT_PATH = 'result/'
RESULT_FILE = RESULT_PATH + 'trap_gate.csv'
SIX_HOURS = 6 * 60 * 60


@jit()
def calculate_trip_id(df=pd.DataFrame, driver_map=dict, timestamp_map=dict, last_trip_index=np.array):
    timestamp = None
    trip_id = driver_map.get(df['DRIVER_ID'])
    if trip_id:
        # The current driver is already running in a trip
        timestamp = timestamp_map.get(df['DRIVER_ID'])
        delta = df['TIMESTAMP'] - timestamp
        if delta > SIX_HOURS:
            # the trip is ended. Increment trip id and save it.
            last_trip_index[0][0] += 1
            trip_id = last_trip_index[0][0]
            driver_map[df['DRIVER_ID']] = trip_id
    else:
        last_trip_index[0][0] += 1
        trip_id = last_trip_index[0][0]
        driver_map[df['DRIVER_ID']] = trip_id

    # Update timestamp map with new timestamp recorded
    timestamp_map[df['DRIVER_ID']] = df['TIMESTAMP']

    return trip_id


print("Start loading csv files from folder")
all_files = glob.glob(os.path.join(PATH, '*.csv'))  # make list of paths
first_file = True
dfs = []

for file in all_files:
    # Reading the file content to create a DataFrame
    df = pd.read_csv(file, sep=";", parse_dates=['timestamp'], infer_datetime_format=True,
                     dtype={'targa': str, 'varco': int, 'corsia': int, 'timestamp': str, 'nazione': str})
    first_file = False
    df.rename(index=str, columns={"targa": "DRIVER_ID", "varco": "GATE", "corsia": "LANE", "timestamp": "TIMESTAMP",
                                  "nazione": "STATE"}, inplace=True)

    # Convert datetime to unix timestamp
    df['TIMESTAMP'] = df['TIMESTAMP'].apply(lambda x: common.datetime_to_unix(x))

    dfs.append(df)

print("Start concatenating df")
big_frame = pd.concat(dfs, ignore_index=True)

# Define a trip
print("Start trip identification")
driver_map = dict()
timestamp_map = dict()
last_trip_index = np.zeros((1, 1), dtype=int)
big_frame['TRIP_ID'] = big_frame.apply(lambda x: calculate_trip_id(x, driver_map, timestamp_map, last_trip_index),
                                       axis=1)
big_frame = big_frame.reindex(columns=['TRIP_ID'] + big_frame.columns[:-1].tolist())

# Write chunk to result csv file
print("Start file writing")
big_frame.to_csv(RESULT_PATH, mode='w', header=True, index=False)
