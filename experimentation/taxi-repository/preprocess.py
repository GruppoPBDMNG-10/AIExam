from pathlib import Path
import pandas as pd
import numpy as np
import itertools
import time

from dateutil.parser import _resultbase

import experimentation.clustering.clustering as clustering
import experimentation.common.common as common


def drop_columns(data_frame=pd.DataFrame, columns_name=list):
    """Drop specified columns from the required data frame"""
    return data_frame.drop(columns_name, axis=1)


def tidy_split(df, column, sep=',', keep=False):
    """
    Split the values of a column and expand so the new DataFrame has one split
    value per row. Filters rows where the column is missing.

    Params
    ------
    df : pandas.DataFrame
        dataframe with the column to split and expand
    column : str
        the column to split and expand
    sep : str
        the string used to split the column's values
    keep : bool
        whether to retain the presplit value as it's own row

    Returns
    -------
    pandas.DataFrame
        Returns a dataframe with the same columns as `df`.
    """
    indexes = list()
    new_values = list()
    df = df.dropna(subset=[column])
    for i, presplit in enumerate(df[column].astype(str)):
        values = presplit.split(sep)
        if keep and len(values) > 1:
            indexes.append(i)
            new_values.append(presplit)
        for value in values:
            indexes.append(i)
            new_values.append(value.replace('[', '').replace(']', ''))
    new_df = df.iloc[indexes, :].copy()
    new_df[column] = new_values
    return new_df.reset_index(drop=True)


def explode(df, cols, split_on=',') -> pd.DataFrame:
    """
    Explode dataframe on the given column, split on given delimeter
    """
    cols_sep = list(set(df.columns) - set(cols))
    df_cols = df[cols_sep]
    explode_len = df[cols[0]].str.split(split_on).map(len)
    repeat_list = []
    for r, e in zip(df_cols.as_matrix(), explode_len):
        repeat_list.extend([list(r)] * e)

    df_repeat = pd.DataFrame(repeat_list, columns=cols_sep)
    df_explode = pd.concat([df[col].str.split(split_on, expand=True).stack().str.strip().reset_index(drop=True)
                            for col in cols], axis=1)

    df_explode = pd.DataFrame(df_explode)[0].apply(lambda x: x.replace('[', '').replace(']', ''))

    df_explode.columns = cols
    return pd.concat((df_repeat, df_explode), axis=1)


start_time = time.time()
# csv_database = create_engine('sqlite:///taxi.db')
result_folder = 'result/'
file = 'dataset/taxi.zip'
result_csv = result_folder + 'taxi_cleaned.csv'
result_coordinates = result_folder + 'coordinates.csv'
result_model = result_folder + 'model.pkl'
columns_to_remove = ["CALL_TYPE", "ORIGIN_CALL", "ORIGIN_STAND", "DAY_TYPE", "MISSING_DATA"]
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
    print("Start dataset processing")
    for df in pd.read_csv(file, chunksize=chunk_size, iterator=True,
                          usecols=['TRIP_ID', 'TAXI_ID', 'TIMESTAMP', 'POLYLINE'],
                          dtype={'TRIP_ID': str, 'TAXI_ID': int, 'TIMESTAMP': int, 'POLYLINE': str}):
        # Remove undesired columns from data set
        # print("Start drop columns")
        #df = drop_columns(df, columns_to_remove)

        # Change coordinates format to be python compliant
        print("Start polyline refactoring")
        df['POLYLINE'] = df['POLYLINE']. \
            apply(lambda x: common.polyline_to_coordinates(x, r'\[(-?[0-9]{1,2}\.[0-9]+),\s*(-?[0-9]{1,2}\.[0-9]+)\]'))

        print("Start calculating coordinates list")
        coordinates = itertools.chain(coordinates, list(itertools.chain.from_iterable(df['POLYLINE'])))

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

file_processing = time.time()
#print("File processing time is %s seconds" % file_processing - start_time)
dist = common.calculate_average_distance(coordinates)
#print("Data set average distance: ", dist)

clusterFile = Path(result_model)
clusterResult = None
clusterModel = None

pre_clustering_time = time.time()

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

after_clustering_time = time.time()
#print("Clustering processing time is %s seconds" % after_clustering_time - pre_clustering_time)
del coordinates

print_header = True
write_mode = 'w'
for df in pd.read_csv(result_csv, chunksize=chunk_size, iterator=True,
                      usecols=['TRIP_ID', 'TAXI_ID', 'TIMESTAMP', 'POLYLINE'],
                      dtype={'TRIP_ID': str, 'TAXI_ID': int, 'TIMESTAMP': int, 'POLYLINE': str}):

    df['POLYLINE'] = df['POLYLINE']. \
        apply(lambda x: [clustering.predict(clusterResult.model, point) for point in
                         common.polyline_to_coordinates(x,
                                                        r'\((-?[0-9]{1,2}\.[0-9]+),\s(-?[0-9]{1,2}\.[0-9]+)\)')]).astype(
        str)

    df = tidy_split(df, 'POLYLINE')

    df.rename(columns={df.columns[3]: "GATE"}, inplace=True)

    df['GATE'] = df['GATE'].apply(lambda x: x.strip())

    lastId = "";
    lastGate = ""
    count = 0;
    for row in df.itertuples():
        if df.at[row.Index, 'TRIP_ID'] == lastId:
            count += 15
            df.at[row.Index, 'TIMESTAMP'] = count
            if df.at[row.Index, 'GATE'] != lastGate:
                lastGate = df.at[row.Index, 'GATE']
                df.at[row.Index, 'TIMESTAMP'] = count
        else:
            count = df.at[row.Index, 'TIMESTAMP']
            lastGate = df.at[row.Index, 'GATE']
            lastId = df.at[row.Index, 'TRIP_ID']

    df.drop_duplicates(['TRIP_ID', 'GATE'], inplace=True)

    df.to_csv(result_folder + 'taxi_gate.csv', mode=write_mode, header=print_header, index=False)

    # Print CSV header only the first time
    print_header = False

    # Change write mode to append. In this way, if the file already exists, it is rewrited only the first time.
    write_mode = 'a'

end_time = time.time()
#print("Final file processing time is %s seconds." % end_time - after_clustering_time)
#print("Total processing time is %s seconds." % end_time - start_time)