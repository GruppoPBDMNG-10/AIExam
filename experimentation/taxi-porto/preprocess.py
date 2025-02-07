from pathlib import Path
import pandas as pd
import itertools
import time
import experimentation.clustering.clustering as clustering
import experimentation.common.common as common
import numpy as np
from numba import jit


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


def manage_timestamp():
    lat_id = ""
    last_gate = ""
    count = 0

    def update_timestamp(df):
        global last_id, count, last_gate
        if df['TRIP_ID'] == last_id:
            count += 15
            df.at['TIMESTAMP'] = count
            if df['GATE'] != last_gate:
                last_gate = df['GATE']
                df.at['TIMESTAMP'] = count
        else:
            count = df['TIMESTAMP']
            last_gate = df['GATE']
            last_id = df['TRIP_ID']


@jit()
def gate_substitution(model, polylines=np.array):
    #return [clustering.predict(clusterResult.model, point) for point in common.polyline_to_coordinates(np.array2string(points), r'\((-?[0-9]{1,2}\.[0-9]+),\s(-?[0-9]{1,2}\.[0-9]+)\)')]
    return [clustering.predict(clusterResult.model, point) for point in [common.polyline_to_coordinates(polyline, r'\((-?[0-9]{1,2}\.[0-9]+),\s(-?[0-9]{1,2}\.[0-9]+)\)') for polyline in np.nditer(polylines, flags=['refs_ok'], op_dtypes=['str'])]]


start_time = time.time()
# csv_database = create_engine('sqlite:///taxi.db')
result_folder = 'result/'
file = 'dataset/taxi.zip'
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
    print("Start dataset processing")
    for df in pd.read_csv(file, chunksize=chunk_size, iterator=True,
                          usecols=['TRIP_ID', 'MISSING_DATA', 'TAXI_ID', 'TIMESTAMP', 'POLYLINE'],
                          dtype={'TRIP_ID': str, 'MISSING_DATA': bool, 'TAXI_ID': int, 'TIMESTAMP': int,
                                 'POLYLINE': str}):
        # Remove undesired columns from data set
        df = pd.DataFrame(df)
        print("Start row filtering")
        df = df[df['MISSING_DATA'] == False]
        df = df[df['POLYLINE'] != '[]']

        print("Start drop columns")
        df.drop(labels=['MISSING_DATA'], axis=1, inplace=True)

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
# print("File processing time is %s seconds" % file_processing - start_time)
dist = common.calculate_average_distance(coordinates)
# print("Data set average distance: ", dist)

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
# print("Clustering processing time is %s seconds" % after_clustering_time - pre_clustering_time)
del coordinates

print_header = True
write_mode = 'w'

print("Start reading cleaned file")
for df in pd.read_csv(result_csv, chunksize=chunk_size, iterator=True,
                      usecols=['TRIP_ID', 'TAXI_ID', 'TIMESTAMP', 'POLYLINE'],
                      dtype={'TRIP_ID': str, 'TAXI_ID': int, 'TIMESTAMP': int, 'POLYLINE': str}):

    print("Start polyline substitution")
    df['POLYLINE'] = df['POLYLINE']. \
        apply(lambda x: [clustering.predict(clusterResult.model, point) for point in
                         common.polyline_to_coordinates(x,
                                                        r'\((-?[0-9]{1,2}\.[0-9]+),\s(-?[0-9]{1,2}\.[0-9]+)\)')]).astype(
        str)

    print("Start tidy_split")
    df = tidy_split(df, 'POLYLINE')

    print("Start columns rename")
    df.rename(columns={df.columns[1]: "DRIVER_ID", df.columns[3]: "GATE"}, inplace=True)

    df['GATE'] = df['GATE'].apply(lambda x: x.strip())

    lastId = "";
    lastGate = ""
    counter = 0;

    print("Start timestamps fixing")
    #df = df.apply(lambda x: update_timestamp(x), axis=1)

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

    print("Start drop duplicates")
    df.drop_duplicates(['TRIP_ID', 'GATE'], inplace=True)

    print("Start file writing")
    df.to_csv(result_folder + 'taxi_porto_gate.csv', mode=write_mode, header=print_header, index=False)

    # Print CSV header only the first time
    print_header = False

    # Change write mode to append. In this way, if the file already exists, it is rewrited only the first time.
    write_mode = 'a'

end_time = time.time()
# print("Final file processing time is %s seconds." % end_time - after_clustering_time)
# print("Total processing time is %s seconds." % end_time - start_time)
