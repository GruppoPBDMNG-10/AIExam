import re
import pandas as pd
import numpy as np
import itertools
import experimentation.clustering.clustering as clustering
import experimentation.common.common as common


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


def explode(df, cols, split_on=',') -> pd.DataFrame:
    """
    Explode dataframe on the given column, split on given delimeter
    """
    cols_sep = list(set(df.columns) - set(cols))
    df_cols = df[cols_sep]
    explode_len = df[cols[0]].str.split(split_on).map(len)
    repeat_list = []
    for r, e in zip(df_cols.as_matrix(), explode_len):
        repeat_list.extend([list(r)]*e)

    df_repeat = pd.DataFrame(repeat_list, columns=cols_sep)
    df_explode = pd.concat([df[col].str.split(split_on, expand=True).stack().str.strip().reset_index(drop=True)
                            for col in cols], axis=1)

    df_explode = pd.DataFrame(df_explode)[0].apply(lambda x: x.replace('[', '').replace(']', ''))

    df_explode.columns = cols
    return pd.concat((df_repeat, df_explode), axis=1)


# csv_database = create_engine('sqlite:///taxi.db')
result_folder = 'result/'
file = 'dataset/taxi.zip'
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
        apply(lambda x: polyline_to_coordinates(x, r'\[(-?[0-9]{1,2}\.[0-9]+),\s*(-?[0-9]{1,2}\.[0-9]+)\]'))

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

dist = common.calculate_distance_array(coordinates)/len(coordinates)
print("Data set average distance: ", dist)

print("Start clustering process")
clusterResult = clustering.make_clustering_2(coordinates, 500)
print("Cluster completed")

del coordinates, coordinates_df

print("Dump model")
clustering.dump_model(clusterResult.model, result_model)

print_header = True
write_mode = 'w'
for df in pd.read_csv(result_csv, chunksize=chunk_size, iterator=True, dtype={'POLYLINE': str}):

    df['POLYLINE'] = df['POLYLINE'].\
        apply(lambda x: [clustering.predict(clusterResult.model, point) for point in
                         polyline_to_coordinates(x, r'\((-?[0-9]{1,2}\.[0-9]+),\s(-?[0-9]{1,2}\.[0-9]+)\)')]).astype(str)

    df = explode(df, ['POLYLINE'])

    df.rename(columns={df.columns[3]: "gate" }, inplace=True)

    lastId = "";
    lastGate = ""
    count = 0;
    for row in df.itertuples():
        if df.at[row.Index, 'TRIP_ID'] == lastId:
            count += 15
            df.at[row.Index, 'TIMESTAMP'] = count
            if df.at[row.Index, 'gate'] != lastGate:
                lastGate = df.at[row.Index, 'gate']
                df.at[row.Index, 'TIMESTAMP'] = count
        else:
            count = df.at[row.Index, 'TIMESTAMP']
            lastGate = df.at[row.Index, 'gate']
            lastId = df.at[row.Index, 'TRIP_ID']

    df.drop_duplicates(['TRIP_ID','gate'], inplace=True)

    df.to_csv(result_csv, mode=write_mode, header=print_header, index=False)

    # Print CSV header only the first time
    print_header = False

    # Change write mode to append. In this way, if the file already exists, it is rewrited only the first time.
    write_mode = 'a'







