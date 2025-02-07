import pandas as pd

import experimentation.clustering.clustering as clustering
import experimentation.common.common as common

result_folder = 'result/';
file = 'dataset/go_track_trackspoints.csv';
result_csv = result_folder + 'go_track_trackpoints_cleaned.csv';
result_gate = result_folder + 'go_track_trackpoints_gate.csv'
result_coordinates = result_folder + 'coordinates.csv';
result_model = result_folder + 'model.pkl'

chunksize = 500000;
dist = 0;
coordinates = [];
print_header = True;
write_mode = 'w';

for df in pd.read_csv(file, chunksize=chunksize, iterator=True, usecols=["latitude","longitude","track_id","time"], dtype={'track_id': int, 'latitude': float, 'longitude': float}, parse_dates=['time'], infer_datetime_format=True):
    #Convert datetime to unix timestamp
    df['time'] = df['time'].apply(lambda x: common.datetime_to_unix(x))

    #Get coordinates as touple
    for latitude, longitude in zip(df['latitude'], df['longitude']):
        coordinates.append((latitude, longitude))

    df.to_csv(result_csv, mode=write_mode, header=print_header, index=False);

    # Print CSV header only the first time
    print_header = False;

    # Change write mode to append. In this way, if the file already exists, it is rewrited only the first time.
    write_mode = 'a';

dist = common.calculate_distance_array(coordinates)/len(coordinates);
print("Dataset lenght: ", len(coordinates))
print("Dataset average distance: ", dist);

print("Start clustering process")
clusterResult = clustering.make_clustering_2(coordinates, 500)
print("Cluster completed")


print("Start Dump model")
clustering.dump_model(clusterResult.model, result_model)
print("End Dump model")

print_header = True
write_mode = 'w'

gates = [];
for df in pd.read_csv(result_csv, chunksize=chunksize, iterator=True, dtype={'track_id': int, 'latitude': float, 'longitude': float, 'time': int}):

    df = pd.DataFrame(df)

    for latitude, longitude in zip(df['latitude'], df['longitude']):
        gates.append(clustering.predict(clusterResult.model, [latitude, longitude]))

    df['GATE'] = gates

    df.drop(['latitude', 'longitude'], axis=1, inplace=True)
    df.drop_duplicates(['track_id','GATE'], inplace=True)

    df['DRIVER_ID'] = df['track_id']

    df.rename(columns={df.columns[0]: "TRIP_ID", df.columns[1]: "TIMESTAMP"}, inplace=True)
    df = df.reindex(columns=['TRIP_ID', 'DRIVER_ID'] + df.columns[1:-1].tolist())

    df.to_csv(result_gate, mode=write_mode, header=print_header, index=False)

    # Print CSV header only the first time
    print_header = False

    # Change write mode to append. In this way, if the file already exists, it is rewrited only the first time.
    write_mode = 'a'

