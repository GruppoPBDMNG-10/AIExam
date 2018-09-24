import numpy as np
from hmmlearn import hmm
import pandas as pd


def __process_raw(df, sequence_map=dict):
    sequence = sequence_map.get(df['TRIP_ID'])
    if not sequence:
        sequence = []
    sequence.append([df['GATE']])
    sequence_map[df['TRIP_ID']] = sequence


def __load_df(file) -> dict:
    chunk_size = 500000
    print("Start loading file")
    sequence_map = {}
    for df in pd.read_csv(file, chunksize=chunk_size, iterator=True, dtype={'TRIP_ID': str, 'DRIVER_ID': int, 'TIMESTAMP': int, 'GATE': int}):
        df.apply(lambda x: __process_raw(x, sequence_map), axis=1)

    return sequence_map



def create_model(file):
    sequence_map = __load_df(file)
    print(sequence_map)

    x = []
    lengths = []

    for values in sequence_map.values():
        x.append(values)
        lengths.append(len(values))

    x = np.concatenate(x)
    model = hmm.GaussianHMM(n_components=500, covariance_type="full")

    print(model.fit(x, lengths))
    print(model.predict((sequence_map.get('1')[0:11])))
    print(sequence_map.get('1')[11:22])



create_model('../taxi-roma/result/taxi_rome_gate.csv')