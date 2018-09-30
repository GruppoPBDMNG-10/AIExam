import numpy as np
from hmmlearn import hmm
import pandas as pd
import itertools


def __process_raw(df, sequence_map=dict):
    sequence = sequence_map.get(df['TRIP_ID'])
    if not sequence:
        sequence = []
    sequence.append(df['GATE'])
    sequence_map[df['TRIP_ID']] = sequence


def __load_df(file) -> dict:
    chunk_size = 500000
    print("Start loading file")
    sequence_map = {}
    for df in pd.read_csv(file, chunksize=chunk_size, iterator=True, dtype={'TRIP_ID': str, 'DRIVER_ID': int, 'TIMESTAMP': int, 'GATE': int}):
        df.apply(lambda x: __process_raw(x, sequence_map), axis=1)

    return sequence_map


def build_model(file) -> hmm.MultinomialHMM:
    sequences_map = __load_df(file)
    gates = sorted(list(set(itertools.chain.from_iterable(sequences_map.values()))))
    gates_index = dict((gate, gates.index(gate)) for gate in gates)

    max_len = len(max(sequences_map.values(), key=len))

    print("Max sequence length:", max_len)

    vector = np.zeros((len(sequences_map.values()), max_len, len(gates)), dtype=np.short)

    for i, sequence in enumerate(sequences_map.values()):
        for t, gate in enumerate(sequence):
            vector[i, t, gates_index[gate]] = 1

    n_samples, nx, ny = vector.shape
    d2_train_dataset = vector.reshape((n_samples * nx * ny), 1)

    lengths = [max_len*len(gates) for i in range(len(sequences_map.values()))]

    model_to_use = None
    component_to_use = 2
    best_score = None
    for i in range(component_to_use, len(gates) + 1):
        print("Start training")
        model = hmm.MultinomialHMM(n_components=i, random_state=42, n_iter=1000, tol=0.001)
        model = model.fit(d2_train_dataset, lengths=lengths)
        score = model.score(d2_train_dataset, lengths=lengths)
        if model_to_use is None or score > best_score:
            model_to_use = model
            best_score = score
            component_to_use = i
        print("States:", i, "Score:", score)

    print("Model to use:", model_to_use)
    return model_to_use


build_model('../taxi-roma/result/taxi_rome_gate.csv')