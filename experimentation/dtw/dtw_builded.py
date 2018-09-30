from dtaidistance import dtw
from dtaidistance import clustering
import numpy as np
import itertools
import pandas as pd
from sklearn.externals import joblib


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

file = "./taxi_rome_gate.csv"
sequences_map = __load_df(file)
gates = sorted(list(set(itertools.chain.from_iterable(sequences_map.values()))))
gates_index = dict((gate, gates.index(gate)) for gate in gates)

max_len = len(max(sequences_map.values(), key=len))

print("Max sequence length:", max_len)

vector = np.zeros((len(sequences_map.values()), max_len, len(gates)), dtype=np.double)

for i, sequence in enumerate(sequences_map.values()):
    for t, gate in enumerate(sequence):
        vector[i, t, gates_index[gate]] = 1

nsamples, nx, ny = vector.shape
d2_train_dataset = vector.reshape((nsamples, nx * ny))

print(d2_train_dataset)

#series = np.matrix([
#    [273, 273, 273, 273, 65, 65, 387, 387, 387, 391, 391, 391, 387, 387, 387, 65, 65, 273, 273, 273, 273],
#    [466, 466, 356, 425, 425, 201, 413, 67, 67, 67, 381, 420, 297, 297, 364, 290, 271, 30, 372, 372, 304],
#    [123, 123, 262, 107, 107, 107, 107, 107, 107, 370, 377, 377, 473, 473, 346, 376, 376, 376, 376, 402, 428],
#    [461, 461, 461, 91, 91, 91, 91, 148, 148, 148, 461, 0, 303, 303, 303, 303, 303, 303, 339, 339, 339],
#    [442, 442, 442, 442, 442, 442, 68, 68, 68, 68, 68, 339, 339, 303, 303, 0, 0, 0, 91, 91, 148],
#    [272, 272, 272, 449, 449, 487, 487, 170, 170, 170, 495, 495, 359, 197, 376, 215, 215, 215, 215, 215, 473],
#    [269, 384, 287, 287, 391, 111, 111, 111, 363, 363, 85, 85, 421, 421, 421, 421, 365, 365, 365, 365, 434]], dtype=np.double)

#ds = dtw.distance_matrix_fast(series)
#print(ds)


# Custom Hierarchical clustering
model1 = clustering.Hierarchical(dtw.distance_matrix_fast, {})
# Augment Hierarchical object to keep track of the full tree
model2 = clustering.HierarchicalTree(model1)
# SciPy linkage clustering
model3 = clustering.LinkageTree(dtw.distance_matrix_fast, {})

print("Building model...")
cluster_idx = model3.fit(d2_train_dataset)
print("Building complete")

joblib.dump(model3, "./model3.dtw")



model3.plot("myplot.png")