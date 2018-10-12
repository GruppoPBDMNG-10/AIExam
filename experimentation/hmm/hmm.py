import numpy as np
from hmmlearn import hmm
import pandas as pd
import itertools
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
    for df in pd.read_csv(file, chunksize=chunk_size, iterator=True,
                          dtype={'TRIP_ID': str, 'DRIVER_ID': int, 'TIMESTAMP': int, 'GATE': int}):
        df.apply(lambda x: __process_raw(x, sequence_map), axis=1)
        del df

    return sequence_map


def prepare_dataset_rapresentation(file) -> (dict, list):
    """Load from the specified file the dataset. The result is a dict having as key the TRIP_ID and as values the
    one-hot representation of each sequence."""
    sequences_map = __load_df(file)

    gates = sorted(list(set(itertools.chain.from_iterable(sequences_map.values()))))
    gates_index = dict((gate, gates.index(gate)) for gate in gates)

    print("Start dictionary representation")

    result = dict()

    for key, sequence in sequences_map.items():
        representation = np.zeros((len(sequence), len(gates)), dtype=np.int8)
        for i, gate in enumerate(sequence):
            representation[i, gates_index[gate]] = 1
        result[key] = representation

    return result, gates


def build_model(dataset=dict, components=2, iter=1000, tol=0.01) -> hmm.MultinomialHMM:
    """It builds a MultinomialHMM from the dataset. The input dataset expected has the same form obtained from
    prepare_dataset_rapresentation method."""

    vector = []
    lengths = []
    elem = None

    for sequence in dataset.values():
        vector.append(sequence)
        lengths.append(len(sequence))

    vector = np.concatenate(vector)

    print("")
    print(np.info(vector))
    print("n_samples:", sum(lengths))

    print("Start model training")
    model = hmm.MultinomialHMM(n_components=components, random_state=42, n_iter=iter, tol=tol)
    model = model.fit(vector, lengths=lengths)

    return model


def dump(model, filename):
    joblib.dump(model, filename)


def load_dump(filename) -> hmm.MultinomialHMM:
    return joblib.load(filename)


def calcualate_scores_dict(dataset=dict, model=hmm.MultinomialHMM) -> dict:
    result = dict((str(key), model.score(value, [len(value)]) / len(value)) for key, value in dataset.items())
    return result


def retrieve_test_samples(dataset=dict, min_sequence_length=3, max_sequence_length=-1, max_test_data_length=-1) -> dict:
    print("Dataset samples before filtering:", len(dataset.keys()))
    if max_sequence_length > 0:
        result = dict(
            (key, value) for key, value in dataset.items() if max_sequence_length >= len(value) >= min_sequence_length)
    else:
        result = dict((key, value) for key, value in dataset.items() if len(value) >= min_sequence_length)
    if max_test_data_length > 0 and len(result.keys()) > max_test_data_length:
        result = dict(itertools.islice(result.items(), max_test_data_length))
    print("Dataset samples after filtering:", len(result.keys()))
    return result