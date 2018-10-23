import numpy as np
import tensorflow as tf
import itertools
import json
import math
import matplotlib.pyplot as plt


def retrieve_test_samples(data_set: dict,
                          min_sequence_length: int=3,
                          max_sequence_length: int=-1,
                          max_test_data_length: int=-1) -> dict:
    print("Data-set samples before filtering:", len(data_set.keys()))
    if max_sequence_length > 0:
        result = dict(
            (key, value) for key, value in data_set.items() if max_sequence_length >= len(value) >= min_sequence_length)
    else:
        result = dict((key, value) for key, value in data_set.items() if len(value) >= min_sequence_length)
    if len(result.keys()) > max_test_data_length > 0:
        result = dict(itertools.islice(result.items(), max_test_data_length))
    print("Data-set samples after filtering:", len(result.keys()))
    return result


def process_data_set(trajectories_map: dict, length: int) -> (list, list, dict, list):
    # This holds our extracted sequences
    trajectories = []

    # This holds the targets (the follow-up characters)
    next_gates = []

    for trajectory in trajectories_map.values():
        for i in range(0, len(trajectory) - length):
            trajectories.append(trajectory[i: i + length])
            next_gates.append(trajectory[i + length])
    print('Extracted trajectory:', len(trajectories))

    # List of unique gates in the data-set
    gates = sorted(list(set(itertools.chain.from_iterable(trajectories_map.values()))))
    print('Unique gates:', len(gates))

    # Dictionary mapping unique gate to their index in `gates`
    gates_index = dict((gate, gates.index(gate)) for gate in gates)

    return trajectories, next_gates, gates_index, gates


def one_hot_encoding(samples: list, classifications: list, labels_index: dict, labels: list) -> (list, list):
    x = np.zeros((len(samples), len(samples[0]), len(labels)), dtype=np.bool)
    y = np.zeros((len(samples), len(labels)), dtype=np.bool)

    for i, sample in enumerate(samples):
        for t, observation in enumerate(sample):
            x[i, t, labels_index[observation]] = 1
        y[i, labels_index[classifications[i]]] = 1

    return x, y


def build_model(x: list, y: list, fold: int=1, epochs: int=1, csv_logger: tf.keras.callbacks.CSVLogger=object)\
        -> (tf.keras.models.Sequential, object):
    for e in range(0, fold):
        num_samples_fold = len(x) // fold
        tf.logging.set_verbosity(tf.logging.INFO)
        rnn = tf.keras.models.Sequential()
        rnn.add(tf.keras.layers.LSTM(256, activation='tanh', input_shape=(len(x[0]), len(y[0]))))
        rnn.add(tf.keras.layers.Dense(len(y[0]), activation='softmax'))
        optimizer = tf.keras.optimizers.RMSprop()
        rnn.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        rnn.summary()

        history = rnn.fit(x[num_samples_fold*e:], y[num_samples_fold*e:],
                batch_size=512,
                epochs=epochs,
                validation_data=(x[:num_samples_fold*(fold-e)], y[:num_samples_fold*(fold-e)]),
                callbacks=[csv_logger])

        return rnn, history


def sample(predictions, temp=1.0):
    predictions = np.asarray(predictions).astype('float64')
    predictions = np.log(predictions) / temp
    exp_predictions = np.exp(predictions)
    predictions = exp_predictions / np.sum(exp_predictions)
    probabilities = np.random.multinomial(1, predictions, 1)
    return np.argmax(probabilities)


def probability_gap(predictions, next_gate_index: int):
    predictions = np.asarray(predictions).astype('float64')
    max_gate_index = np.argmax(predictions)
    return predictions[max_gate_index] - predictions[next_gate_index]


def calculate_scores_dict(trajectories_map: dict, model: tf.keras.models.Sequential, gates_index: dict, gates: list, length: int) -> list:
    result = dict()
    for key, value in trajectories_map.items():
        num_followers = len(value) - length
        # sequence_probability = 0
        sequence_probability = 1
        if num_followers > 0:
            for i in range(0, num_followers):
                samples = []
                classification = []
                samples.append(value[i: i + length])
                classification.append(value[i + length])
                items = one_hot_encoding(samples, classification, gates_index, gates)
                # sequence_probability += probability_gap((model.predict(items[0], verbose=0)[0]), gates_index[value[i + length]])
                sequence_probability *= model.predict(items[0], verbose=0)[0][gates_index[value[i + length]]]
                # result[key] = (1 - (sequence_probability / num_followers))
            result[key] = (math.log(sequence_probability) / num_followers)
    return result


def calculate_save_statistics(scores_dict=dict, out_file=str):
    scores = [value for value in scores_dict.values()]
    mean = np.mean(scores)
    variance = np.var(scores)
    std = np.std(scores)
    total = np.sum(scores)
    result = {'mean': mean, 'variance': variance, 'std': std, 'total:': total}
    with open(out_file, 'w') as outfile:
        json.dump(result, outfile)

    return mean, variance, std, total

