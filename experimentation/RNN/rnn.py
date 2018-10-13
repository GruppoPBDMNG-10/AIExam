import numpy as np
import tensorflow as tf
import gc
import sys
import common.common as common
import itertools
import random

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-d", "--data-set", dest="data_set", default='./data/train_taxi.csv',
                    help="data-set file path", metavar="DATA-SET")
parser.add_argument("-m", "--model", dest="model", default='./result/model_2.rnn',
                    help="Model file store path", metavar="MODEL")
parser.add_argument("--len", "--trajectory_length", dest="trajectory_length", default=4,
                    help="Minimum Length of trajectory", metavar="MIN_LEN")
parser.add_argument("-e", "--epochs", dest="epochs", default=1000,
                    help="Epoch for each iteration", metavar="EPOCHS")

args = parser.parse_args()
print(args)

# Data-set file path
data_set_file_path = args.__dict__['data_set']

# Model file store path
model_file_path = args.__dict__['model']

# Length of trajectory to extract
trajectory_length = args.__dict__['trajectory_length']

# Number of epoch for each iteration
epochs = args.__dict__['epochs']

# In [16]
tf.logging.set_verbosity(tf.logging.INFO)

trajectories_map = common.__load_df(data_set_file_path)

max_num_gates = np.max([len(x) for x in trajectories_map.values()])
min_num_gates = np.min([len(x) for x in trajectories_map.values()])
print('max trajectory length {}'.format(max_num_gates))
print('min trajectory length {}'.format(min_num_gates))

print('num of trajectories =', len(trajectories_map))
trajectories_map = dict((k, v) for k, v in trajectories_map.items() if len(v) >= trajectory_length)
print('num of reviews len [min_trajectory_length:] =', len(trajectories_map))
print('trajectories_map', trajectories_map)

# In [17]
gc.collect()

# This holds our extracted sequences
trajectories = []

# This holds the targets (the follow-up characters)
next_gates = []

for trajectory in trajectories_map.values():
    for i in range(0, len(trajectory) - trajectory_length):
        trajectories.append(trajectory[i: i + trajectory_length])
        next_gates.append(trajectory[i + trajectory_length])
print('Extracted trajectory:', len(trajectories))

# List of unique gates in the data-set
gates = sorted(list(set(itertools.chain.from_iterable(trajectories_map.values()))))
print('Unique gates:', len(gates))

# Dictionary mapping unique gate to their index in `gates`
gates_index = dict((gate, gates.index(gate)) for gate in gates)

# Next, one-hot encode the characters into binary arrays.
print('Vectorization...')
x = np.zeros((len(trajectories), trajectory_length, len(gates)), dtype=np.bool)
y = np.zeros((len(trajectories), len(gates)), dtype=np.bool)

for i, trajectory in enumerate(trajectories):
    for t, gate in enumerate(trajectory):
        x[i, t, gates_index[gate]] = 1
    y[i, gates_index[next_gates[i]]] = 1

# In [19]
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(256, activation='tanh', input_shape=(trajectory_length, len(gates))))
model.add(tf.keras.layers.Dense(len(gates), activation='softmax'))
optimizer = tf.keras.optimizers.RMSprop()
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.summary()


# In [20]
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


num_folds = 1
num_samples_fold = len(y) // num_folds

for e in range(0, num_folds):
    print('epoch', e)

    model.fit(x[num_samples_fold*e:], y[num_samples_fold*e:],
              batch_size=512,
              epochs=epochs,
              validation_data=(x[:num_samples_fold*(num_folds-e)], y[:num_samples_fold*(num_folds-e)]))
    model.save(model_file_path)

    # Select a random trajectory
    i = random.randint(0, len(trajectories))
    generated_trajectory = trajectories[i][0:trajectory_length]

    print('--- Generating with seed:"', str(generated_trajectory))

    for temperature in [0.2, 0.5, 1.0]:
        print('------ temperature:', temperature)
        sys.stdout.write('SEED: ')
        sys.stdout.write(str(generated_trajectory))
        print()
        sys.stdout.write('PREDICTION: ')

        # We generate 4 gates
        for i in range(4):
            sampled = np.zeros((1, trajectory_length, len(gates)))
            for t, gate in enumerate(generated_trajectory):
                sampled[0, t, gates_index[gate]] = 1.

            predict = model.predict(sampled, verbose=0)[0]
            next_index = sample(predict, temperature)
            next_gate = gates[next_index]

            generated_trajectory.append(next_gate)
            generated_trajectory = generated_trajectory[1:]

            sys.stdout.write(str(next_gate))
            sys.stdout.write(', ')
            sys.stdout.flush()
        print()
        sys.stdout.write('FINAL: ')
        sys.stdout.write(str(generated_trajectory))
        print()
    print()

