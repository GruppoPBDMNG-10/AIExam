import numpy as np
import tensorflow as tf
import gc
import sys
import common.common as common
import itertools

# In [16]
tf.logging.set_verbosity(tf.logging.INFO)

trajectories_map = common.__load_df('./data/train_taxi.csv')

max_num_gates = np.max([len(x) for x in trajectories_map.values()])
min_num_gates = np.min([len(x) for x in trajectories_map.values()])
print('max trajectory length {}'.format(max_num_gates))
print('min trajectory length {}'.format(min_num_gates))

print('num of trajectories =', len(trajectories_map))
trajectories_map = dict((k, v) for k, v in trajectories_map.items() if len(v) >= 10)
print('num of reviews len [100:200] =', len(trajectories_map))

# In [17]
gc.collect()

# In [18]
# text = ' '.join(reviews[:8000])
num_trajectories = 2000

# Length of extracted character sequences# Length
maxlen = 10

# We sample a new sequence every `step` characters
step = 1

# This holds our extracted sequences
trajectories = []

# This holds the targets (the follow-up characters)
next_gates = []

for trajectory in trajectories_map.values():
    for i in range(0, len(trajectory) - maxlen, step):
        trajectories.append(trajectory[i: i + maxlen])
        next_gates.append(trajectory[i + maxlen])
print('Number of sequences:', len(trajectories))


# List of unique characters in the corpus
gates = sorted(list(set(itertools.chain.from_iterable(trajectories_map.values()))))
print('Unique gates:', len(gates))

# Dictionary mapping unique characters to their index in `gates`
gates_index = dict((gate, gates.index(gate)) for gate in gates)

# Next, one-hot encode the characters into binary arrays.
print('Vectorization...')
x = np.zeros((len(trajectories), maxlen, len(gates)), dtype=np.bool)
y = np.zeros((len(trajectories), len(gates)), dtype=np.bool)

for i, trajectory in enumerate(trajectories):
    for t, gate in enumerate(trajectory):
        x[i, t, gates_index[gate]] = 1
    y[i, gates_index[next_gates[i]]] = 1

# In [19]
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(256, activation='tanh',
                               input_shape=(maxlen, len(gates))))
model.add(tf.keras.layers.Dense(len(gates), activation='softmax'))

optimizer = tf.keras.optimizers.RMSprop()
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

model.summary()


# In [20]
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


# In [23]
for e in range(1, 30):
    print('epoch', e)
    model.fit(x, y, batch_size=512, epochs=1)

    # Select a text seed at random
    # i = random.randint(0, 6000)
    # select the same review each time
    if e % 2 == 0:
        generated_trajectory = trajectories[1][0:maxlen]

        print('--- Generating with seed:"', str(generated_trajectory))

        for temperature in [0.2, 0.5, 1.0]:
            print('------ temperature:', temperature)
            sys.stdout.write('SEED: ')
            sys.stdout.write(str(generated_trajectory))

            # We generate 400 characters
            for i in range(100):
                sampled = np.zeros((1, maxlen, len(gates)))
                for t, gate in enumerate(generated_trajectory):
                    sampled[0, t, gates_index[gate]] = 1.

                predict = model.predict(sampled, verbose=0)[0]
                next_index = sample(predict, temperature)
                next_gate = gates[next_index]

                generated_trajectory.append(next_gate)
                generated_trajectory = generated_trajectory[1:]

                sys.stdout.flush()
            print()
            sys.stdout.write('FINAL: ')
            sys.stdout.write(str(generated_trajectory))
            print()
        print()
