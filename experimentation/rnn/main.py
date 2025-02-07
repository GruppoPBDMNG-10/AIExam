import sys
import random
import experimentation.common as common
import experimentation.RNN.rnn as rnn
import numpy as np
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument(
    "-d",
    "--data-set",
    dest="data_set",
    default='../taxi-roma/data/taxi_rome_gate_light.csv',
    help="data-set file path",
    metavar="DATA-SET")

parser.add_argument(
    "-s",
    "--samples",
    dest="samples",
    default=0,
    help="Number of samples to use",
    metavar="SAMPLES")

parser.add_argument(
    "-m",
    "--model",
    dest="model",
    default='../taxi-roma/result/experimentation/rnn/model_light.rnn',
    help="Model file store path",
    metavar="MODEL")

parser.add_argument(
    "--len",
    "--trajectory_length",
    dest="trajectory_length",
    default=4,
    help="Minimum Length of trajectory",
    metavar="MIN_LEN")

parser.add_argument(
    "-e",
    "--epochs",
    dest="epochs",
    default=100,
    help="Epoch for each iteration",
    metavar="EPOCHS")

args = parser.parse_args()
print(args)

data_set_file_path = args.__dict__['data_set']
model_file_path = args.__dict__['model']
num_samples = args.__dict__['samples']
trajectory_length = args.__dict__['trajectory_length']
epochs = args.__dict__['epochs']

trajectories_map = common.__load_df(data_set_file_path)

print('max trajectory length {}'.format(np.max([len(x) for x in trajectories_map.values()])))
print('min trajectory length {}'.format(np.min([len(x) for x in trajectories_map.values()])))

trajectories_map = \
    rnn.retrieve_test_samples(trajectories_map,
                              min_sequence_length=trajectory_length,
                              max_test_data_length=(len(trajectories_map) if num_samples == 0 else num_samples))
print('num of trajectories =', len(trajectories_map))

trajectories, next_gates, gates_index, gates = rnn.process_data_set(trajectories_map, trajectory_length)

# Next, one-hot encode the characters into binary arrays.
print('Vectorization...')
x, y = rnn.one_hot_encoding(trajectories, next_gates, gates_index, gates)

print('Building the model...')
model = rnn.build_model(x, y, epochs=epochs)
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
        next_index = rnn.sample(predict, temperature)
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


