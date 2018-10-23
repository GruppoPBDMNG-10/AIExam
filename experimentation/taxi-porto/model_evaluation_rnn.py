import sys
import random
import experimentation.common.common as common
import experimentation.rnn.rnn as rnn
import numpy as np
import tensorflow as tf
from argparse import ArgumentParser
import json
import matplotlib.pyplot as plt


parser = ArgumentParser()
parser.add_argument(
    "-d",
    "--data-set",
    dest="data_set",
    default='./result/taxi_porto_gate.csv',
    help="data-set file path",
    metavar="DATA-SET")

parser.add_argument(
    "-s",
    "--samples",
    dest="samples",
    default=7000,
    help="Number of samples to use",
    metavar="SAMPLES")

parser.add_argument(
    "-m",
    "--model",
    dest="model",
    default='./result/experimentation/rnn/model_ev.rnn',
    help="Model file store path",
    metavar="MODEL")

parser.add_argument(
    "-bm",
    "--build-model",
    dest="build_model",
    default=True,
    help="Build model",
    metavar="BUILD_MODEL")

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
build_model = args.__dict__['build_model']

trajectories_map = common.__load_df(data_set_file_path)

base_plot = 220

dict_color = dict()
dict_color[3] = 'r'
dict_color[4] = 'b'
dict_color[5] = 'y'

for i, epochs in enumerate([100, 150, 200]):
    plt.figure(1)
    plt.subplot(211)
    plt.title('Training')
    plt.xlabel('')
    plt.ylabel('Accuracy')

    plt.subplot(212)
    plt.title('')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    for trajectory_length in [3, 4, 5]:
        print('max trajectory length {}'.format(np.max([len(x) for x in trajectories_map.values()])))
        print('min trajectory length {}'.format(np.min([len(x) for x in trajectories_map.values()])))

        trajectories_map = \
            rnn.retrieve_test_samples(trajectories_map,
                                      min_sequence_length=trajectory_length,
                                      max_test_data_length=(len(trajectories_map) if num_samples == 0 else num_samples))
        print('num of trajectories =', len(trajectories_map))
        print('trajectories_map', trajectories_map)

        trajectories, next_gates, gates_index, gates = rnn.process_data_set(trajectories_map, trajectory_length)

        if build_model:
            # Next, one-hot encode the characters into binary arrays.
            print('Vectorization...')
            x, y = rnn.one_hot_encoding(trajectories, next_gates, gates_index, gates)

            print('Building the model...')
            logger = tf.keras.callbacks.CSVLogger(model_file_path+'_ep'+str(epochs)+'_neurons'+str(trajectory_length)+'.csv')
            model, history = rnn.build_model(x, y, epochs=epochs, csv_logger=logger)
            model.save(model_file_path+'_ep'+str(epochs)+'_neurons'+str(trajectory_length))

            plot_epochs = range(1, epochs + 1)

            plt.subplot(211)
            plt.plot(plot_epochs, history.history['acc'], dict_color[trajectory_length],
                     label=('Accuracy ' + str(trajectory_length) + ' neurons'))
            plt.legend()

            plt.subplot(212)
            plt.plot(plot_epochs, history.history['loss'], dict_color[trajectory_length],
                     label=('Loss ' + str(trajectory_length) + ' neurons'))
            plt.legend()

    plt.savefig(model_file_path+'_'+str(i)+'.png')
    plt.clf()

