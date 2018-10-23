from experimentation.hmm import hmm
from pathlib import Path
import numpy as np
import json

RESULT_PATH = 'result/'
EXP_PATH = RESULT_PATH + 'experimentation/hmm_500/'
MODEL_PATH = EXP_PATH + 'model_hmm.pkl'
DATASET_PATH = RESULT_PATH + 'taxi_porto_gate.csv'
MODE = ''
MAX_TEST_LENGTH = 7000

dataset, gates = hmm.prepare_dataset_rapresentation(DATASET_PATH)
hmm_model_file = Path(MODEL_PATH)

print("Dataset loading completed")

test_data = hmm.retrieve_test_samples(dataset, max_sequence_length=150,
                                  max_test_data_length=MAX_TEST_LENGTH)

model = None

if hmm_model_file.is_file() & hmm_model_file.exists():
    print('Loading model from dump')
    model = hmm.load_dump(MODEL_PATH)
else:
    print('Creating model from scratch')
    model = hmm.build_model(test_data, components=len(gates))
    print("Model created:", model)
    hmm.dump(model, MODEL_PATH)

print("Start calculating score for each sample")
scores_dict = hmm.calcualate_scores_dict(dataset, model)

print('Saving scores dictionary')

with open(EXP_PATH + 'scores.json', 'w') as outfile:
    json.dump(scores_dict, outfile)

print("Start statistics calculation")
mean, variance, std, total = hmm.calculate_save_statistics(scores_dict, EXP_PATH + 'statistics.json')
print("Total:", total, ", Mean:", mean, ", Variance", variance, "Standard Derivation:", std)

anomalous = dict(
    (key, value) for key, value in scores_dict.items() if value < mean and (abs(abs(value) - abs(mean)) > std))
print(anomalous)
print("Anomalous lenght:", len(anomalous))
print("Saving anomalous sequences")
with open(EXP_PATH + 'anomalous.json', 'w') as outfile:
    json.dump(anomalous, outfile)
