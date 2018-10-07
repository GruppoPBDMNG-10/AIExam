from experimentation.hmm import hmm
from pathlib import Path
import numpy as np
import json
import itertools

RESULT_PATH = 'result/'
EXP_PATH = RESULT_PATH + 'experimentation/hmm/'
MODEL_PATH = EXP_PATH + 'model_hmm.pkl'
DATASET_PATH = RESULT_PATH + 'taxi_porto_gate.csv'
MODE = ''
MAX_TEST_LENGTH = -1


def calcualate_scores_dict(dataset=dict, model=hmm.hmm.MultinomialHMM) -> dict:
    result = dict((key, model.score(value.toarray(), [len(value.toarray())]) / len(value.toarray())) for key, value in dataset.items())
    return result


def retrieve_test_samples(dataset=dict, min_sequence_length=3, max_sequence_length=-1, max_test_data_length=-1) -> dict:
    print("Dataset samples before filtering:", len(dataset.keys()))
    if max_sequence_length > 0:
        result = dict(
            (key, value) for key, value in dataset.items() if max_sequence_length >= len(value.toarray()) >= min_sequence_length)
    else:
        result = dict((key, value) for key, value in dataset.items() if len(value.toarray()) >= min_sequence_length)
    if max_test_data_length > 0 and len(result.keys()) > max_test_data_length:
        result = dict(itertools.islice(result.items(), max_test_data_length))
    print("Dataset samples after filtering:", len(result.keys()))
    return result


dataset, gates = hmm.prepare_dataset_rapresentation(DATASET_PATH)
hmm_model_file = Path(MODEL_PATH)

print("Dataset loading completed")

test_data = retrieve_test_samples(dataset, min_sequence_length=5, max_sequence_length=150,
                                  max_test_data_length=MAX_TEST_LENGTH)

model = None

if hmm_model_file.is_file() & hmm_model_file.exists():
    print('Loading model from dump')
    model = hmm.load_dump(MODEL_PATH)
else:
    print('Creating model from scratch')
    model = hmm.build_model(test_data)
    print("Model created:", model)
    hmm.dump(model, MODEL_PATH)

print("Start calculating score for each sample")
scores_dict = calcualate_scores_dict(dataset, model)

print('Saving scores dictionary')

with open(EXP_PATH + 'scores.json', 'w') as outfile:
    json.dump(scores_dict, outfile)

print("Start statistics calculation")
scores = [value for value in scores_dict.values()]
mean = np.mean(scores)
variance = np.var(scores)
std = np.std(scores)
total = np.sum(scores)

print("Total:", total, ", Mean:", mean, ", Variance", variance, "Standard Derivation:", std)

anomalous = dict(
    (key, value) for key, value in scores_dict.items() if value < mean and (abs(abs(value) - abs(mean)) > std))
print(anomalous)
print("Anomalous lenght:", len(anomalous))
print("Saving anomalous sequences")
with open(EXP_PATH + 'anomalous.json', 'w') as outfile:
    json.dump(anomalous, outfile)
