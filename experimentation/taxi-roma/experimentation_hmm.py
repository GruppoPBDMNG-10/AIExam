from experimentation.hmm import hmm
from pathlib import Path
import numpy as np


MODEL_PATH = 'result/model_hmm.pkl'
DATASET_PATH = 'result/taxi_rome_gate.csv'

dataset, gates = hmm.prepare_dataset_rapresentation(DATASET_PATH)
hmm_model_file = Path(MODEL_PATH)

print("Dataset loading completed")

model = None

if hmm_model_file.is_file() & hmm_model_file.exists():
    print('Loading model from dump')
    model = hmm.load_dump(MODEL_PATH)
else:
    print('Creating model from scratch')
    model = hmm.build_model(dataset)
    print("Model created:", model)
    hmm.dump(model, MODEL_PATH)

print("Start calculating score for each sample")
logprob = dict((key, model.score(value, [len(value)])) for key, value in dataset.items())

print(logprob)
print("Mean", np.mean([value for value in logprob.values()]))
print("Variance", np.var([value for value in logprob.values()]))

