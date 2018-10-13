from experimentation.woman import woman_input_preprocessing as process


DATASET = 'result/taxi_porto_gate.csv'
RESULT = 'result/taxi_porto_woman_small.txt'


process.to_woman_input(DATASET, RESULT, 'taxi_porto', min_sequence_length=5, max_sequence_length=150, max_samples_length=175114)
