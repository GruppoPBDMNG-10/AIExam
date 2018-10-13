from experimentation.woman import woman_input_preprocessing as process


DATASET = 'result/taxi_rome_gate.csv'
RESULT = 'result/taxi_rome_woman.txt'


process.to_woman_input(DATASET, RESULT, 'taxi_rome')