from experimentation.woman import woman_input_preprocessing as process


DATASET = 'result/go_track_trackpoints_gate.csv'
RESULT = 'result/go_track_woman.txt'


process.to_woman_input(DATASET, RESULT, 'go_track')
