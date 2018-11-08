import experimentation.statistics.statistics as statistics

intersection = statistics.find_matches_from_file('result/experimentation/hmm/anomalous.json', 'result/experimentation/rnn/anomalous.json')

print(len(intersection))