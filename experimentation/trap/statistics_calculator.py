import experimentation.statistics.statistics as statistics

intersection = statistics.find_matches_from_file('result/experimentation/hmm/anomalous.json', 'result/experimentation/hmm_old/anomalous.json')

print(len(intersection))