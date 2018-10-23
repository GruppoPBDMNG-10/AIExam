import json


def find_matches(anomalous_first=dict, anomalous_second=dict):
    if len(anomalous_first.keys()) < len(anomalous_second.keys()):
        return [key for key in anomalous_first.keys() if anomalous_second.get(key) is not None]
    return [key for key in anomalous_second.keys() if anomalous_first.get(key) is not None]


def find_matches_from_file(anomalous_first_file=str, anomalous_second_file=str):
    anomalous_first = dict()
    anomalous_second = dict()
    with open(anomalous_first_file) as f:
        anomalous_first = json.load(f)
    with open(anomalous_second_file) as f:
        anomalous_second = json.load(f)
    return find_matches(anomalous_first, anomalous_second)
