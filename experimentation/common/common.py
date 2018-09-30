import geopy.distance
import json
import datetime
import time
import re
import pandas as pd


REGEX_PATTERN_MAP = {}


def calculate_distance(first, second):
    """It calculates the distance between two points. The distance is in meters."""
    return geopy.distance.vincenty((first[0],first[1]),(second[0],second[1])).meters;


def calculate_distance_array(elements=list):
    """It calculates the distance between two points. The distance is in meters."""
    dist = 0;
    if len(elements) > 1:
        dist += calculate_distance(elements[0], elements[1]);
        for i in range(2, len(elements) - 2):
            dist += calculate_distance(elements[i], elements[i+1]);
    return dist;


def calculate_average_distance(elements=list):
    yield calculate_distance_array(elements)/len(elements)


def to_json_file(file_path=str, json_data=str):
    """It writes json data to the required file path"""
    with open(file_path, 'w') as outfile:
        yield json.dump(json_data, outfile);


def datetime_to_unix(date=datetime.datetime):
    """It converts a datetime object in unix timestamp"""
    return int(time.mktime(date.timetuple()))


def polyline_to_coordinates(polyline, regex):
    """It transforms the polyline column in coordinates array"""
    pattern = REGEX_PATTERN_MAP.get(regex)

    if not pattern:
        pattern = re.compile(regex)
        REGEX_PATTERN_MAP[regex] = pattern

    coordinates_array = []
    matches = pattern.finditer(polyline)
    # matches = re.finditer(r'\[(-?[0-9]{1,2}\.[0-9]+),\s(-?[0-9]{1,2}\.[0-9]+)\]', polyline)
    for match in matches:
        coordinates_array.append((float(match.group(1)), float(match.group(2))))

    return coordinates_array


def __process_raw(df, sequence_map=dict):
    sequence = sequence_map.get(df['TRIP_ID'])
    if not sequence:
        sequence = []
    sequence.append(df['GATE'])
    sequence_map[df['TRIP_ID']] = sequence


def __load_df(file) -> dict:
    chunk_size = 500000
    print("Start loading file")
    sequence_map = {}
    for df in pd.read_csv(file, chunksize=chunk_size, iterator=True, dtype={'TRIP_ID': str, 'DRIVER_ID': int, 'TIMESTAMP': int, 'GATE': int}):
        df.apply(lambda x: __process_raw(x, sequence_map), axis=1)

    return sequence_map
