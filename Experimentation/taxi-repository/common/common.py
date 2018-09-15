import geopy.distance
import json


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


def to_json_file(file_path=str, json_data=str):
    """It writes json data to the required file path"""
    with open(file_path, 'w') as outfile:
        yield json.dump(json_data, outfile);