import geopy.distance


def calculate_distance(first, second):
    """It calculates the distance between two points. The distance is in meters."""
    return geopy.distance.vincenty((first[0],first[1]),(second[0],second[1])).meters;