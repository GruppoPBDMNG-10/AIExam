import pandas as pd
from common import calculate_distance
import geopy.distance
import numpy as np
from sqlalchemy import create_engine


def polyline_to_coordinates(polyline):
    """It transforms the polyline column in coordinates array"""
    coordinates = [];
    temp = polyline.replace("[", "").replace("]", "");
    if temp:
        list = df['POLYLINE'][index].replace("[", "").replace("]", "").split(",");
        if len(list) > 1 & len(list) % 2 == 0:
            for i in range(0, len(list) - 1, 2):
                coordinates.append((float(list[i]),float(list[i+1])));
    return coordinates;


def drop_columns(df=pd.DataFrame, columsName=list):
    """Drop specified colums from the required dataframe"""
    return df.drop(columsName, axis=1);


csv_database = create_engine('sqlite:///taxi.db');
file = 'dataset/taxi_small.csv';
fileCleaned = 'dataset/taxi_cleaned.csv';
columns_to_remove = ["CALL_TYPE","ORIGIN_CALL","ORIGIN_STAND","DAY_TYPE","MISSING_DATA"];

chunksize = 100000;
dist = 0;
firstDf = True;

for df in pd.read_csv(file, chunksize=chunksize, iterator=True, dtype={'POLYLINE': str}):
    #Remove undesired colums from dataset
    df = drop_columns(df, columns_to_remove);

    for index in df.index:
        #Convert text to coordinates
        coordinates = polyline_to_coordinates(df.at[index, 'POLYLINE']);

        #If there is more than one point, calculate distance between point
        if len(coordinates) > 1:
            #Set coordinates in the right place
            df.at[index, 'POLYLINE'] = coordinates;

            #Calculate average distance between all point in the dataset
            first = coordinates[0];
            second = coordinates[1];
            dist += calculate_distance(first, second);
            first = second;
            if len(coordinates) > 2:
                for i in range(2,len(coordinates)):
                    second = coordinates[i];
                    dist += calculate_distance(first,second);
                    first = second;
            dist = dist / len(coordinates);

            #Write chunk to result csv file
            df.to_csv(fileCleaned, mode='a', header=True if index == 0 else False, index=False);
            firstDf = False;
        else:
            #If no coordinates are associated with the entry, remove it.
            print("Entry with index: ", index + 2, " removed.");
            df = df.drop([index]);

print("Dataset average distance: ", dist);

