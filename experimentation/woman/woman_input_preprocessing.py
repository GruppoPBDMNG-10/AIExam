import pandas as pd
from datetime import datetime
from experimentation.woman.activity_type import ActivityType
import itertools


__CHUNK_SIZE = 500000


def __process_raw(df, sequence_map=dict):
    key = str(df['TRIP_ID']) + '_' + str(df['DRIVER_ID'])
    sequence = sequence_map.get(key)
    if not sequence:
        sequence = []
    sequence.append((df['TIMESTAMP'], 'act_' + str(df['GATE'])))
    sequence_map[key] = sequence


def __load_df(file) -> dict:
    chunk_size = 500000
    print("Start loading file")
    sequence_map = {}
    for df in pd.read_csv(file, chunksize=chunk_size, iterator=True,
                          dtype={'TRIP_ID': str, 'DRIVER_ID': int, 'TIMESTAMP': int, 'GATE': int}):
        df.apply(lambda x: __process_raw(x, sequence_map), axis=1)
        del df

    return sequence_map


def __format_entry(timestamp=str, type=ActivityType, workflow_name=str, process_name=str, activity=str, activity_counter=int):
    return 'entry(' + ','.join([timestamp, type.value, workflow_name, workflow_name + '_' + process_name, activity, str(activity_counter)]) + ').\n'


def __activity_counter(activity=str, activity_map=dict):
    count = activity_map.get(activity)
    if not count:
        count = 0
    count += 1
    activity_map[activity] = count
    return count


def __filter_data(sequence_map=dict, min_sequence_length=int, max_sequence_length=int, max_samples_length=int):
    print("Dataset samples before filtering:", len(sequence_map.keys()))
    result = dict((key, value) for key, value in sequence_map.items() if min_sequence_length <= len(value) <= max_sequence_length)

    if len(result.keys()) > max_samples_length > 0:
        result = dict(itertools.islice(result.items(), max_samples_length))

    print("Dataset samples after filtering:", len(result.keys()))
    return result


def to_woman_input(file, result_file, workflow_name=str, min_sequence_length=-1, max_sequence_length=1000000, max_samples_length=-1):
    sequence_map = __filter_data(__load_df(file), min_sequence_length, max_sequence_length, max_samples_length)
    write_mode = 'w'
    with open(result_file, write_mode) as out:
        for key, value in sequence_map.items():
            timestamp = None
            last_item = None
            last_counter = None
            activity_counter = dict()
            for couple in value:
                timestamp = datetime.utcfromtimestamp(couple[0]).strftime('%Y%m%d%H%M%S')
                if not last_item:
                    # First time, print also BEGIN_PROCESS
                    out.write(__format_entry(timestamp, ActivityType.BEGIN_PROCESS, workflow_name, key, 'start', 0))
                else:
                    # Print END_ACTIVITY
                    out.write(__format_entry(timestamp, ActivityType.END_ACTIVITY, workflow_name, key, last_item, last_counter))

                last_item = couple[1]
                last_counter = __activity_counter(last_item, activity_counter)
                # Print START_ACTIVITY
                out.write(__format_entry(timestamp, ActivityType.BEGIN_ACTIVITY, workflow_name, key, last_item, last_counter))

            # Print END_ACTIVITY for last item
            out.write(__format_entry(timestamp, ActivityType.END_ACTIVITY, workflow_name, key, last_item, last_counter))

            # Print END_PROCESS
            out.write(__format_entry(timestamp, ActivityType.END_PROCESS, workflow_name, key, 'stop', 1))

    print("Created file", result_file)
