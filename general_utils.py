import pandas as pd
from termcolor import colored

from datetime import datetime


def to_int_timestamp(timestamp_str):
    return int(datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%S.%fZ').timestamp() * 1000)


def to_str_timestamp(timestamp_int):
    dt_object = datetime.fromtimestamp(timestamp_int / 1000)
    return dt_object.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def get_data_info(data_df: pd.DataFrame, prefix: str = ""):
    """ columns: limb, side, timestamp, type, x, y, z, timestamp_ms, keys_tuple"""
    measurement_ids = data_df.measurementId.unique()
    print(colored("\n{} measurement ids: {}".format(prefix, measurement_ids), color="blue"))

    for measurement_id in measurement_ids:
        print("\nmeasurement {}".format(measurement_id))
        meas_df = data_df[data_df.measurementId == measurement_id]

        keys_tuples = data_df.keys_tuple.unique()
        print("keys_tuples:")
        for keys_tuple in keys_tuples:
            key_df = meas_df[meas_df.keys_tuple == keys_tuple]

            timestamp_ms_min = key_df.timestamp_ms.min()
            timestamp_ms_max = key_df.timestamp_ms.max()
            first_timestamp = datetime.fromtimestamp(timestamp_ms_min / 1000)
            last_timestamp = datetime.fromtimestamp(timestamp_ms_max / 1000)
            delta = last_timestamp - first_timestamp
            print("{}: from {} ({}) to {} ({}), delta (min): {:.2f}".format(keys_tuple, key_df.timestamp.min(),
                                                                            timestamp_ms_min,
                                                                            key_df.timestamp.max(),
                                                                            timestamp_ms_max,
                                                                            delta.total_seconds() / 60))
