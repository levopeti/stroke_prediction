import numpy as np

from termcolor import colored
from datetime import datetime
from typing import Union


def to_int_timestamp(timestamp_str: str) -> int:
    """ timestamp in microseconds"""
    return int(datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%S.%fZ').timestamp() * 1000)


def to_str_timestamp(timestamp: Union[int, datetime]):
    if isinstance(timestamp, (int, np.int64)):
        dt_object = datetime.fromtimestamp(timestamp / 1000)
    else:
        dt_object = timestamp
    return dt_object.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def hour_to_millisec(hour: Union[int, float]) -> int:
    return int(hour * 60 * 60 * 1000)


def min_to_millisec(minute: Union[int, float]) -> int:
    return int(minute * 60 * 1000)


def get_data_info(data_dict: dict, prefix: str = ""):
    """ columns: limb, side, timestamp, type, x, y, z, timestamp_ms, keys_tuple"""
    measurement_ids = sorted(data_dict.keys())
    print(colored("\n{} measurement ids: {}".format(prefix, measurement_ids), color="blue"))

    for measurement_id in measurement_ids:
        print("\nmeasurement {}".format(measurement_id))
        meas_df = data_dict[measurement_id]

        keys_tuples = meas_df.keys_tuple.unique()
        print("keys_tuples:")
        for keys_tuple in keys_tuples:
            key_df = meas_df[meas_df.keys_tuple == keys_tuple]

            timestamp_ms_min = key_df.timestamp_ms.min()
            timestamp_ms_max = key_df.timestamp_ms.max()
            first_timestamp = datetime.fromtimestamp(timestamp_ms_min / 1000)  # from micro sec to milli sec
            last_timestamp = datetime.fromtimestamp(timestamp_ms_max / 1000)  # from micro sec to milli sec
            delta = last_timestamp - first_timestamp
            print("{}: from {} ({}) to {} ({}), delta (min): {:.2f}".format(keys_tuple, key_df.timestamp.min(),
                                                                            timestamp_ms_min,
                                                                            key_df.timestamp.max(),
                                                                            timestamp_ms_max,
                                                                            delta.total_seconds() / 60))
