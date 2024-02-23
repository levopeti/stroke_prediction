import numpy as np

from datetime import datetime, timedelta
from typing import Union
from utils.log_maker import write_log


def get_meas_info(meas):
    print("measurement_dict")
    for key, values_df in meas.measurement_dict.items():
        print(key)
        min_ts = values_df.timestamp_ms.min()
        max_ts = values_df.timestamp_ms.max()
        print("min ts: {}, {}".format(to_str_timestamp(min_ts), min_ts))
        print("max ts: {}, {}".format(to_str_timestamp(max_ts), max_ts))
        print("length: {}\n".format(len(values_df)))

    print("diff_dict")
    for key in meas.diff_dict.keys():
        print(key)
        meas.calculate_diff(key)
        values = meas.diff_dict[key]
        print("length: {}\n".format(len(values)))


def to_int_timestamp(timestamp_str: str) -> int:
    """ timestamp in microseconds
        first case: 2023-04-28T14:47:00Z
        second case: 2023-04-28T14:47:12.123Z
    """
    if timestamp_str.find(".") == -1:
        return int(datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%SZ").timestamp() * 1000)
    else:
        return int(datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S.%fZ").timestamp() * 1000)


def to_str_timestamp(timestamp: Union[int, datetime]):
    if isinstance(timestamp, (int, np.int64)):
        dt_object = datetime.fromtimestamp(timestamp / 1000)
    else:
        dt_object = timestamp
    return dt_object.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def from_int_to_datetime(timestamp: Union[int, None]) -> Union[datetime, None]:
    if timestamp is None:
        return None
    else:
        return datetime.fromtimestamp(timestamp / 1000)


def hour_to_millisec(hour: Union[int, float]) -> int:
    return int(hour * 60 * 60 * 1000)


def min_to_millisec(minute: Union[int, float]) -> int:
    return int(minute * 60 * 1000)


def min_to_ticks(time_min: int, frequency: int) -> int:
    # min -> sec -> sec * Hz
    num_of_tick = int(time_min * 60 * frequency)
    return num_of_tick


def sec_to_ticks(time_sec: int, frequency: int) -> int:
    # sec -> sec * Hz
    num_of_tick = int(time_sec * frequency)
    return num_of_tick


def get_length_from_timestamps(start_ts: int, end_ts: int, full_format: bool = False) -> timedelta:
    start = datetime.fromtimestamp(start_ts / 1000)
    end = datetime.fromtimestamp(end_ts / 1000)
    length = end - start

    if not full_format:
        mm, ss = divmod(length.total_seconds(), 60)
        hh, mm = divmod(mm, 60)
        length = "{}:{}".format(int(hh), int(mm))
    return length


def get_data_info(data_dict: dict, prefix: str = ""):
    """ columns: limb, side, timestamp, type, x, y, z, timestamp_ms, keys_tuple"""
    measurement_ids = sorted(data_dict.keys())
    # print(colored("\n{} measurement ids: {}".format(prefix, measurement_ids), color="blue"))
    write_log("main_loop.txt", "{} measurement ids: {}".format(prefix, measurement_ids),
              title="GetDataInfo", print_out=True, color="blue", add_date=True)

    for measurement_id in measurement_ids:
        # print("\nmeasurement {}".format(measurement_id))
        write_log("main_loop.txt", "measurement {}".format(measurement_id),
                  title="GetDataInfo", print_out=True, color="blue", add_date=True)
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
            # print("{}: from {} ({}) to {} ({}), delta (min): {:.2f}".format(keys_tuple, key_df.timestamp.min(),
            #                                                                 timestamp_ms_min,
            #                                                                 key_df.timestamp.max(),
            #                                                                 timestamp_ms_max,
            #                                                                 delta.total_seconds() / 60))
            write_log("main_loop.txt", "{}: from {} ({}) to {} ({}), delta (min): {:.2f}".format(keys_tuple, key_df.timestamp.min(),
                                                                                                 timestamp_ms_min,
                                                                                                 key_df.timestamp.max(),
                                                                                                 timestamp_ms_max,
                                                                                                 delta.total_seconds() / 60),
                      title="GetDataInfo", print_out=True, color="blue", add_date=True)
