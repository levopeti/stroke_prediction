from datetime import datetime


def to_int_timestamp(timestamp_str):
    return int(datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%S.%fZ').timestamp() * 1000)


def to_str_timestamp(timestamp_int):
    dt_object = datetime.fromtimestamp(timestamp_int / 1000)
    return dt_object.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

