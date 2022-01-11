import pandas as pd
import datetime
from termcolor import colored


class Measurement(object):
    def __init__(self, measurement_name, row_id, lightweight=True):
        self.measurement_name = measurement_name
        self. row_id = row_id
        self.lightweight = lightweight
        self.valid = True

        self.valid_start_time = None
        self.valid_end_time = None

        self.measurement_path_dict = {
            ("left", "arm", "acc"): None,
            ("left", "arm", "gyr"): None,
            ("left", "leg", "acc"): None,
            ("left", "leg", "gyr"): None,
            ("right", "arm", "acc"): None,
            ("right", "arm", "gyr"): None,
            ("right", "leg", "acc"): None,
            ("right", "leg", "gyr"): None,
        }

    def add_measurement_path(self, path):
        key = [None, None, None]

        if path.split('/')[-1].find("bal") != -1:
            key[0] = "left"
        elif path.split('/')[-1].find("jobb") != -1:
            key[0] = "right"
        else:
            raise ValueError("Bad measurement path: {}".format(path))

        if path.split('/')[-1].find("lab") != -1:
            key[1] = "leg"
        elif path.split('/')[-1].find("kar") != -1:
            key[1] = "arm"
        else:
            raise ValueError("Bad measurement path: {}".format(path))

        if path.split('/')[-1].find("Gyroscope") != -1:
            key[2] = "gyr"
        elif path.split('/')[-1].find("Accelerometer") != -1:
            key[2] = "acc"
        else:
            raise ValueError("Bad measurement path: {}".format(path))

        assert tuple(key) in self.measurement_path_dict
        self.measurement_path_dict[tuple(key)] = path

    def check_measurement_path_dict(self):
        for k, v in self.measurement_path_dict.items():
            if v is None:
                print(colored("{} measurement_path_dict is not full".format(self.measurement_name), "red"))
                self.valid = False
                return
        print(colored("{} measurement_path_dict is OK".format(self.measurement_name), "green"))

    def add_aux_data(self, aux_data_df):
        date = [int(x) for x in aux_data_df[1].split(" ")[0].split(".") if len(x) > 0]
        time = [int(x) for x in aux_data_df[1].split(" ")[1].split(":") if len(x) > 0]
        self.valid_start_time = datetime.datetime(date[0], date[1], date[2], time[0], time[1], time[2])

        date = [int(x) for x in aux_data_df[8].split(" ")[0].split(".") if len(x) > 0]
        time = [int(x) for x in aux_data_df[8].split(" ")[1].split(":") if len(x) > 0]
        self.valid_end_time = datetime.datetime(date[0], date[1], date[2], time[0], time[1], time[2])

    def get_measurement_df(self, key, only_valid=True):
        meas_df = pd.read_csv(self.measurement_path_dict[key])

        if only_valid and self.valid_start_time is not None and self.valid_end_time is not None:
            meas_df = meas_df[self.valid_start_time.timestamp() * 1000 < meas_df["epoch (ms)"]]
            meas_df = meas_df[meas_df["epoch (ms)"] < self.valid_end_time.timestamp() * 1000]

        return meas_df

    def get_all_measurements_df(self, only_valid=True):
        result_dict = dict()

        for k in self.measurement_path_dict.keys():
            result_dict[k] = self.get_measurement_df(k, only_valid=only_valid)

        return result_dict


