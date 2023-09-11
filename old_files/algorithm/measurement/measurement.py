from typing import Union

import numpy as np

from datetime import datetime
from termcolor import colored
from random import randint

from ....utils.general_utils import min_to_millisec
from .get_measurement_utils import read_csv, synchronize

class Measurement(object):
    def __init__(self, measurement_name, row_id, neurology_df, synchronizing=True, lightweight=False, ratio=None):
        self.measurement_name = measurement_name
        self.row_id = row_id
        self.synchronizing = synchronizing
        self.lightweight = lightweight
        self.set_type = None
        self.valid = False

        self.valid_start_time = None
        self.valid_end_time = None

        self.ratio = ratio
        self.length = None

        self.log_list = list()

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

        self.measurement_dict = {
            ("left", "arm", "acc"): None,
            ("left", "arm", "gyr"): None,
            ("left", "leg", "acc"): None,
            ("left", "leg", "gyr"): None,
            ("right", "arm", "acc"): None,
            ("right", "arm", "gyr"): None,
            ("right", "leg", "acc"): None,
            ("right", "leg", "gyr"): None,
        }

        self.diff_dict = {
            ("left", "arm", "acc"): None,
            ("left", "arm", "gyr"): None,
            ("left", "leg", "acc"): None,
            ("left", "leg", "gyr"): None,
            ("right", "arm", "acc"): None,
            ("right", "arm", "gyr"): None,
            ("right", "leg", "acc"): None,
            ("right", "leg", "gyr"): None,
        }

        if neurology_df is not None:
            self.class_value_dict = {
                ("left", "arm"): int(eval(neurology_df["ParStatBK"][self.row_id]) * 5),
                ("left", "leg"): int(eval(neurology_df["ParStatBL"][self.row_id]) * 5),
                ("right", "arm"): int(eval(neurology_df["ParStatJK"][self.row_id]) * 5),
                ("right", "leg"): int(eval(neurology_df["ParStatJL"][self.row_id]) * 5),
            }
        else:
            self.class_value_dict = {
                ("left", "arm"): None,
                ("left", "leg"): None,
                ("right", "arm"): None,
                ("right", "leg"): None,
            }

    def add_measurement_path(self, path):
        self.valid = True
        key = [None, None, None]
        # ['202110230', 'MetaWear-R-L', '2021-10-24T01.42.26.469', 'D087CCFD3C25', 'Accelerometer', '1.6.2.csv']

        if path.split('/')[-1].split('_')[1].split('-')[1] == 'L':
            key[0] = "left"
        elif path.split('/')[-1].split('_')[1].split('-')[1] == 'R':
            key[0] = "right"
        else:
            raise ValueError("Bad measurement path: {}".format(path))

        if path.split('/')[-1].split('_')[1].split('-')[2] == 'L':
            key[1] = "leg"
        elif path.split('/')[-1].split('_')[1].split('-')[2] == 'A':
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
        assert self.measurement_path_dict[tuple(key)] is None, self.measurement_path_dict[tuple(key)]
        self.measurement_path_dict[tuple(key)] = path

    def num_of_missing_paths(self):
        result = 0
        for v in self.measurement_path_dict.values():
            if v is None:
                result += 1

        return result

    # #### checks for measurement ####

    def check_measurement_path_dict(self):
        missing_keys = list()
        for k, v in self.measurement_path_dict.items():
            if v is None:
                missing_keys.append(k)

        if len(missing_keys) > 0:
            if len(missing_keys) == 8:
                self.log_list.append(colored("missing keys: ALL", "red"))
            else:
                self.log_list.append(colored("missing keys: {}".format(missing_keys), "red"))
            self.valid = False

    def check_five_class(self):
        for class_value in self.class_value_dict.values():
            if class_value == 5:
                return

        self.log_list.append(colored("no limb with class 5".format(self.measurement_name), "red"))

    def check_loading(self, only_valid=True):
        for k in self.measurement_path_dict.keys():
            if k is not None:
                try:
                    self.get_measurement_df(k, only_valid=only_valid)
                except ValueError as e:
                    self.log_list.append(colored("{}, {}, {}".format(self.measurement_name,
                                                                     self.measurement_path_dict[k], e), "red"))

    def check_frequency(self, expected_delta, eps):
        for df in self.measurement_dict.values():
            if df is not None:
                time_stamps = df["epoch"].values
                deltas = np.diff(time_stamps)
                if np.any(deltas < expected_delta - eps) or np.any(deltas > expected_delta + eps):
                    self.log_list.append(colored("frequency is not correct,"
                                                 " min: {}, max: {}, avg: {}".format(np.min(deltas),
                                                                                     np.max(deltas),
                                                                                     np.mean(deltas)), "red"))

    def add_aux_data(self, aux_data_df):
        self.valid_start_time = aux_data_df["The last sensor is on the patient"].values[0]

        if isinstance(self.valid_start_time, datetime):
            self.valid_start_time = int(self.valid_start_time.timestamp() * 1e3)
        elif isinstance(self.valid_start_time, np.datetime64):
            self.valid_start_time = int(self.valid_start_time.astype(datetime) / 1e6)
        else:
            raise TypeError("valid start time neither datetime nor \
            np.datetime64 ({}, {})".format(type(self.valid_start_time), self.valid_start_time))

        self.valid_end_time = aux_data_df["Take off the first sensor"].values[0]

        if isinstance(self.valid_end_time, datetime):
            self.valid_end_time = int(self.valid_end_time.timestamp() * 1e3)
        elif isinstance(self.valid_end_time, np.datetime64):
            self.valid_end_time = int(self.valid_end_time.astype(datetime) / 1e6)
        else:
            raise TypeError("valid end time neither datetime nor \
            np.datetime64 ({}, {})".format(type(self.valid_end_time), self.valid_end_time))

    def print_log(self):
        print(colored("### {} (la: {}, ll: {}, ra: {}, rl: {}) ###".format(self.measurement_name,
                                                                           *[self.class_value_dict[("left", "arm")],
                                                                             self.class_value_dict[("left", "leg")],
                                                                             self.class_value_dict[("right", "arm")],
                                                                             self.class_value_dict[("right", "leg")]]), "blue"))
        for log in self.log_list:
            print(log)

    def get_absolute_class_value(self, start_idx=None, length=None):
        if self.ratio is None:
            return min(self.class_value_dict.values())
        else:
            if start_idx is not None and length is not None:
                if start_idx + length < self.length[0]:
                    return min(min(self.class_value_dict.values(), key=lambda x: x[0]))
                else:
                    return min(min(self.class_value_dict.values(), key=lambda x: x[1]))
            else:
                return (min(min(self.class_value_dict.values(), key=lambda x: x[0])),
                        min(min(self.class_value_dict.values(), key=lambda x: x[1])))

    def get_class_value(self, key):
        if len(key) == 3:
            key = key[:2]
        return self.class_value_dict[key]

    def get_limb_class_value(self, limb):
        assert limb in ["arm", "leg"]
        return min(self.class_value_dict[("left", limb)], self.class_value_dict[("right", limb)])

    def get_measurement_df(self, key, only_valid=True):
        columns_key_dict = {"acc": ("epoch (ms)", "epoc (ms)", "x-axis (g)", "y-axis (g)", "z-axis (g)"),
                            "gyr": ("epoch (ms)", "epoc (ms)", "x-axis (deg/s)", "y-axis (deg/s)", "z-axis (deg/s)")}

        if not self.lightweight and self.measurement_dict[key] is not None:
            meas_df = self.measurement_dict[key]
        else:
            if self.synchronizing:
                tmp_measurement_dict = dict()
                for k in self.measurement_path_dict.keys():
                    meas_df = read_csv(k, self, columns_key_dict, only_valid)
                    tmp_measurement_dict[k] = meas_df

                tmp_measurement_dict = synchronize(tmp_measurement_dict, self)
                meas_df = tmp_measurement_dict[key]

                if not self.lightweight:
                    self.measurement_dict = tmp_measurement_dict
            else:
                meas_df = read_csv(key, self, columns_key_dict, only_valid)
                if not self.lightweight:
                    self.measurement_dict[key] = meas_df

        assert len(meas_df) > 0
        # if len(meas_df) == 0:
        #     print(self.measurement_name)
        #     print(key)
        #     exit()

        return meas_df

    def get_all_measurements_df(self, only_valid=True):
        result_dict = dict()

        for k in self.measurement_path_dict.keys():
            result_dict[k] = self.get_measurement_df(k, only_valid=only_valid)

        return result_dict

    def get_mutual_limb_masks(self, limb, meas_type="acc", only_valid=True):
        left_meas = self.get_measurement_df(("left", limb, meas_type), only_valid)
        right_meas = self.get_measurement_df(("right", limb, meas_type), only_valid)

        left_mask = (left_meas["epoch"] >= right_meas["epoch"].min()) & \
                    (left_meas["epoch"] <= right_meas["epoch"].max())
        right_mask = (right_meas["epoch"] >= left_meas["epoch"].min()) & \
                     (right_meas["epoch"] <= left_meas["epoch"].max())

        return left_mask, right_mask

    def shift_all_measurements(self, delay_min: Union[int, float]):
        for key in self.measurement_dict.keys():
            assert self.measurement_dict[key] is not None, "self.measurement_dict[{}] is None".format(key)
            self.measurement_dict[key] += min_to_millisec(delay_min)

    def calculate_diff(self, key, use_abs=True, only_valid=True):
        if not self.lightweight and self.diff_dict[key] is not None:
            result = self.diff_dict[key]
        else:
            meas_type = key[2]
            meas = self.get_measurement_df(key, only_valid=only_valid)
            x_y_z = [meas[("x-axis", "y-axis", "z-axis")[i]] for i in range(3)]

            if meas_type == "acc":
                x_diff, y_diff, z_diff = [np.diff(m) for m in x_y_z]
            else:
                x_diff, y_diff, z_diff = [m.values for m in x_y_z]

            if use_abs:
                result = np.abs(x_diff) + np.abs(y_diff) + np.abs(z_diff)
            else:
                result = x_diff + y_diff + z_diff

            if not self.lightweight:
                self.diff_dict[key] = result

        assert len(result) > 0
        return result

    def get_diff(self, key, length=None, start_idx=None, use_abs=True, only_valid=True, mask=None):
        result = self.calculate_diff(key, use_abs, only_valid)

        if mask is not None:
            try:
                result = result[mask[:len(result)]]
            except Exception as e:
                print(e)
                print(key, length, start_idx)
                print(result.shape)
                print(mask[:len(result)].shape)
                exit()

        if length is not None:
            assert length < len(result)

            start_idx = start_idx if start_idx is not None else randint(0, len(result) - (length + 1))
            if start_idx > len(result) - (length + 1):
                raise ValueError("start_idx is too large")

            result = result[start_idx:start_idx + length]

        assert len(result) > 0
        return result

    def sweep_diff(self, key, length, mean=False):
        result_list = list()
        start_idx = 0

        while True:
            try:
                if mean:
                    result_list.append(self.get_diff(key, length, start_idx).mean())
                else:
                    result_list.append(self.get_diff(key, length, start_idx))
            except ValueError:
                break
            start_idx = start_idx + 1

        return result_list

    def get_all_diff(self, only_valid=True):
        result_dict = dict()

        for k in self.measurement_path_dict.keys():
            result_dict[k] = self.get_diff(k, only_valid=only_valid)

        return result_dict

    def get_limb_diff_mean(self, limb, meas_type="acc", length=None, start_idx=None, use_abs=True, only_valid=True):
        assert limb in ["arm", "leg"]
        assert meas_type in ["acc", "gyr"]
        left_key, right_key = ("left", limb, meas_type), ("right", limb, meas_type)

        if not self.synchronizing:
            left_mask, right_mask = self.get_mutual_limb_masks(limb, meas_type)
        else:
            left_mask, right_mask = None, None

        left_diff = self.get_diff(left_key, length, start_idx, use_abs, only_valid, left_mask)
        right_diff = self.get_diff(right_key, length, start_idx, use_abs, only_valid, right_mask)

        result = np.abs(left_diff.mean() - right_diff.mean())
        is_five = self.class_value_dict[("left", limb)] == 5 or self.class_value_dict[("right", limb)] == 5
        return result, is_five

    def get_limb_ratio_mean(self, limb, meas_type="acc", length=None, start_idx=None, use_abs=True, only_valid=True,
                            mean_first=True):
        assert limb in ["arm", "leg"]
        assert meas_type in ["acc", "gyr"]
        left_key, right_key = ("left", limb, meas_type), ("right", limb, meas_type)

        if not self.synchronizing:
            left_mask, right_mask = self.get_mutual_limb_masks(limb, meas_type)
        else:
            left_mask, right_mask = None, None

        left_diff = self.get_diff(left_key, length, start_idx, use_abs, only_valid, left_mask)
        right_diff = self.get_diff(right_key, length, start_idx, use_abs, only_valid, right_mask)

        if mean_first:
            if self.class_value_dict[("left", limb)] > self.class_value_dict[("right", limb)]:
                result = left_diff.sum() / right_diff.sum()
            else:
                result = right_diff.sum() / left_diff.sum()
        else:
            left_diff = left_diff + 0.1
            right_diff = right_diff + 0.1
            if self.class_value_dict[("left", limb)] > self.class_value_dict[("right", limb)]:
                result = np.mean(left_diff / right_diff)
            else:
                result = np.mean(right_diff / left_diff)

        is_five = self.class_value_dict[("left", limb)] == 5 or self.class_value_dict[("right", limb)] == 5
        return result, is_five
