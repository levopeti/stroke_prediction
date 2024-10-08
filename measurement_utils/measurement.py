import pandas as pd
import numpy as np

from termcolor import colored

from utils.general_utils import to_str_timestamp
from typing import Union
from utils.discord import DiscordBot
from utils.log_maker import write_log


class NotEnoughData(Exception):
    pass


class TimeStampTooHigh(Exception):
    pass


class SynchronizationError(Exception):
    pass


key_list_long = [("left", "arm", "acc"),
                 ("left", "arm", "gyr"),
                 ("left", "leg", "acc"),
                 ("left", "leg", "gyr"),
                 ("right", "arm", "acc"),
                 ("right", "arm", "gyr"),
                 ("right", "leg", "acc"),
                 ("right", "leg", "gyr")]

key_list_short = [("l", "a", "a"),
                  ("l", "a", "g"),
                  ("l", "l", "a"),
                  ("l", "l", "g"),
                  ("r", "a", "a"),
                  ("r", "a", "g"),
                  ("r", "l", "a"),
                  ("r", "l", "g")]

key_map = {
    ("l", "a", "a"): ("left", "arm", "acc"),
    ("l", "a", "g"): ("left", "arm", "gyr"),
    ("l", "l", "a"): ("left", "leg", "acc"),
    ("l", "l", "g"): ("left", "leg", "gyr"),
    ("r", "a", "a"): ("right", "arm", "acc"),
    ("r", "a", "g"): ("right", "arm", "gyr"),
    ("r", "l", "a"): ("right", "leg", "acc"),
    ("r", "l", "g"): ("right", "leg", "gyr"),
}


class Measurement(object):
    def __init__(self, measurement_id: str, synchronizing: int = True):
        self.measurement_id = measurement_id
        self.synchronizing = synchronizing

        self.log_list = list()

        # self.measurement_dict = {
        #     ("left", "arm", "acc"): None,
        #     ("left", "arm", "gyr"): None,
        #     ("left", "leg", "acc"): None,
        #     ("left", "leg", "gyr"): None,
        #     ("right", "arm", "acc"): None,
        #     ("right", "arm", "gyr"): None,
        #     ("right", "leg", "acc"): None,
        #     ("right", "leg", "gyr"): None,
        # }

        # TODO
        self.measurement_dict = {
            ("l", "a", "a"): None,
            ("l", "a", "g"): None,
            # ("l", "l", "a"): None,
            # ("l", "l", "g"): None,
            # ("r", "a", "a"): None,
            # ("r", "a", "g"): None,
            # ("r", "l", "a"): None,
            # ("r", "l", "g"): None,
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

    def fill_from_df(self, data_df: pd.DataFrame):
        """
        data_df must have columns: "timestamp_ms", "keys_tuple", "x", "y", "z"
        """
        for keys in data_df.keys_tuple.unique():
            keys_df = data_df[data_df.keys_tuple == keys]
            self.measurement_dict[keys] = keys_df[["timestamp_ms", "x", "y", "z"]]

        # TODO: check each key

    # #### checks for measurement ####
    def check_frequency(self, expected_delta_ms: int, eps: int) -> bool:
        check_ok = True
        for keys, df in self.measurement_dict.items():
            if df is not None:
                time_stamps = df["timestamp_ms"].values
                deltas = np.diff(time_stamps)
                less_mask = deltas < expected_delta_ms - eps
                more_mask = deltas > expected_delta_ms + eps
                if np.any(less_mask) or np.any(more_mask):
                    less_deltas = deltas[less_mask]
                    more_deltas = deltas[more_mask]

                    # less_timestamps = time_stamps[:-1][less_mask]
                    # more_timestamps = time_stamps[:-1][more_mask]

                    less_timestamps = ", ".join(
                        ["{} ({})".format(to_str_timestamp(time_stamps[:-1][less_mask][i]), less_deltas[i]) for i in
                         range(len(less_deltas))])
                    more_timestamps = ", ".join(
                        ["{} ({})".format(to_str_timestamp(time_stamps[:-1][more_mask][i]), more_deltas[i]) for i in
                         range(len(more_deltas))])

                    log = ("frequency is not correct, with key: {} min: {}, max: {}, avg: {:.2f}\n"
                           "timestamps of smaller: {}\ntimestamps of larger: {}".format(keys,
                                                                                        np.min(deltas),
                                                                                        np.max(deltas),
                                                                                        np.mean(deltas),
                                                                                        less_timestamps,
                                                                                        more_timestamps))
                    write_log("main_loop.txt", log, title="SynchronizationError", print_out=True, color="red",
                              add_date=True, write_discord=True)
                    self.log_list.append(colored(log, "red"))
                    check_ok = False
        return check_ok

    def get_missing_keys(self) -> list:
        missing_keys = list()
        for key, meas in self.measurement_dict.items():
            if meas is None:
                missing_keys.append(key)
        return missing_keys

    def check_length(self, length: int) -> bool:
        current_length = self.get_last_timestamp_ms() - self.get_first_timestamp_ms()
        if current_length > length:
            return True
        else:
            print("current length: {}, expected length: {}".format(current_length, length))
            return False

    def print_log(self):
        for log in self.log_list:
            print(log)

    def synchronize_measurement_dict(self):
        # def cut_valid_part(_meas_df):
        #     if only_valid and self.valid_start_time is not None and self.valid_end_time is not None:
        #         # _meas_df = _meas_df[_meas_df["epoch"] > self.valid_start_time.timestamp() * 1000]
        #         # _meas_df = _meas_df[_meas_df["epoch"] < self.valid_end_time.timestamp() * 1000]
        #         _meas_df = _meas_df[_meas_df["timestamp"] > int(self.valid_start_time / np.timedelta64(1, 'ms'))]
        #         _meas_df = _meas_df[_meas_df["timestamp"] < int(self.valid_end_time / np.timedelta64(1, 'ms'))]
        #     return _meas_df

        def cut_for_mutual_part(_measurement_dict):
            min_ts = 0
            max_ts = float('inf')

            for meas in _measurement_dict.values():
                if meas["timestamp_ms"].min() > min_ts:
                    min_ts = meas["timestamp_ms"].min()

                if meas["timestamp_ms"].max() < max_ts:
                    max_ts = meas["timestamp_ms"].max()

            for _k, meas in _measurement_dict.items():
                # print(len(meas[(meas["epoch"] >= min_ts) & (meas["epoch"] <= max_ts)]))
                _measurement_dict[_k] = meas[(meas["timestamp_ms"] >= min_ts) & (meas["timestamp_ms"] <= max_ts)]
            return _measurement_dict

        def synchronize(_measurement_dict):
            _measurement_dict = cut_for_mutual_part(_measurement_dict)

            base_df = None
            for _k, _df in _measurement_dict.items():
                if base_df is None:
                    base_df = _df.sort_values('timestamp_ms')
                else:
                    _df = _df.sort_values('timestamp_ms')
                    # TODO: tolerance parameter
                    merged_df = pd.merge_asof(base_df, _df, on="timestamp_ms", tolerance=40, direction='nearest')
                    if merged_df.isna().sum().sum() != 0:
                        raise SynchronizationError("merged df has nans during synchronization")

                    columns_for_drop = list()
                    for c_name in merged_df.columns:
                        if c_name.find("_x") != -1:
                            columns_for_drop.append(c_name)
                    merged_df.drop(columns_for_drop, inplace=True, axis=1)

                    # remove _y
                    for c_name in merged_df.columns:
                        if c_name.find("_y") != -1:
                            merged_df.rename(columns={c_name: c_name[:-2]}, inplace=True)

                    _measurement_dict[_k] = merged_df

            for _k, _df in _measurement_dict.items():
                if len(_df) == 0:
                    print(colored("zero length of data {}, {}".format(self.measurement_id, _k), "red"))
            return _measurement_dict

        self.measurement_dict = synchronize(self.measurement_dict)

    def get_first_timestamp_ms(self) -> Union[None, int]:
        for keys in self.measurement_dict.keys():
            if self.measurement_dict[keys] is not None:
                return self.measurement_dict[keys]["timestamp_ms"].min()

    def get_last_timestamp_ms(self) -> Union[None, int]:
        for keys in self.measurement_dict.keys():
            if self.measurement_dict[keys] is not None and len(self.measurement_dict[keys]) > 0:
                return self.measurement_dict[keys]["timestamp_ms"].max()
            else:
                return

    # def get_all_measurements_df(self, only_valid=True):
    #     result_dict = dict()
    #
    #     for k in self.measurement_path_dict.keys():
    #         result_dict[k] = self.get_measurement_df(k, only_valid=only_valid)
    #
    #     return result_dict

    def get_mutual_limb_masks(self, limb, meas_type="acc"):
        left_meas = self.measurement_dict[("left", limb, meas_type)]
        right_meas = self.measurement_dict[("right", limb, meas_type)]

        left_mask = (left_meas["timestamp_ms"] >= right_meas["timestamp_ms"].min()) & \
                    (left_meas["timestamp_ms"] <= right_meas["timestamp_ms"].max())
        right_mask = (right_meas["timestamp_ms"] >= left_meas["timestamp_ms"].min()) & \
                     (right_meas["timestamp_ms"] <= left_meas["timestamp_ms"].max())

        return left_mask, right_mask

    def calculate_diff(self, key, use_abs=True):
        meas_type = key[2]
        meas = self.measurement_dict[key]

        if meas_type == "acc":
            timestamps = meas["timestamp_ms"].values[:-1]
        else:
            timestamps = meas["timestamp_ms"].values

        if self.diff_dict[key] is not None:
            # TODO: use_abs ???
            return self.diff_dict[key], timestamps

        x_y_z = [meas[("x", "y", "z")[i]] for i in range(3)]

        if meas_type == "acc":
            x_diff, y_diff, z_diff = [np.diff(m) for m in x_y_z]
        else:
            x_diff, y_diff, z_diff = [m.values for m in x_y_z]

        if use_abs:
            result = np.abs(x_diff) + np.abs(y_diff) + np.abs(z_diff)
        else:
            result = x_diff + y_diff + z_diff

        self.diff_dict[key] = result

        assert len(result) > 0, "len(result) = 0"
        return result, timestamps

    def get_diff(self, key, length=None, end_ts=None, use_abs=True, mask=None):
        result, timestamps = self.calculate_diff(key, use_abs)
        if end_ts > timestamps.max():
            raise TimeStampTooHigh("End timestamp ({})"
                                   " is higher than the maximum ({})".format(to_str_timestamp(end_ts),
                                                                             to_str_timestamp(timestamps.max())))
        result = result[timestamps < end_ts]

        if mask is not None:
            result = result[mask[:len(result)]]

        if length is not None:
            if length > len(result):
                raise NotEnoughData("After filtering we have less data ({}) than expected ({})".format(len(result),
                                                                                                       length))
            else:
                result = result[-length:]

        assert len(result) > 0, "len(result) = 0"
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

    def get_all_diff(self):
        result_dict = dict()

        for k in self.measurement_dict.keys():
            result_dict[k] = self.get_diff(k)

        return result_dict

    def get_limb_diff_mean(self, limb, meas_type="acc", length=None, end_ts=None, use_abs=True, only_valid=True):
        assert limb in ["arm", "leg"], "{} not in [arm, leg]".format(limb)
        assert meas_type in ["acc", "gyr"], "{} not in [acc, gyr]".format(meas_type)
        left_key, right_key = ("left", limb, meas_type), ("right", limb, meas_type)

        if not self.synchronizing:
            left_mask, right_mask = self.get_mutual_limb_masks(limb, meas_type)
        else:
            left_mask, right_mask = None, None

        left_diff = self.get_diff(left_key, length, end_ts, use_abs, left_mask)
        right_diff = self.get_diff(right_key, length, end_ts, use_abs, right_mask)

        result = np.abs(left_diff.mean() - right_diff.mean())
        # is_five = self.class_value_dict[("left", limb)] == 5 or self.class_value_dict[("right", limb)] == 5
        return result

    def get_limb_ratio_mean(self, limb, meas_type="acc", length=None, end_ts=None, use_abs=True, only_valid=True,
                            mean_first=True):
        assert limb in ["arm", "leg"], "{} not in [arm, leg]".format(limb)
        assert meas_type in ["acc", "gyr"], "{} not in [acc, gyr]".format(meas_type)
        left_key, right_key = ("left", limb, meas_type), ("right", limb, meas_type)

        if not self.synchronizing:
            left_mask, right_mask = self.get_mutual_limb_masks(limb, meas_type)
        else:
            left_mask, right_mask = None, None

        left_diff = self.get_diff(left_key, length, end_ts, use_abs, left_mask)
        right_diff = self.get_diff(right_key, length, end_ts, use_abs, right_mask)

        if mean_first:
            result = (left_diff.sum() + 0.1) / (right_diff.sum() + 0.1)
            # if self.class_value_dict[("left", limb)] > self.class_value_dict[("right", limb)]:
            #     result = left_diff.sum() / right_diff.sum()
            # else:
            #     result = right_diff.sum() / left_diff.sum()
        else:
            left_diff = left_diff + 0.1
            right_diff = right_diff + 0.1
            result = np.mean(left_diff / right_diff)
            # if self.class_value_dict[("left", limb)] > self.class_value_dict[("right", limb)]:
            #     result = np.mean(left_diff / right_diff)
            # else:
            #     result = np.mean(right_diff / left_diff)

        # is_five = self.class_value_dict[("left", limb)] == 5 or self.class_value_dict[("right", limb)] == 5
        return result
