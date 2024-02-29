import numpy as np
import pandas as pd

from random import random
from typing import Union
from datetime import datetime, timedelta

from utils.general_utils import to_int_timestamp, min_to_ticks, to_str_timestamp, get_data_info
from measurement_utils.measurement import key_list_short
from utils.log_maker import write_log

pd.set_option('display.max_rows', 500)


class MeasurementManager(object):
    def __init__(self, config_dict: dict):
        self.config_dict = config_dict
        self.all_measurement_dict = dict()
        self.save_prediction_time = dict()
        self.timezone = config_dict["timezone"]

        # TODO:
        self.save_prediction_delay = timedelta(minutes=30)

    def is_time_to_save(self, measurement_id: str) -> bool:
        if measurement_id in self.save_prediction_time:
            if self.save_prediction_time[measurement_id] + self.save_prediction_delay < datetime.now(self.timezone):
                self.save_prediction_time[measurement_id] = datetime.now(self.timezone)
                return True
            else:
                return False
        else:
            self.save_prediction_time[measurement_id] = datetime.now(self.timezone)
            return True

    def get_last_timestamp(self, measurement_id: str) -> Union[int, None]:
        if measurement_id in self.all_measurement_dict:
            current_df = self.all_measurement_dict[measurement_id]
            max_list = list()
            for key in key_list_short:
                timestamps_ms = current_df[current_df["keys_tuple"] == key]["timestamp_ms"]
                if len(timestamps_ms) > 0:
                    max_list.append(current_df[current_df["keys_tuple"] == key]["timestamp_ms"].max())

            return min(max_list)
        else:
            return None

    def del_measurement(self, measurement_id: str) -> None:
        del self.all_measurement_dict[measurement_id]

    def drop_old_data(self):
        meas_ids_to_drop = list()
        for measurement_id in self.all_measurement_dict.keys():
            ts_now = datetime.now(self.timezone)
            until_ok_ts = ts_now - timedelta(minutes=self.config_dict["meas_length_to_keep_min"])
            df = self.all_measurement_dict[measurement_id]
            self.all_measurement_dict[measurement_id] = df[df["timestamp_ms"] >= until_ok_ts.timestamp() * 1000]

            if len(self.all_measurement_dict[measurement_id]) == 0:
                meas_ids_to_drop.append(measurement_id)
            else:
                assert self.all_measurement_dict[measurement_id]["timestamp_ms"].min() >= until_ok_ts.timestamp() * 1000

        for measurement_id in meas_ids_to_drop:
            del self.all_measurement_dict[measurement_id]

    def add_data(self, measurement_id: str, data_list: list, time_of_request: datetime) -> None:
        """ columns: limb, side, timestamp, type, x, y, z"""

        def cut_part_before_too_large_time_diff(_data_df: pd.DataFrame) -> pd.DataFrame:
            timestamp_ms = _data_df["timestamp_ms"].values
            too_large_diffs = np.diff(timestamp_ms) > self.config_dict["init_time_diff_threshold"]
            timestamp_limits = timestamp_ms[1:][too_large_diffs]

            if len(timestamp_limits) > 0:
                limit_ts_ms = timestamp_limits.max()
                _data_df = _data_df[_data_df["timestamp_ms"] >= limit_ts_ms]
                write_log("meas_manager.txt",
                          "beginning is cut at new measurement {} before timestamp {}".format(measurement_id,
                                                                                              limit_ts_ms),
                          title="CutBeginning", print_out=True, color="red", add_date=True)
            return _data_df

        def get_init_df() -> pd.DataFrame:
            min_ts = data_df["timestamp_ms"].values.min()
            df = pd.read_csv(self.config_dict["init_data"])
            df.sort_values(by="epoch", inplace=True, ascending=False)

            init_data_list = list()
            for idx, (_ , row) in enumerate(df.iterrows()):
                ts_ms = int(min_ts - (idx + 1) * (1000 / self.config_dict["frequency"]))  # 40 ms
                for meas_type in ["acc", "gyr"]:
                    init_data_list.append({
                        "side": "r",
                        "limb": "a",
                        "type": meas_type[0],
                        "timestamp": to_str_timestamp(ts_ms),
                        "timestamp_ms": ts_ms,
                        "x": row[str(("right", "arm", meas_type, "x"))],
                        "y": row[str(("right", "arm", meas_type, "y"))],
                        "z": row[str(("right", "arm", meas_type, "z"))]
                    })

            df = pd.DataFrame(init_data_list)
            df.sort_values(by="timestamp_ms", inplace=True)
            return df

        if len(data_list) == 0:
            return

        new_meas = False
        if measurement_id not in self.all_measurement_dict:
            self.all_measurement_dict[measurement_id] = pd.DataFrame()
            new_meas = True

        data_df = pd.DataFrame(data_list)
        data_df["timestamp_ms"] = data_df.apply(lambda row: to_int_timestamp(row.timestamp), axis=1)

        if new_meas:
            data_df = cut_part_before_too_large_time_diff(data_df)
            init_data_df = get_init_df()
            data_df = pd.concat([init_data_df, data_df], ignore_index=True)

        # TODO: get rid of mapping because of its time consumption
        # data_df["keys_tuple"] = data_df.apply(lambda row: key_map[(row.side, row.limb, row.type)], axis=1)
        data_df["keys_tuple"] = data_df.apply(lambda row: (row.side, row.limb, row.type), axis=1)
        data_df["time_of_request"] = time_of_request
        # get_data_info({measurement_id: data_df}, "new")

        self.all_measurement_dict[measurement_id] = pd.concat([self.all_measurement_dict[measurement_id], data_df],
                                                              ignore_index=True)

        current_df = self.all_measurement_dict[measurement_id]
        duplicates = current_df[current_df.columns.difference(["time_of_request"])].duplicated(keep="first")
        self.all_measurement_dict[measurement_id] = current_df[~duplicates]

        assert self.all_measurement_dict[measurement_id].duplicated(keep=False).sum() == 0, \
            self.all_measurement_dict[measurement_id][self.all_measurement_dict[measurement_id][
                self.all_measurement_dict[measurement_id].columns.difference(["time_of_request"])].duplicated(
                keep=False)]

        if self.config_dict["verbose"]:
            get_data_info(self.all_measurement_dict, "all")

    def get_df(self, measurement_id: str) -> pd.DataFrame:
        if measurement_id in self.all_measurement_dict:
            return self.all_measurement_dict[measurement_id]
        else:
            write_log("meas_manager.txt", "{} is not found in measurement manager ({})".format(measurement_id,
                                                                                               self.all_measurement_dict.keys()),
                      title="NotFound", print_out=True, color="red", add_date=True)

    def save_each_measurement(self) -> None:
        for measurement_id in self.all_measurement_dict.keys():
            path = "{}_{}.csv".format(measurement_id, datetime.now(self.timezone))
            self.all_measurement_dict[measurement_id].to_csv(path, index=False)
            print("saved measurement with path: {}".format(path))
