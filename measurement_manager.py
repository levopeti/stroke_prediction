import pandas as pd

from typing import Union
from datetime import datetime, timedelta

from general_utils import to_int_timestamp, to_str_timestamp, get_data_info
from measurement import key_map


class MeasurementManager(object):
    def __init__(self, config_dict: dict):
        self.config_dict = config_dict
        self.all_measurement_dict = dict()

    def get_last_timestamp(self, measurement_id: str) -> Union[int, None]:
        if measurement_id in self.all_measurement_dict:
            return self.all_measurement_dict[measurement_id]["timestamp_ms"].max()
        else:
            return None

    def drop_old_data(self, measurement_id: str):
        if measurement_id in self.all_measurement_dict:
            ts_now = datetime.now()
            until_ok_ts = ts_now - timedelta(minutes=self.config_dict["meas_length_to_keep_min"])
            df = self.all_measurement_dict[measurement_id]
            self.all_measurement_dict[measurement_id] = df[df["timestamp_ms"] >= until_ok_ts.timestamp() * 1000]

            assert self.all_measurement_dict[measurement_id]["timestamp_ms"].min() >= until_ok_ts.timestamp() * 1000

    def add_data(self, measurement_id: str, data_list: list):
        """ columns: limb, side, timestamp, type, x, y, z"""

        if measurement_id not in self.all_measurement_dict:
            self.all_measurement_dict[measurement_id] = pd.DataFrame()

        data_df = pd.DataFrame(data_list)
        data_df["timestamp_ms"] = data_df.apply(lambda row: to_int_timestamp(row.timestamp), axis=1)
        data_df["keys_tuple"] = data_df.apply(lambda row: key_map[(row.side, row.limb, row.type)], axis=1)
        get_data_info({measurement_id: data_df}, "new")

        self.all_measurement_dict[measurement_id] = pd.concat([self.all_measurement_dict[measurement_id], data_df],
                                                              ignore_index=True)
        assert self.all_measurement_dict[measurement_id].duplicated(keep=False).sum() == 0,\
            self.all_measurement_dict[measurement_id].duplicated(keep=False).sum()
        get_data_info(self.all_measurement_dict, "all")

    def get_df(self, measurement_id: str) -> pd.DataFrame:
        if measurement_id in self.all_measurement_dict:
            return self.all_measurement_dict[measurement_id]
        else:
            print("{} is not found in measurement manager ({})".format(measurement_id,
                                                                       self.all_measurement_dict.keys()))
