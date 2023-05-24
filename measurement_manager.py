import pandas as pd
import pytz

from typing import Union
from datetime import datetime, timedelta

from general_utils import to_int_timestamp, to_str_timestamp, get_data_info
from measurement import key_map

pd.set_option('display.max_rows', 500)


class MeasurementManager(object):
    def __init__(self, config_dict: dict, timezone: pytz.timezone):
        self.config_dict = config_dict
        self.all_measurement_dict = dict()
        self.timezone = timezone

    def get_last_timestamp(self, measurement_id: str) -> Union[int, None]:
        if measurement_id in self.all_measurement_dict:
            return self.all_measurement_dict[measurement_id]["timestamp_ms"].max()
        else:
            return None

    def drop_old_data(self):
        for measurement_id in self.all_measurement_dict.keys():
            ts_now = datetime.now(self.timezone)
            until_ok_ts = ts_now - timedelta(minutes=self.config_dict["meas_length_to_keep_min"])
            df = self.all_measurement_dict[measurement_id]
            self.all_measurement_dict[measurement_id] = df[df["timestamp_ms"] >= until_ok_ts.timestamp() * 1000]

            if len(self.all_measurement_dict[measurement_id]) == 0:
                del self.all_measurement_dict[measurement_id]

            if len(self.all_measurement_dict[measurement_id]):
                assert self.all_measurement_dict[measurement_id]["timestamp_ms"].min() >= until_ok_ts.timestamp() * 1000

    def add_data(self, measurement_id: str, data_list: list, time_of_request: datetime):
        """ columns: limb, side, timestamp, type, x, y, z"""

        if measurement_id not in self.all_measurement_dict:
            self.all_measurement_dict[measurement_id] = pd.DataFrame()

        data_df = pd.DataFrame(data_list)
        data_df["timestamp_ms"] = data_df.apply(lambda row: to_int_timestamp(row.timestamp), axis=1)
        data_df["keys_tuple"] = data_df.apply(lambda row: key_map[(row.side, row.limb, row.type)], axis=1)
        data_df["time_of_request"] = time_of_request
        # get_data_info({measurement_id: data_df}, "new")

        self.all_measurement_dict[measurement_id] = pd.concat([self.all_measurement_dict[measurement_id], data_df],
                                                              ignore_index=True)

        current_df = self.all_measurement_dict[measurement_id]
        duplicates = current_df[current_df.columns.difference(["time_of_request"])].duplicated(keep="first")
        self.all_measurement_dict[measurement_id] = current_df[~duplicates]

        assert self.all_measurement_dict[measurement_id].duplicated(keep=False).sum() == 0,\
            self.all_measurement_dict[measurement_id][self.all_measurement_dict[measurement_id][
                self.all_measurement_dict[measurement_id].columns.difference(["time_of_request"])].duplicated(keep=False)]

        # get_data_info(self.all_measurement_dict, "all")

    def get_df(self, measurement_id: str) -> pd.DataFrame:
        if measurement_id in self.all_measurement_dict:
            return self.all_measurement_dict[measurement_id]
        else:
            print("{} is not found in measurement manager ({})".format(measurement_id,
                                                                       self.all_measurement_dict.keys()))
