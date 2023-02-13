import pandas as pd

from general_utils import to_int_timestamp, to_str_timestamp, get_data_info
from measurement import key_map


class AllMeasurementsDF(object):
    def __init__(self):
        self.all_data_df = pd.DataFrame()

    def add_data(self, data_list: list):
        """ columns: limb, side, timestamp, type, x, y, z"""

        data_df = pd.DataFrame(data_list)
        data_df["timestamp_ms"] = data_df.apply(lambda row: to_int_timestamp(row.timestamp), axis=1)
        data_df["keys_tuple"] = data_df.apply(lambda row: key_map[(row.side, row.limb, row.type)], axis=1)
        get_data_info(data_df, "new")

        self.all_data_df = pd.concat([self.all_data_df, data_df], ignore_index=True)
        get_data_info(self.all_data_df, "all")

    def reset(self):
        self.all_data_df = pd.DataFrame()

