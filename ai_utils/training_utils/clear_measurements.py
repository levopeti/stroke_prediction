import json
import os
import random
import pandas as pd

from glob import glob


class ClearMeasurements(object):
    def __init__(self, folder_path: str, clear_json_path: str, cache_size: int = 1) -> None:
        assert cache_size > 0, "cache_size must be positive integer"
        self.cache_size = cache_size
        self.id_path_dict = dict()
        self.cache_dict = dict()
        self.clear_ids_dict = dict()

        self.read_csv_path(folder_path)
        self.read_clear_json(clear_json_path)

    def get_meas_id_list(self, data_type: str) -> list:
        return sorted(self.clear_ids_dict[data_type])

    def read_clear_json(self, clear_json_path: str) -> None:
        with open(clear_json_path, "r") as read_file:
            self.clear_ids_dict = json.load(read_file)

        all_meas_ids = set(self.id_path_dict.keys())
        for meas_id in self.clear_ids_dict["train"]:
            assert meas_id in all_meas_ids

        for meas_id in self.clear_ids_dict["test"]:
            assert meas_id in all_meas_ids

    def read_csv_path(self, folder_path: str) -> None:
        for csv_path in sorted(glob(os.path.join(folder_path, "*.csv"))):
            file_name = os.path.basename(csv_path)
            meas_id = file_name.split("-")[0]
            self.id_path_dict[int(meas_id)] = csv_path

    def drop_random_from_cache_dict(self):
        self.cache_dict.pop(random.choice(list(self.cache_dict.keys())))

    def get_measurement(self, meas_id: int) -> pd.DataFrame:
        if meas_id in self.cache_dict:
            #print("use cache")
            df = self.cache_dict[meas_id]
        else:
            #print("read new")
            while len(self.cache_dict) >= self.cache_size:
                # print("drop from cache")
                self.drop_random_from_cache_dict()

            csv_path = self.id_path_dict[meas_id]
            #print("read csv")
            df = pd.read_csv(csv_path)
            #print("done")
            self.cache_dict[meas_id] = df
            # assert len(self.cache_dict) <= self.cache_size, (len(self.cache_dict), self.cache_size)
            # if len(self.cache_dict) > self.cache_size:
            #     print("Number of cached measurements ({}) is more than the cache size ({})".format(len(self.cache_dict), self.cache_size))
        return df