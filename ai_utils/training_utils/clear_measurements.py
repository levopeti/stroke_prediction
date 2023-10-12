import json
import os
import random
import pandas as pd

from glob import glob

# from pympler.asizeof import asizeof

from measurement_utils.measure_db import MeasureDB

class ClearMeasurements(object):
    def __init__(self,
                 measDB: MeasureDB,
                 folder_path: str,
                 clear_json_path: str,
                 cache_size: int = 1) -> None:
        assert cache_size > 0, "cache_size must be positive integer"
        self.measDB = measDB
        self.cache_size = cache_size
        self.id_path_dict = dict()
        self.cache_dict = dict()
        self.clear_ids_dict = dict()
        self.healthy_ids = list()

        self.current_meas_id = None
        self.current_df = None

        self.read_csv_path(folder_path)
        self.read_clear_json(clear_json_path)
        self.collect_healthy_ids()

    def get_meas_id_list(self, data_type: str) -> list:
        return sorted(self.clear_ids_dict[data_type])

    def get_class_value_dict(self, meas_id: int)-> dict:
        class_value_dict = self.measDB.get_class_value_dict(meas_id=meas_id)
        return class_value_dict

    def get_min_class_value(self, meas_id: int)-> int:
        class_value_dict = self.measDB.get_class_value_dict(meas_id=meas_id)
        min_class_value = min(class_value_dict.values())
        return min_class_value

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

    def drop_random_from_cache_dict(self)-> None:
        self.cache_dict.pop(random.choice(list(self.cache_dict.keys())))

    def get_measurement(self, meas_id: int) -> pd.DataFrame:
        csv_path = self.id_path_dict[meas_id]
        if self.cache_size == 1:
            if meas_id == self.current_meas_id:
                df = self.current_df
            else:
                df = pd.read_csv(csv_path)
                self.current_df = df
                self.current_meas_id = meas_id
        else:
            if meas_id in self.cache_dict:
                df = self.cache_dict[meas_id]
            else:
                while len(self.cache_dict) >= self.cache_size:
                    self.drop_random_from_cache_dict()
                df = pd.read_csv(csv_path)
                self.cache_dict[meas_id] = df
                # assert len(self.cache_dict) <= self.cache_size, (len(self.cache_dict), self.cache_size)
                # if len(self.cache_dict) > self.cache_size:
                #     print("Number of cached measurements ({}) is more than the cache size ({})".format(len(self.cache_dict), self.cache_size))
        # print(" {}: {:.2f}".format(type(self).__name__, asizeof(self) / 1e6))
        return df

    def collect_healthy_ids(self) -> None:
        for meas_id in self.clear_ids_dict["train"]:
            min_class_value = self.get_min_class_value(meas_id)

            if min_class_value == 5:
                self.healthy_ids.append(meas_id)

    def print_stat(self)-> None:
        stat_dict = dict()
        for type_of_set, id_list in self.clear_ids_dict.items():
            stat_dict[type_of_set] = {class_value: 0 for class_value in range(6)}

            for meas_id in id_list:
                min_class_value = self.get_min_class_value(meas_id)
                stat_dict[type_of_set][min_class_value] += 1

        for type_of_set, class_value_dict in stat_dict.items():
            total = sum(class_value_dict.values())
            print("\n", type_of_set)
            for class_value in range(6):
                print("{}: {} {:.1f}%".format(class_value,
                                          class_value_dict[class_value],
                                          100 * class_value_dict[class_value] / total))