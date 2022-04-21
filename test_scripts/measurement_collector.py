import pandas as pd
import os
import glob
import random
from termcolor import colored
from tqdm import tqdm

from accdb_reading import get_measure_df
from measurement import Measurement


class MeasurementCollector(object):
    def __init__(self, base_path, db_path, m_path, synchronizing=True, lightweight=False):
        self.lightweight = lightweight
        self.synchronizing = synchronizing
        self.dict_of_df = get_measure_df(db_path, write=False)
        self.aux_data = pd.read_excel(m_path)
        self.aux_data["Measure ID"] = pd.to_numeric(self.aux_data["Measure ID"], downcast='integer')
        # self.aux_data["The last sensor is on the patient"] = pd.to_datetime(self.aux_data["The last sensor is on the patient"],
        #                                                                     format='%Y-%m-%dT%H:%M:%S.%f')
        # self.aux_data["Take off the first sensor"] = pd.to_datetime(self.aux_data["Take off the first sensor"],
        #                                                             format='%Y-%m-%dT%H:%M:%S.%f')
        self.measurement_dict = dict()
        self.collect_measurment(base_path)
        self.print_statistics()

    def collect_measurment(self, base_path):
        # TODO: from one drive
        for row_id, measurement_name in enumerate(self.dict_of_df["Z_1ÁLTALÁNOS"]["VizsgAz"]):
            self.measurement_dict[measurement_name] = Measurement(measurement_name, row_id,
                                                                  self.dict_of_df["Z_3NEUROLÓGIA"],
                                                                  synchronizing=self.synchronizing,
                                                                  lightweight=self.lightweight)
            for path in glob.glob(base_path + "/*/*.csv"):
                if path.split('/')[-1].find(str(measurement_name)) == 0:
                    self.measurement_dict[measurement_name].add_measurement_path(path)

            if measurement_name in self.aux_data["Measure ID"].values:
                self.measurement_dict[measurement_name].add_aux_data(self.aux_data.loc[self.aux_data["Measure ID"] == measurement_name])
            else:
                print(colored("{} is not found in aux data".format(measurement_name), "red"))

            self.measurement_dict[measurement_name].check_measurement_path_dict()
            self.measurement_dict[measurement_name].check_five_class()

            if not self.measurement_dict[measurement_name].valid:
                print(colored("{} is not valid (deleted)".format(measurement_name), "red"))
                del self.measurement_dict[measurement_name]

    def print_statistics(self):
        print(colored("Number of measurements {}".format(len(self.measurement_dict)), "blue"))

        stat_dict = {i: 0 for i in range(6)}
        for meas in self.measurement_dict.values():
            stat_dict[meas.get_absolute_class_value()] = stat_dict[meas.get_absolute_class_value()] + 1

        for k, v in stat_dict.items():
            print(colored("Number of {}: {} ({:.2f} %)".format(k, v, 100 * v / len(self.measurement_dict)), "blue"))

    def get_measurement_df(self, measurement_name, key=None, only_valid=True):
        if key is None:
            return self.measurement_dict[measurement_name].get_all_measurements_df(only_valid=only_valid)
        else:
            return self.measurement_dict[measurement_name].get_measurement_df(key, only_valid=only_valid)

    def get_all_valid_measurement_df(self):
        result_dict = dict()

        for m_name, meas in self.measurement_dict.items():
            if meas.valid:
                result_dict[m_name] = meas.get_all_measurements_df()

        return result_dict

    def get_diff(self, measurement_name, key):
        return self.measurement_dict[measurement_name].get_diff(key)

    def get_all_diff_df(self):
        result_dict = dict()

        for m_name, meas in self.measurement_dict.items():
            if meas.valid:
                result_dict[m_name] = meas.get_all_diff()

        return result_dict

    def get_class_value(self, measurement_name, key):
        return self.measurement_dict[measurement_name].get_class_value(key)

    def get_random_diff_with_class(self, length, meas_type="acc"):
        meas = random.choice(list(self.measurement_dict.values()))
        key = random.choice(list(meas.measurement_dict.keys()))
        key = (key[0], key[1], meas_type)

        random_diff = meas.get_diff(key, length)

        return random_diff, meas.get_class_value(key),

    def get_random_limb_diff_mean_with_class(self, limb, meas_type="acc", length=None):
        meas = random.choice(list(self.measurement_dict.values()))
        random_diff_mean = meas.get_limb_diff_mean(limb, meas_type, length)

        return random_diff_mean, meas.get_limb_class_value(limb)

    def get_random_diff_mean_with_class_all(self, length=None):
        keys_in_order = (("arm", "acc"),
                         ("leg", "acc"),
                         ("arm", "gyr"),
                         ("leg", "gyr"))
        meas = random.choice(list(self.measurement_dict.values()))

        random_diff_dict = dict()
        for key in keys_in_order:
            diff_mean, _ = meas.get_limb_diff_mean(key[0], key[1], length)
            random_diff_dict[key] = diff_mean

        class_value = meas.get_absolute_class_value()
        return random_diff_dict, class_value

    def get_random_ratio_mean_with_class_all(self, length=None, mean_first=True):
        keys_in_order = (("arm", "acc"),
                         ("leg", "acc"),
                         ("arm", "gyr"),
                         ("leg", "gyr"))
        meas = random.choice(list(self.measurement_dict.values()))

        random_ratio_dict = dict()
        for key in keys_in_order:
            ratio_mean, _ = meas.get_limb_ratio_mean(key[0], key[1], length, mean_first=mean_first)
            random_ratio_dict[key] = ratio_mean

        class_value = meas.get_absolute_class_value()
        return random_ratio_dict, class_value

    def get_random_mean_with_class_all(self, mean_type='all', length=None, mean_first=True):
        assert mean_type in ['all', 'diff', 'ratio']
        keys_in_order = (("arm", "acc"),
                         ("leg", "acc"),
                         ("arm", "gyr"),
                         ("leg", "gyr"))
        meas = random.choice(list(self.measurement_dict.values()))

        random_mean_dict = dict()
        for key in keys_in_order:
            if mean_type == 'diff':
                diff_mean, _ = meas.get_limb_diff_mean(key[0], key[1], length)
                random_mean_dict[key] = diff_mean
            elif mean_type == 'ratio':
                ratio_mean, _ = meas.get_limb_ratio_mean(key[0], key[1], length, mean_first=mean_first)
                random_mean_dict[key] = ratio_mean
            else:
                diff_mean, _ = meas.get_limb_diff_mean(key[0], key[1], length)
                ratio_mean_first, _ = meas.get_limb_ratio_mean(key[0], key[1], length, mean_first=True)
                ratio_mean, _ = meas.get_limb_ratio_mean(key[0], key[1], length, mean_first=False)
                random_mean_dict[key] = [diff_mean, ratio_mean_first, ratio_mean]

        class_value = meas.get_absolute_class_value()
        return random_mean_dict, class_value

    def sweep_diff_generator(self, length, mean=False, meas_type_first=True):
        keys_in_order = (("left", "arm", "acc"),
                         ("right", "arm", "acc"),
                         ("left", "leg", "acc"),
                         ("right", "leg", "acc"),
                         ("left", "arm", "gyr"),
                         ("right", "arm", "gyr"),
                         ("left", "leg", "gyr"),
                         ("right", "leg", "gyr"))

        for meas_name, meas in self.measurement_dict.items():
            for key in keys_in_order:
                diff_list = meas.sweep_diff(key, length=length, mean=mean)
                assert len(diff_list) > 0
                yield meas_name, key, meas.get_class_value(key), diff_list

    def sweep_mean_with_class_generator(self, mean_type='all', length=None, step_size=1, mean_first=True):
        assert mean_type in ['all', 'diff', 'ratio']
        keys_in_order = (("arm", "acc"),
                         ("leg", "acc"),
                         ("arm", "gyr"),
                         ("leg", "gyr"))

        for meas_name, meas in self.measurement_dict.items():
            class_value = meas.get_absolute_class_value()
            # mean_dict = {key: list() for key in keys_in_order}
            mean_dict = dict()
            start_idx = 0

            while True:
                try:
                    for key in keys_in_order:
                        if mean_type == 'diff':
                            diff_mean, _ = meas.get_limb_diff_mean(key[0], key[1], length, start_idx)
                            # mean_dict[key].append(diff_mean)
                            mean_dict[key] = diff_mean
                        elif mean_type == 'ratio':
                            ratio_mean, _ = meas.get_limb_ratio_mean(key[0], key[1], length, mean_first=mean_first,
                                                                     start_idx=start_idx)
                            # mean_dict[key].append(ratio_mean)
                            mean_dict[key] = ratio_mean
                        else:
                            diff_mean, _ = meas.get_limb_diff_mean(key[0], key[1], length, start_idx)
                            ratio_mean_first, _ = meas.get_limb_ratio_mean(key[0], key[1], length, mean_first=True,
                                                                           start_idx=start_idx)
                            ratio_mean, _ = meas.get_limb_ratio_mean(key[0], key[1], length, mean_first=False,
                                                                     start_idx=start_idx)
                            # mean_dict[key].append([diff_mean, ratio_mean_first, ratio_mean])
                            mean_dict[key] = [diff_mean, ratio_mean_first, ratio_mean]
                    yield mean_dict, class_value, meas_name
                except ValueError:
                    break
                start_idx = start_idx + step_size





if __name__ == "__main__":
    _db_path = "/home/levcsi/projects/stroke_prediction/data/WUS-v4meresek 20220202.accdb"
    _m_path = "/home/levcsi/projects/stroke_prediction/data/biocal.xlsx"
    mc = MeasurementCollector('/home/levcsi/projects/stroke_prediction/data', _db_path, _m_path,
                              synchronizing=True)

    # df = mc.get_measurement_df("1meresjenei", ("right", "leg", "acc"))
    # result_list = list()

    limb_dict = {"arm": 1, "leg": 2}
    meas_type_dict = {"acc": 1, "gyr": 2}
    num_sample = 50
    time_length = 25 * 60 * 90

    mc.get_measurement_df(202201310, only_valid=True)
    # meas_df = mc.get_random_ratio_mean_with_class_all(length=time_length, mean_first=False)
    # print(meas_df)
    # for _k, _df in meas_df.items():
    #     print(_k)
    #     print(len(_df))
    #     print()