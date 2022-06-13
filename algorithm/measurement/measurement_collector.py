import pandas as pd
import os
import glob
import random
import traceback
from termcolor import colored

from .accdb_reading import get_measure_df
from .measurement import Measurement


class MeasurementCollector(object):
    def __init__(self, base_path, db_path, m_path, ucanaccess_path, synchronizing=True, lightweight=False):
        self.lightweight = lightweight
        self.synchronizing = synchronizing
        self.dict_of_df = get_measure_df(ucanaccess_path, db_path, write=False)
        self.aux_data = pd.read_excel(m_path)
        self.aux_data["Measure ID"] = pd.to_numeric(self.aux_data["Measure ID"], downcast='integer')
        # self.aux_data["The last sensor is on the patient"] = pd.to_datetime(self.aux_data["The last sensor is on the patient"],
        #                                                                     format='%Y-%m-%dT%H:%M:%S.%f')
        # self.aux_data["Take off the first sensor"] = pd.to_datetime(self.aux_data["Take off the first sensor"],
        #                                                             format='%Y-%m-%dT%H:%M:%S.%f')
        self.measurement_dict = {
            "train": dict(),
            "test": dict(),
            "mixed": dict()
        }
        self.collect_measurment(os.path.join(base_path, "train"), "train")
        self.collect_measurment(os.path.join(base_path, "test"), "test")
        self.create_mixed_measurment(self.measurement_dict["train"][202112020], self.measurement_dict["train"][202112171])
        self.print_statistics()

    def create_mixed_measurment(self, meas_1, meas_2, ratio=0.5):
        meas_name = "mixed_1"
        all_meas_1 = meas_1.get_all_measurements_df()
        all_meas_2 = meas_2.get_all_measurements_df()

        mixed_meas = Measurement(meas_name, None, None, synchronizing=self.synchronizing, lightweight=self.lightweight,
                                 ratio=ratio)

        for k in all_meas_1.keys():
            chunk_1 = all_meas_1[k].iloc[:int(len(all_meas_1[k]) * ratio), :]
            chunk_2 = all_meas_2[k].iloc[int(len(all_meas_2[k]) * (1- ratio)):, :]
            mixed_df = pd.concat([chunk_1, chunk_2])

            mixed_meas.measurement_dict[k] = mixed_df

            if mixed_meas.length is None:
                mixed_meas.length = (len(chunk_1), len(chunk_2))
            else:
                assert mixed_meas.length == (len(chunk_1), len(chunk_2)), (mixed_meas.length, (len(chunk_1), len(chunk_2)))

        for k in meas_1.class_value_dict.keys():
            mixed_meas.class_value_dict[k] = (meas_1.class_value_dict[k], meas_2.class_value_dict[k])

        self.measurement_dict["mixed"][meas_name] = mixed_meas

    def collect_measurment(self, base_path, type_of_set="train"):
        print("\n##### Load measurements for {} #####".format(type_of_set.upper()))
        for row_id, measurement_name in enumerate(self.dict_of_df["Z_1ÁLTALÁNOS"]["VizsgAz"].values):
            meas = Measurement(measurement_name, row_id, self.dict_of_df["Z_3NEUROLÓGIA"],
                               synchronizing=self.synchronizing, lightweight=self.lightweight)

            for path in glob.glob(base_path + "/*/*.csv"):
                if path.split('/')[-1].find(str(measurement_name)) == 0:
                    meas.add_measurement_path(path)

            if measurement_name in self.aux_data["Measure ID"].values:
                meas.add_aux_data(self.aux_data.loc[self.aux_data["Measure ID"] == measurement_name])
            else:
                meas.log_list.append(colored("measurement is not found in aux data", "red"))

            meas.check_measurement_path_dict()
            meas.check_five_class()
            meas.print_log()

            if not meas.valid:
                print(colored("measurement is not valid (deleted)", "red"))
            else:
                print(colored("measurement_path_dict is OK", "green"))
                self.measurement_dict[type_of_set][measurement_name] = meas

    def print_statistics(self):
        for type_of_set, meas_dict in self.measurement_dict.items():
            print(colored("\nType of set: {}".format(type_of_set), "blue"))
            print(colored("Number of measurements {}".format(len(meas_dict)), "blue"))

            if len(meas_dict) > 0:
                stat_dict = dict()
                for meas in meas_dict.values():
                    abs_class_value = meas.get_absolute_class_value()

                    if abs_class_value not in stat_dict:
                        stat_dict[abs_class_value] = 0
                    stat_dict[abs_class_value] = stat_dict[abs_class_value] + 1

                for k, v in stat_dict.items():
                    print(colored("Number of {}: {} ({:.2f} %)".format(k, v, 100 * v / len(meas_dict)), "blue"))

    def get_random_mean_with_class_all(self, mean_type='all', limb='all', length=None, mean_first=True,
                                       type_of_set="train"):
        assert mean_type in ['all', 'diff', 'ratio']
        assert limb in ["all", "arm", "leg"]
        assert type_of_set in ["train", "test", "mixed"]
        keys_in_order = (("arm", "acc"),
                         ("leg", "acc"),
                         ("arm", "gyr"),
                         ("leg", "gyr"))
        meas = random.choice(list(self.measurement_dict[type_of_set].values()))

        try:
            random_mean_dict = dict()
            for key in keys_in_order:
                if limb != "all" and key[0] != limb:
                    continue

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
        except Exception as e:
            print(meas.info)
            traceback.print_exc()
            print(e)
            exit()
        return random_mean_dict, class_value

    def sweep_mean_with_class_generator(self, mean_type='all', limb='all', length=None, step_size=1, mean_first=True,
                                        type_of_set="train"):
        assert mean_type in ['all', 'diff', 'ratio']
        keys_in_order = (("arm", "acc"),
                         ("leg", "acc"),
                         ("arm", "gyr"),
                         ("leg", "gyr"))

        for meas_name, meas in self.measurement_dict[type_of_set].items():
            # mean_dict = {key: list() for key in keys_in_order}
            mean_dict = dict()
            start_idx = 0

            while True:
                try:
                    for key in keys_in_order:
                        if limb != "all" and key[0] != limb:
                            continue
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
                    class_value = meas.get_absolute_class_value(start_idx, length)
                    yield mean_dict, class_value, meas_name
                except ValueError:
                    break
                start_idx = start_idx + step_size

    ##############
    # deprecated #
    ##############

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

    def sweep_diff_generator(self, length, mean=False):
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


if __name__ == "__main__":
    _db_path = "/home/levcsi/projects/stroke_prediction/data/WUS-v4meresek 20220302.accdb"
    _m_path = "/data/biocal.xlsx"
    mc = MeasurementCollector('/data', _db_path, _m_path,
                              synchronizing=True)

    # df = mc.get_measurement_df("1meresjenei", ("right", "leg", "acc"))
    # result_list = list()

    limb_dict = {"arm": 1, "leg": 2}
    meas_type_dict = {"acc": 1, "gyr": 2}
    num_sample = 50
    time_length = 25 * 60 * 90

    # mc.get_measurement_df(202201310, only_valid=True)
    # meas_df = mc.get_random_ratio_mean_with_class_all(length=time_length, mean_first=False)
    # print(meas_df)
    # for _k, _df in meas_df.items():
    #     print(_k)
    #     print(len(_df))
    #     print()
