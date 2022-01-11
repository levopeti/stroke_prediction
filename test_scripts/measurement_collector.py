import pandas as pd
import os
import glob
from termcolor import colored

from accdb_reading import get_measure_df
from measurement import Measurement


class MeasurementCollector(object):
    def __init__(self, base_path, db_path, m_path):
        self.dict_of_df = get_measure_df(db_path, write=False)
        self.aux_data = pd.read_csv(m_path)
        self.measurement_dict = dict()
        self.collect_measurment(base_path)

    def collect_measurment(self, base_path):
        # TODO: from one drive
        for row_id, measurement_name in enumerate(self.dict_of_df["Z_1ÁLTALÁNOS"]["Név"]):
            self.measurement_dict[measurement_name] = Measurement(measurement_name, row_id, lightweight=True)
            for path in glob.glob(base_path + "/*/*.csv"):
                if path.split('/')[-1].find(measurement_name) == 0:
                    self.measurement_dict[measurement_name].add_measurement_path(path)

            self.measurement_dict[measurement_name].check_measurement_path_dict()
            if measurement_name in self.aux_data.columns:
                self.measurement_dict[measurement_name].add_aux_data(self.aux_data[measurement_name])
            else:
                print(colored("{} is not found in aux data".format(measurement_name), "red"))

    def get_measurement_df(self, measurement_name, key):
        return self.measurement_dict[measurement_name].get_measurement_df(key)

    def get_all_valid_measurement_df(self):
        result_dict = dict()

        for m_name, meas in self.measurement_dict.items():
            if meas.valid:
                result_dict[m_name] = meas.get_all_measurements_df()

        return result_dict


if __name__ == "__main__":
    _db_path = "/home/levcsi/projects/stroke_prediction/data/WUS-v4meresek_20211220.accdb"
    _m_path = "/home/levcsi/projects/stroke_prediction/data/Meres_masolata.csv"
    mc = MeasurementCollector('/home/levcsi/projects/stroke_prediction/data', _db_path, _m_path)

    df = mc.get_measurement_df("1meresjenei", ("right", "leg", "acc"))
    print(df.head())
