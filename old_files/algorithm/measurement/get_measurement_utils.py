import pandas as pd

from typing import Union
from termcolor import colored


def min_to_millisec(minute: Union[int, float]) -> int:
    return int(minute * 60 * 1000)


def cut_valid_part(_meas_df, meas, only_valid):
    if only_valid:
        if meas.valid_start_time is not None:
            # _meas_df = _meas_df[_meas_df["epoch"] > self.valid_start_time.timestamp() * 1000]
            _meas_df = _meas_df[_meas_df["epoch"] > meas.valid_start_time]
        if meas.valid_end_time is not None:
            # _meas_df = _meas_df[_meas_df["epoch"] < self.valid_end_time.timestamp() * 1000]
            _meas_df = _meas_df[_meas_df["epoch"] < meas.valid_end_time]
    return _meas_df


def read_csv(_key, meas, columns_key_dict, only_valid):
    try:
        _meas_df = pd.read_csv(meas.measurement_path_dict[_key])  # , usecols=columns_key_dict[_key[2]])
        column_mask = _meas_df.columns.str.contains("axis|epoc")
        _meas_df = _meas_df[_meas_df.columns[column_mask]]
    except ValueError:
        _meas_df = pd.read_csv(meas.measurement_path_dict[_key])
        print(columns_key_dict[_key[2]])
        raise ValueError("{} could not be loaded because of columns:\n{}\nexpected: {}".format(meas.measurement_name, _meas_df.columns, columns_key_dict[_key[2]]))

    for c_name in _meas_df.columns:
        _meas_df.rename(columns={c_name: c_name.split(' ')[0]}, inplace=True)

    _meas_df.rename(columns={"epoc": "epoch"}, inplace=True)
    _meas_df = cut_valid_part(_meas_df, meas, only_valid)
    return _meas_df


def cut_for_mutual_part(_measurement_dict):
    min_ts = 0
    max_ts = float('inf')

    for meas in _measurement_dict.values():
        if meas["epoch"].min() > min_ts:
            min_ts = meas["epoch"].min()

        if meas["epoch"].max() < max_ts:
            max_ts = meas["epoch"].max()

    for _k, meas in _measurement_dict.items():
        # print(len(meas[(meas["epoch"] >= min_ts) & (meas["epoch"] <= max_ts)]))
        _measurement_dict[_k] = meas[(meas["epoch"] >= min_ts) & (meas["epoch"] <= max_ts)]

    return _measurement_dict


def synchronize(_measurement_dict, meas):
    _measurement_dict = cut_for_mutual_part(_measurement_dict)

    base_df = None
    for _k, _df in _measurement_dict.items():
        if base_df is None:
            base_df = _df.sort_values('epoch')
        else:
            _df = _df.sort_values('epoch')
            merged_df = pd.merge_asof(base_df, _df, on="epoch", tolerance=40, direction='nearest')
            assert merged_df.isna().sum().sum() == 0

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
            print(colored("zero length of data {}, {}".format(meas.measurement_name, _k), "red"))
    return _measurement_dict