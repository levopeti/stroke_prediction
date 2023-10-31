from datetime import datetime

import numpy as np

from pprint import pprint
from ai_utils.training_utils.clear_measurements import ClearMeasurements
from ai_utils.training_utils.func_utils import get_input_from_df
from measurement_utils.measure_db import MeasureDB

frequency = 25  # Hz


def get_acc_sum(meas_df, _min_h=None, _max_h=None):
    meas_df["hour"] = meas_df.epoch.apply(lambda x: datetime.fromtimestamp(x / 1000).hour)

    if _min_h is not None and _max_h is not None:
        if _min_h < _max_h:
            meas_df = meas_df[(meas_df["hour"] >= _min_h) & (meas_df["hour"] < _max_h)]
        else:
            meas_df = meas_df[(meas_df["hour"] >= _min_h) | (meas_df["hour"] < _max_h)]

    keys_in_order = (("arm", "acc"),
                     ("leg", "acc"))

    acc_sum_dict = dict()
    for key in keys_in_order:
        # class_value_left = class_value_dict[("left", key[0])]
        # class_value_right = class_value_dict[("right", key[0])]

        for side in ["left", "right"]:
            x_y_z = [meas_df[str((side, key[0], key[1], "x"))].values,
                     meas_df[str((side, key[0], key[1], "y"))].values,
                     meas_df[str((side, key[0], key[1], "z"))].values]

            for array_idx in range(len(x_y_z)):
                x_y_z[array_idx] = np.expand_dims(x_y_z[array_idx], 0)

            x_y_z = np.concatenate(x_y_z, 0)
            x_y_z = np.abs(x_y_z)
            acc_sum = np.sum(x_y_z)
            num_of_sec = x_y_z.shape[1] / frequency

            acc_sum_dict[side, key[0]] = acc_sum, num_of_sec
    return acc_sum_dict


def get_id_list(_stroke_side, _stroke_limb, _threshold):
    _id_list = list()
    for meas_id in clear_measurements.all_meas_ids:
        class_value_dict = clear_measurements.get_class_value_dict(meas_id)

        values = list()
        for keys in class_value_dict.keys():
            if (keys[0] == _stroke_side or _stroke_side == "all") and (keys[1] == _stroke_limb or _stroke_limb == "all"):
                values.append(class_value_dict[keys])

        if min(values) < _threshold:
            _id_list.append(meas_id)
    return _id_list


def get_stat():
    print("stroke_side: {}, stroke_limb: {}, threshold: {}, min_h: {}, max_h: {}".format(stroke_side, stroke_limb, threshold, min_h, max_h))
    print(len(id_list))
    print(id_list)
    meas_acc_sum_dict = {("right", "arm"): [0, 0],
                         ("right", "leg"): [0, 0],
                         ("left", "arm"): [0, 0],
                         ("left", "leg"): [0, 0]
                         }
    for _meas_id in id_list:
        _meas_df = clear_measurements.get_measurement(_meas_id)
        acc_sum_dict = get_acc_sum(_meas_df, min_h, max_h)

        for _key in meas_acc_sum_dict.keys():
            # acc_sum
            meas_acc_sum_dict[_key][0] += acc_sum_dict[_key][0]
            # num_of_sec
            meas_acc_sum_dict[_key][1] += acc_sum_dict[_key][1]

    all_acc_sum = 0
    all_sec = 0
    result_string = ""
    for _key, value in meas_acc_sum_dict.items():
        print(_key)
        result_string += "{:.2f}\n\n".format(value[0] / value[1]).replace(".", ",")
        all_acc_sum += value[0]
        all_sec += value[1]

    print("all\n{:.2f}\n".format(all_acc_sum / all_sec).replace(".", ","))
    print(result_string)

if __name__ == "__main__":
    params = {"accdb_path": "./data/WUS-v4measure202307311.accdb",
              "ucanaccess_path": "./ucanaccess/",
              "folder_path": "./data/clear_data/",
              "clear_json_path": "./data/clear_train_test_ids.json",
              "length": int(1.5 * 60 * 60 * 25),  # 1.5 hours, 25 Hz
              }

    measDB = MeasureDB(params["accdb_path"], params["ucanaccess_path"])
    clear_measurements = ClearMeasurements(measDB, params["folder_path"], params["clear_json_path"])
    clear_measurements.print_stat()

    # id_list = clear_measurements.collect_healthy_ids("all")
    stroke_side = "all"  # right left all
    stroke_limb = "all"  # arm leg all
    threshold = 5
    min_h = 7
    max_h = 23
    # for stroke_limb in ["arm", "leg", "all"]:
    #     stroke_side_list = ["all"] if stroke_limb in ["arm", "leg"] else ["right", "left", "all"]
    #     for stroke_side in stroke_side_list:
    #         threshold_list = [5, 3] if stroke_limb == "all" and stroke_side != "all" else [3]
    #         for threshold in threshold_list:
    #             id_list = get_id_list(stroke_side, stroke_limb, threshold)
    #             get_stat()

    # id_list = get_id_list(stroke_side, stroke_limb, threshold)
    # get_stat()
    id_list = clear_measurements.all_meas_ids
    print(clear_measurements.all_meas_ids)

    for _meas_id in id_list:
        _meas_df = clear_measurements.get_measurement(_meas_id)
        _acc_sum_dict = get_acc_sum(_meas_df)
        print_list = [_acc_sum_dict[("right", "arm")][0] / _acc_sum_dict["right", "arm"][1],
                      _acc_sum_dict[("left", "arm")][0] / _acc_sum_dict["left", "arm"][1],
                      _acc_sum_dict[("right", "leg")][0] / _acc_sum_dict["right", "leg"][1],
                      _acc_sum_dict[("left", "leg")][0] / _acc_sum_dict["left", "leg"][1]]
        print("{},{:.2f},{:.2f},{:.2f},{:.2f}".format(_meas_id, *print_list))
