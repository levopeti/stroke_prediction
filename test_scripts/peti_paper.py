import numpy as np

from pprint import pprint
from ai_utils.training_utils.clear_measurements import ClearMeasurements
from ai_utils.training_utils.func_utils import get_input_from_df
from measurement_utils.measure_db import MeasureDB

frequency = 25  # Hz

def get_acc_sum(meas_df):
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


if __name__ == "__main__":
    params = {"accdb_path": "./data/WUS-v4measure202307311.accdb",
              "ucanaccess_path": "./ucanaccess/",
              "folder_path": "./data/clear_data/",
              "clear_json_path": "./data/clear_train_test_ids.json",
              "length": int(1.5 * 60 * 60 * 25),  # 1.5 hours, 25 Hz
              }

    measDB = MeasureDB(params["accdb_path"], params["ucanaccess_path"])
    clear_measurements = ClearMeasurements(measDB, params["folder_path"], params["clear_json_path"])
    # clear_measurements.print_stat()

    id_list = clear_measurements.collect_healthy_ids("all")

    print(len(id_list))
    print(id_list)

    meas_acc_sum_dict = {("left", "arm"): [0, 0],
                         ("left", "leg"): [0, 0],
                         ("right", "arm"): [0, 0],
                         ("right", "leg"): [0, 0]
                        }
    for meas_id in id_list:
        _meas_df = clear_measurements.get_measurement(meas_id)
        _acc_sum_dict = get_acc_sum(_meas_df)

        for _key in meas_acc_sum_dict.keys():
            # acc_sum
            meas_acc_sum_dict[_key][0] += _acc_sum_dict[_key][0]
            # num_of_sec
            meas_acc_sum_dict[_key][1] += _acc_sum_dict[_key][1]

    all_acc_sum = 0
    all_sec = 0
    for _key, value in meas_acc_sum_dict.items():
        print("{}: {:.2f}".format(_key, value[0] / value[1]))
        all_acc_sum += value[0]
        all_sec += value[1]

    print("all: {:.2f}".format(all_acc_sum / all_sec))
