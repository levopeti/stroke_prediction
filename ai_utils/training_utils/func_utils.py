import random

import numpy as np
import pandas as pd


def calculate_diff(x_y_z, meas_type_acc) -> np.ndarray:
    if meas_type_acc:
        x_diff, y_diff, z_diff = [np.diff(m) for m in x_y_z]
    else:
        x_diff, y_diff, z_diff = x_y_z

    result = np.abs(x_diff) + np.abs(y_diff) + np.abs(z_diff)
    assert len(result) > 0
    return result

def get_diff(x_y_z, meas_type_acc, length=None, start_idx=None):
    cut_x_y_z = list()
    for array in x_y_z:
      if length is not None:
          assert length < len(array)

          start_idx = start_idx if start_idx is not None else random.randint(0, len(array) - (length + 1))
          if start_idx > len(array) - (length + 1):
              raise ValueError("start_idx is too large")

          cut_x_y_z.append(array[start_idx:start_idx + length])
      else:
          cut_x_y_z = x_y_z

    result = calculate_diff(cut_x_y_z, meas_type_acc)

    assert len(result) > 0
    return result

def get_limb_diff_mean(left_x_y_z, right_x_y_z, meas_type_acc, length=None, start_idx=None):
    left_diff = get_diff(left_x_y_z, meas_type_acc, length, start_idx)
    right_diff = get_diff(right_x_y_z, meas_type_acc, length, start_idx)
    result = np.abs(left_diff.mean() - right_diff.mean())
    return result

def get_limb_ratio_mean(left_x_y_z, right_x_y_z, meas_type_acc,
                        class_value_left, class_value_right,
                        length=None, start_idx=None, mean_first=True):
    left_diff = get_diff(left_x_y_z, meas_type_acc, length, start_idx)
    right_diff = get_diff(right_x_y_z, meas_type_acc, length, start_idx)

    if mean_first:
        if class_value_left > class_value_right:
            result = left_diff.sum() / right_diff.sum()
        else:
            result = right_diff.sum() / left_diff.sum()
    else:
        left_diff = left_diff + 0.1
        right_diff = right_diff + 0.1
        if class_value_left > class_value_right:
            result = np.mean(left_diff / right_diff)
        else:
            result = np.mean(right_diff / left_diff)
    return result

def get_input_from_df(meas_df: pd.DataFrame, length: int, class_value_dict: dict) -> np.ndarray:
    keys_in_order = (("arm", "acc"),
                     ("leg", "acc"),
                     ("arm", "gyr"),
                     ("leg", "gyr"))

    result = list()
    for key in keys_in_order:
        class_value_left = class_value_dict[("left", key[0])]
        class_value_right = class_value_dict[("right", key[0])]

        left_x_y_z = [meas_df[str(("left", key[0], key[1], "x"))].values,
                      meas_df[str(("left", key[0], key[1], "y"))].values,
                      meas_df[str(("left", key[0], key[1], "z"))].values]

        right_x_y_z = [meas_df[str(("right", key[0], key[1], "x"))].values,
                       meas_df[str(("right", key[0], key[1], "y"))].values,
                       meas_df[str(("right", key[0], key[1], "z"))].values]
        meas_type_acc = key[1] == "acc"
        diff_mean = get_limb_diff_mean(left_x_y_z, right_x_y_z, meas_type_acc, length, start_idx=None)
        ratio_mean_first = get_limb_ratio_mean(left_x_y_z, right_x_y_z, meas_type_acc, class_value_left, class_value_right, length, mean_first=True, start_idx=None)
        ratio_mean = get_limb_ratio_mean(left_x_y_z, right_x_y_z, meas_type_acc, class_value_left, class_value_right, length, mean_first=False, start_idx=None)
        result.extend([diff_mean, ratio_mean, ratio_mean_first])

    return np.expand_dims(np.array(result), axis=0)