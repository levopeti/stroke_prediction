import numpy as np

from scipy import signal
from scipy.ndimage import uniform_filter1d
from typing import Dict, Union


def get_3d_arrays_from_df(meas_dict: dict) -> Dict[tuple, np.ndarray]:
    """
    :param meas_dict: dict of pandas dataframe
    :return: dict, keys: (side, limb, meas_type), values: numpy array with shape (length, 3)
    """
    array_3d_dict = dict()
    for side, limb, meas_type in meas_dict.keys():
        result_array = np.concatenate(
            [np.expand_dims(meas_dict[(side, limb, meas_type)]["x"].values, axis=1),
             np.expand_dims(meas_dict[(side, limb, meas_type)]["y"].values, axis=1),
             np.expand_dims(meas_dict[(side, limb, meas_type)]["z"].values, axis=1)], axis=1)
        array_3d_dict[(side, limb, meas_type)] = result_array
    return array_3d_dict


def butter_high_pass_filter(meas_3d_arrays: Dict[tuple, np.ndarray]) -> Dict[tuple, np.ndarray]:
    """ fifth-order Butterworth filter with a cut-off frequency of 3 Hz """
    sos = signal.butter(5, 3, "highpass", output="sos", fs=25)
    for key, array_3d in meas_3d_arrays.items():
        filtered_list = list()
        if key[2] == "a":
            for i in range(3):
                sig = array_3d[:, i]
                filtered_list.append(np.expand_dims(signal.sosfilt(sos, sig), axis=1))
            meas_3d_arrays[key] = np.concatenate(filtered_list, axis=1)
    return meas_3d_arrays


def cut_array_to_length(meas_3d_arrays: Dict[tuple, np.ndarray],
                        length: int,
                        start_idx: int) -> Dict[tuple, np.ndarray]:
    cut_array_dict = dict()
    for key, array_3d in meas_3d_arrays.items():
        assert length < len(array_3d), (length, len(array_3d))
        if start_idx > len(array_3d) - length:
            raise ValueError("start_idx is too large {} {} {} {}".format(start_idx, length, start_idx + length, len(array_3d)))

        cut_array_dict[key] = array_3d[start_idx:start_idx + length]
        assert len(cut_array_dict[key]) == length, (len(cut_array_dict[key]), length)
    return cut_array_dict


def calculate_euclidean_length(meas_3d_arrays: Dict[tuple, np.ndarray]) -> Dict[tuple, np.ndarray]:
    euclidean_length_dict = {key: np.linalg.norm(array_3d, ord=2, axis=1) for key, array_3d in meas_3d_arrays.items()}
    return euclidean_length_dict


def clip_values(meas_1d_arrays: Dict[tuple, np.ndarray], max_value: int, meas_type: str) -> Dict[tuple, np.ndarray]:
    assert meas_type in ["a", "g"], meas_type
    for key, array_1d in meas_1d_arrays.items():
        if key[2] == meas_type:
            clipped_array_1d = np.clip(array_1d, 0, max_value)
            meas_1d_arrays[key] = clipped_array_1d
    return meas_1d_arrays


def divide_values(meas_1d_arrays: Dict[tuple, np.ndarray],
                  div_value: Union[int, float],
                  meas_type: str) -> Dict[tuple, np.ndarray]:
    assert meas_type in ["a", "g"], meas_type
    for key, array_1d in meas_1d_arrays.items():
        if key[2] == meas_type:
            new_array_1d = array_1d / div_value
            meas_1d_arrays[key] = new_array_1d
    return meas_1d_arrays


def moving_average(meas_1d_arrays: Dict[tuple, np.ndarray], window_size: int) -> Dict[tuple, np.ndarray]:
    for key, array_1d in meas_1d_arrays.items():
        averaged_array = uniform_filter1d(array_1d, size=window_size, mode="constant", cval=0.0)
        meas_1d_arrays[key] = averaged_array
    return meas_1d_arrays


def down_sampling(meas_1d_arrays: Dict[tuple, np.ndarray], subsampling_factor: int):
    # 135000 / 50 = 2700 (90 min)
    for key, array_1d in meas_1d_arrays.items():
        meas_1d_arrays[key] = array_1d[::subsampling_factor]
    return meas_1d_arrays


def create_multivariate_time_series(meas_arrays: Dict[tuple, np.ndarray]) -> np.ndarray:
    return np.concatenate([np.expand_dims(array, axis=0) for array in meas_arrays.values()], axis=0)


