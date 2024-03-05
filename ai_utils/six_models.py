import os
import numpy as np
import torch

from pprint import pprint
from typing import Tuple
from glob import glob
from numpy.lib.stride_tricks import sliding_window_view

from ai_utils.data_preprocessing import get_3d_arrays_from_df, cut_array_to_length, calculate_euclidean_length, \
    moving_average, divide_values, down_sampling, butter_high_pass_filter, create_multivariate_time_series
from ai_utils.model_abstract import Model
from measurement_utils.measurement import Measurement
from utils.general_utils import min_to_ticks, sec_to_ticks

# TODO: r -> l
only_left_arm_key_list = [("r", "a", "a"),
                          ("r", "a", "g")]

# risk 2
"""avg_prob_threshold_dict = {"inverted": {30: 0.405, 60: 0.467, 90: 0.999},
                           "non-inverted": {30: 0.781, 60: 0.807, 90: 0.939}}
"""
avg_prob_threshold_dict = {"inverted": {30: 0.95, 60: 0.95, 90: 0.999},
                           "non-inverted": {30: 0.8, 60: 0.85, 90: 0.939}}

window_length_dict = {"inverted": {30: 10252, 60: 10549, 90: 10667},
                      "non-inverted": {30: 10774, 60: 10731, 90: 6697}}

combination_threshold = 1  # at least one is true from the 3 models
stroke_threshold = 1.5  # [0, 1] -> stroke, [2] -> healthy


class SixModels(Model):
    def __init__(self, config_dict: dict, to_cuda: int = False):
        self.frequency = config_dict["frequency"]
        self.batch_size = config_dict["batch_size"]
        self.step_size_sec = config_dict["step_size_sec"]
        self.to_cuda = to_cuda
        model_paths = glob(os.path.join(config_dict["model_folder"], "*.pt"))
        assert len(model_paths) == 6, model_paths

        self.model_dict = dict()
        for model_path in model_paths:
            model_name = model_path.split("/")[-1].split(".")[0]
            inverted, length_min, limb, date = model_name.split("_")
            model = torch.jit.load(model_path, map_location=torch.device("cpu"))
            if to_cuda:
                model.to("cuda")
            else:
                model.to("cpu")
            model.eval()
            self.model_dict[inverted, int(length_min)] = model

    def compute_prediction(self, measurement: Measurement, debug_print: bool = False):
        inference_dict, timestamp_ms = self.get_inference_dict(measurement)
        pred_is_stroke_dict = self.get_pred_is_stroke_dict(inference_dict, debug_print)
        non_inverted_array, inverted_array = self.get_is_stroke_arrays(pred_is_stroke_dict)
        if debug_print:
            pprint(pred_is_stroke_dict)
            print("non_inverted_array: {}, inverted_array: {}".format(non_inverted_array, inverted_array))

        non_inverted_array = (non_inverted_array.sum(axis=1) >= combination_threshold).astype(int)
        inverted_array = (inverted_array.sum(axis=1) >= combination_threshold).astype(int)
        timestamp_ms = timestamp_ms[-len(inverted_array):]

        # TODO
        is_stroke = np.logical_or(non_inverted_array, inverted_array)
        assert len(is_stroke) == len(timestamp_ms), (len(is_stroke), len(timestamp_ms))

        # diff_of_arrays -> 0: no stroke, 1: stroke on non-inverted side, -1: stroke on inverted side, 2: stroke on both
        sum_of_arrays = non_inverted_array + inverted_array
        diff_of_arrays = non_inverted_array - inverted_array
        diff_of_arrays[sum_of_arrays == 2] = 2

        prediction_dict = {"probabilities": diff_of_arrays.tolist(),
                           # "labels": diff_of_arrays.tolist(),
                           "is_stroke": is_stroke.tolist(),
                           "timestamps": timestamp_ms.tolist()}
        return prediction_dict

    def get_inference_dict(self, measurement: Measurement) -> Tuple[dict, np.ndarray]:
        max_length_ticks = min_to_ticks(90, self.frequency)
        step_size_ticks = sec_to_ticks(self.step_size_sec, self.frequency)
        array_3d_dict = get_3d_arrays_from_df(measurement.measurement_dict, only_left_arm_key_list)
        inference_dict = dict()
        with torch.no_grad():
            for length_min in [30, 60, 90]:
                batch_idx = 0
                batch = list()
                ni_predictions = list()
                i_predictions = list()

                num_of_samples = int(
                    (len(array_3d_dict[only_left_arm_key_list[0]]) - max_length_ticks) / step_size_ticks) + 1
                for sample_index in range(num_of_samples):
                    end_idx = sample_index * step_size_ticks + max_length_ticks
                    length_ticks = min_to_ticks(length_min, self.frequency)
                    start_idx = end_idx - length_ticks
                    assert start_idx >= 0, start_idx

                    sample_array = self.get_input_array(array_3d_dict, length_ticks, start_idx=start_idx)
                    batch.append(np.expand_dims(sample_array, axis=0))
                    batch_idx += 1

                    if batch_idx % self.batch_size == 0 or sample_index == (num_of_samples - 1):
                        batch_array = np.concatenate(batch, axis=0)
                        batch_tensor = torch.from_numpy(batch_array).float()
                        if self.to_cuda:
                            batch_tensor = batch_tensor.to("cuda")
                        ni_prediction = self.model_dict["non-inverted", length_min](batch_tensor).to("cpu").numpy()
                        i_prediction = self.model_dict["inverted", length_min](batch_tensor).to("cpu").numpy()
                        if len(batch_array) == 1:
                            ni_prediction = np.expand_dims(ni_prediction, axis=0)
                            i_prediction = np.expand_dims(i_prediction, axis=0)

                        ni_predictions.append(ni_prediction)
                        i_predictions.append(i_prediction)
                        batch = list()

                inference_dict["non-inverted", length_min] = np.concatenate(ni_predictions, axis=0)
                inference_dict["inverted", length_min] = np.concatenate(i_predictions, axis=0)

            timestamp_ms = measurement.measurement_dict[only_left_arm_key_list[0]]["timestamp_ms"].values
            timestamp_ms = timestamp_ms[max_length_ticks::step_size_ticks]
        return inference_dict, timestamp_ms

    def get_pred_is_stroke_dict(self, inference_dict, debug_print: bool = False):
        pred_is_stroke_dict = dict()
        for (inverted, length_min), predictions in inference_dict.items():
            window_length = int(window_length_dict[inverted][length_min] / self.step_size_sec)
            predictions = predictions.argmax(axis=1)
            pred_is_stroke = (predictions < stroke_threshold).astype(int)

            if window_length > 1:
                avg_prob_threshold = avg_prob_threshold_dict[inverted][length_min]
                if len(pred_is_stroke) < window_length:
                    current_window_length = len(pred_is_stroke)
                else:
                    current_window_length = window_length
                avg_pred_is_stroke = sliding_window_view(pred_is_stroke, current_window_length).mean(axis=1)
                if debug_print:
                    print("{}-{}\npred_is_stroke: {}\navg_pred_is_stroke: {}".format(inverted, length_min,
                                                                                     pred_is_stroke, avg_pred_is_stroke))
                pred_is_stroke = (avg_pred_is_stroke >= avg_prob_threshold).astype(int)

            pred_is_stroke_dict[inverted, length_min] = pred_is_stroke
        return pred_is_stroke_dict

    @staticmethod
    def get_is_stroke_arrays(pred_is_stroke_dict):
        index_for_length = {30: 0, 60: 1, 90: 2}

        # predictions in pred_is_stroke_dict can have different lengths
        # but at the end we need arrays with the same length, so those are cut to the length os the shortest
        shortest_length = min([len(pred) for pred in pred_is_stroke_dict.values()])

        non_inverted_array = np.zeros((shortest_length, 3))  # 3 -> (30, 60, 90)
        inverted_array = np.zeros((shortest_length, 3))  # 3 -> (30, 60, 90)
        for (inverted, length_min), pred_is_stroke in pred_is_stroke_dict.items():
            idx = index_for_length[length_min]
            if inverted == "inverted":
                inverted_array[:, idx] = pred_is_stroke[-shortest_length:]
            else:
                non_inverted_array[:, idx] = pred_is_stroke[-shortest_length:]
        return non_inverted_array, inverted_array

    @staticmethod
    def get_input_array(array_3d_dict: dict, length: int, start_idx: int) -> np.ndarray:
        filtered_array_3d_dict = butter_high_pass_filter(array_3d_dict)
        cut_array_dict = cut_array_to_length(filtered_array_3d_dict, length, start_idx=start_idx)
        euclidean_length_dict = calculate_euclidean_length(cut_array_dict)
        euclidean_length_dict = moving_average(euclidean_length_dict, 100)
        divided_euclidean_length_dict = divide_values(euclidean_length_dict, 200, "g")
        euclidean_length_dict = down_sampling(divided_euclidean_length_dict, subsampling_factor=50)

        input_array = create_multivariate_time_series(euclidean_length_dict)
        return input_array
