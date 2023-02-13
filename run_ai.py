import os
import keras
import time
import numpy as np
import pandas as pd


from mlp import MLP
from datetime import datetime, timedelta
from typing import List

from measurement import Measurement, NotEnoughData, TimeStampTooHigh, SynchronizationError, key_list, key_map
from api_utils import get_configuration, get_data_for_prediction, get_predictions_from_time_point, save_predictions
from general_utils import to_int_timestamp, to_str_timestamp, get_data_info
from utils.arg_parser_and_config import get_config_dict
from all_measurements_df import AllMeasurementsDF


def get_measurements(am_df: AllMeasurementsDF) -> List[Measurement]:
    measurement_list = list()
    for measurement_id in am_df.all_data_df.measurementId.unique():
        meas = Measurement(measurement_id)
        meas.fill_from_df(am_df.all_data_df[am_df.all_data_df.measurementId == measurement_id])
        measurement_list.append(meas)
    return measurement_list


def get_instances(measurement: Measurement, config_dict: dict):
    expected_delta = (1 / config_dict["frequency"]) * 1000  # ms
    eps = 3
    measurement.check_frequency(expected_delta, eps=eps)
    measurement.synchronize_measurement_dict()

    first_timestamp_ms = measurement.get_first_timestamp_ms()

    length = config_dict["frequency"] * 60 * config_dict["meas_length_min"]
    inference_delta_ms = config_dict["inference_delta_sec"] * 1e3
    keys_in_order = (("arm", "acc"),
                     ("leg", "acc"),
                     ("arm", "gyr"),
                     ("leg", "gyr"))

    def collect_instances():
        _instance_list = list()
        _inference_ts_list = list()
        i = 0
        while True:
            end_ts = int(first_timestamp_ms + inference_delta_ms * i)
            i += 1
            _instance = list()
            for key in keys_in_order:
                try:
                    diff_mean = measurement.get_limb_diff_mean(key[0], key[1], length, end_ts=end_ts)
                    ratio_mean_first = measurement.get_limb_ratio_mean(key[0], key[1], length, end_ts=end_ts,
                                                                       mean_first=True)
                    ratio_mean = measurement.get_limb_ratio_mean(key[0], key[1], length, end_ts=end_ts, mean_first=False)
                except NotEnoughData:
                    break
                except TimeStampTooHigh:
                    return _instance_list, _inference_ts_list

                _instance.append([diff_mean, ratio_mean_first, ratio_mean])

            if len(_instance) > 0:
                _instance_list.append(sum(_instance, []))
                _inference_ts_list.append(end_ts)

    return collect_instances()


def get_instances_and_make_predictions(model: keras.Model,
                                       measurement_list: List[Measurement],
                                       config_dict: dict):
    prediction_for_measurement_dict = dict()
    for measurement in measurement_list:
        instances, inference_ts_list = get_instances(measurement, config_dict)
        if len(instances) == 0:
            print("no prediction (len of instances = 0) for measurement: {}".format(measurement.measurement_id))
            # TODO: log
            continue
        prediction_dict = model.compute_prediction(instances, inference_ts_list)
        prediction_for_measurement_dict[measurement.measurement_id] = prediction_dict
    return prediction_for_measurement_dict


def upload_predictions(prediction_for_measurement_dict):
    for measurement_id, prediction_dict in prediction_for_measurement_dict.items():
        predictions = list()
        for i in range(len(prediction_dict["is_stroke"])):
            predictions.append({
                "prediction": "stroke" if prediction_dict["is_stroke"][i] else "ok",
                "probability": float(prediction_dict["probabilities"][i]),
                "timestamp": to_str_timestamp(prediction_dict["timestamps"][i]),
            })

        _body = {
            "predictions": predictions,
            "measurementId": measurement_id,
            "softwareVersion": "Predictor 1.0",
            "APIVersion": "MotionScan API 1.0"
        }

        save_predictions(_configuration, _body)
        print("uploaded {} predictions with measurement id {}".format(len(predictions), measurement_id))


def main_loop(model, configuration, config_dict):
    am_df = AllMeasurementsDF()

    while True:
        timestamp_now = datetime.now()
        last_x_hours = (timestamp_now - timedelta(hours=config_dict["timedelta_h"])).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

        data_list = get_data_for_prediction(configuration, last_x_hours, config_dict)
        print("\nget data for prediction ({}), timedelta_h: {:.2f}".format(len(data_list), config_dict["timedelta_h"]))

        if len(data_list) > 0:
            am_df.add_data(data_list)
            measurement_list = get_measurements(am_df)
            prediction_for_measurement_dict = get_instances_and_make_predictions(model, measurement_list, config_dict)
            upload_predictions(prediction_for_measurement_dict)

        config_dict["timedelta_h"] += config_dict["interval_ms"] / (1000 * 60 * 60)  # hours


if __name__ == "__main__":
    _config_dict = get_config_dict()
    _configuration = get_configuration(_config_dict)
    _model = MLP(_config_dict)

    main_loop(_model, _configuration, _config_dict)

    # timestamp_now = datetime.now()
    # last_3_hours = timestamp_now - timedelta(hours=9)
    # last_3_hours = last_3_hours.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    # prediction_list = get_predictions_from_time_point(_configuration, _from=last_3_hours, _interval=3000000)
    # print(prediction_list)
