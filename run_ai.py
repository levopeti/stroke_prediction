import os

import keras
import numpy as np
import pandas as pd
from mlp import MLP
from datetime import datetime, timedelta
from typing import List

from pprint import pprint
from measurement import Measurement, NotEnoughData, TimeStampTooHigh, SynchronizationError, key_list, key_map
from api_utils import get_configuration, get_data_for_prediction, get_predictions_from_time_point, save_predictions
from general_utils import to_int_timestamp, to_str_timestamp
from utils.arg_parser_and_config import get_config_dict

"""
ssh motionscan@109.61.102.122
"""


# def df_from_query(_data_list):
#     """
#     - limb
#     - side
#     - timestamp
#     - type
#     - "x"
#     - "y"
#     - "z"
#     """
#
#     df_dict = {"timestamp": list(),
#                "v1": list(),
#                "v2": list(),
#                "v3": list()}
#
#     cast_dict = {"timestamp": lambda x: to_int_timestamp(x),
#                  "v1": lambda x: float(x),
#                  "v2": lambda x: float(x),
#                  "v3": lambda x: float(x)}
#
#     data_dict = dict()
#     for data in _data_list:
#         _measurement_id = data["measurementId"]
#         key = (data["side"], data["limb"], data["type"])
#         key = key_map[key]
#
#         if _measurement_id not in data_dict:
#             data_dict[_measurement_id] = {keys: df_dict.copy() for keys in key_list}
#
#         data_dict[_measurement_id][key]["timestamp"].append(cast_dict["timestamp"](data["timestamp"]))
#         data_dict[_measurement_id][key]["v1"].append(cast_dict["v1"](data["x"]))
#         data_dict[_measurement_id][key]["v2"].append(cast_dict["v2"](data["y"]))
#         data_dict[_measurement_id][key]["v3"].append(cast_dict["v3"](data["z"]))
#
#     for meas_dict in data_dict.values():
#         for key, inner_data_dict in meas_dict.items():
#             meas_dict[key] = pd.DataFrame.from_dict(inner_data_dict)
#
#     print(data_dict.keys())
#     return data_dict


def get_measurements(data_list: list) -> List[Measurement]:
    """ limb, side, timestamp, type, x, y, z"""

    data_df = pd.DataFrame(data_list)
    data_df["timestamp_ms"] = data_df.apply(lambda row: to_int_timestamp(row.timestamp), axis=1)
    data_df["keys_tuple"] = data_df.apply(lambda row: key_map[(row.side, row.limb, row.type)], axis=1)

    measurement_list = list()
    for measurement_id in data_df.measurementId.unique():
        meas = Measurement(measurement_id)
        meas.fill_from_df(data_df[data_df.measurementId == measurement_id])
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


# def create_measurements(data_dict):
#     meas_dict = dict()
#     for _measurement_id in data_dict.keys():
#         _meas = Measurement(_measurement_id)
#         _meas.measurement_dict = {key: data_dict[_measurement_id][key] for key in key_list}
#
#         if 0 in [len(x) for x in _meas.measurement_dict.values()]:
#             raise ValueError("Measurement does not have all type of value (yet)")
#
#         meas_dict[_measurement_id] = _meas
#     return meas_dict


def get_instances_and_make_predictions(model: keras.Model,
                                       measurement_list: List[Measurement],
                                       config_dict: dict):
    prediction_for_measurement_dict = dict()
    for measurement in measurement_list:
        instances, inference_ts_list = get_instances(measurement, config_dict)
        if len(instances) == 0:
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
    while True:
        timestamp_now = datetime.now()
        last_x_hours = (timestamp_now - timedelta(hours=config_dict["timedelta_h"])).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

        data_list = get_data_for_prediction(configuration, last_x_hours, config_dict)
        print("get data for prediction ({})".format(len(data_list)))

        if len(data_list) == 0:
            continue

        measurement_list = get_measurements(data_list)
        prediction_for_measurement_dict = get_instances_and_make_predictions(model, measurement_list, config_dict)
        upload_predictions(prediction_for_measurement_dict)


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
