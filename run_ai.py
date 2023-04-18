from mlp import MLP
from datetime import datetime, timedelta
from typing import List, Tuple, Union
from time import time, sleep

from measurement import Measurement, NotEnoughData, TimeStampTooHigh, SynchronizationError
from api_utils import get_measurement_ids, get_configuration, get_data_for_prediction, save_predictions
from general_utils import to_str_timestamp, hour_to_millisec, min_to_millisec
from utils.arg_parser_and_config import get_config_dict
from measurement_manager import MeasurementManager
from openapi_client import Configuration


# def get_measurements(am_df: AllMeasurementsDF) -> List[Measurement]:
#     measurement_list = list()
#     for measurement_id in am_df.all_data_df.measurementId.unique():
#         meas = Measurement(measurement_id)
#         meas.fill_from_df(am_df.all_data_df[am_df.all_data_df.measurementId == measurement_id])
#         measurement_list.append(meas)
#     return measurement_list


def get_measurement(mm: MeasurementManager, measurement_id: str) -> Measurement:
    meas = Measurement(measurement_id)
    meas.fill_from_df(mm.get_df(measurement_id))
    return meas


def get_instances(measurement: Measurement,
                  first_timestamp_ms: int,
                  config_dict: dict) -> Tuple[list, Union[list, None]]:
    expected_delta = (1 / config_dict["frequency"]) * 1000  # ms
    eps = 3

    missing_keys = measurement.get_missing_keys()

    if len(missing_keys) != 0:
        return missing_keys, None

    measurement.check_frequency(expected_delta, eps=eps)
    measurement.synchronize_measurement_dict()

    length = config_dict["frequency"] * 60 * config_dict["meas_length_min"]
    inference_delta_ms = config_dict["inference_delta_sec"] * 1e3
    keys_in_order = (("arm", "acc"),
                     ("leg", "acc"),
                     ("arm", "gyr"),
                     ("leg", "gyr"))

    def collect_instances() -> Tuple[list, list]:
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
                    ratio_mean = measurement.get_limb_ratio_mean(key[0], key[1], length, end_ts=end_ts,
                                                                 mean_first=False)
                except NotEnoughData as e:
                    print(e)
                    break
                except TimeStampTooHigh as e:
                    return _instance_list, _inference_ts_list

                _instance.append([diff_mean, ratio_mean_first, ratio_mean])

            if len(_instance) > 0:
                _instance_list.append(sum(_instance, []))
                _inference_ts_list.append(end_ts)

    return collect_instances()


# def get_instances_and_make_predictions(model: MLP,
#                                        measurement_list: List[Measurement],
#                                        config_dict: dict):
#     prediction_for_measurement_dict = dict()
#     for measurement in measurement_list:
#         first_timestamp_ms = measurement.get_first_timestamp_ms()
#         try:
#             instances, inference_ts_list = get_instances(measurement, first_timestamp_ms, config_dict)
#         except SynchronizationError:
#             # synchronization error
#             error_message = "problem during synchronizing the measurements"
#             prediction_for_measurement_dict[measurement.measurement_id] = {"probabilities": [1],
#                                                                            "labels": [None],
#                                                                            "is_stroke": [error_message],
#                                                                            "timestamps": [first_timestamp_ms]}
#             continue
#
#         if inference_ts_list is None:
#             # missing key error
#             error_message = "keys are missing: {}".format(instances)
#             prediction_for_measurement_dict[measurement.measurement_id] = {"probabilities": [1],
#                                                                            "labels": [None],
#                                                                            "is_stroke": [error_message],
#                                                                            "timestamps": [first_timestamp_ms]}
#             continue
#
#         if len(instances) == 0:
#             # not enough data error
#             print("no prediction (len of instances = 0) for measurement: {}".format(measurement.measurement_id))
#             error_message = "not enough data (yet)"
#             prediction_for_measurement_dict[measurement.measurement_id] = {"probabilities": [1],
#                                                                            "labels": [None],
#                                                                            "is_stroke": [error_message],
#                                                                            "timestamps": [first_timestamp_ms]}
#             continue
#         prediction_dict = model.compute_prediction(instances, inference_ts_list)
#         prediction_for_measurement_dict[measurement.measurement_id] = prediction_dict
#     return prediction_for_measurement_dict


def get_instances_and_make_predictions(model: MLP,
                                       measurement: Measurement,
                                       config_dict: dict):
    first_timestamp_ms = measurement.get_first_timestamp_ms()
    try:
        instances, inference_ts_list = get_instances(measurement, first_timestamp_ms, config_dict)
    except SynchronizationError:
        # synchronization error
        error_message = "problem during synchronizing the measurements"
        return {"probabilities": [1],
                "labels": [None],
                "is_stroke": [error_message],
                "timestamps": [first_timestamp_ms]}

    if inference_ts_list is None:
        # missing key error
        error_message = "keys are missing: {}".format(instances)
        return {"probabilities": [1],
                "labels": [None],
                "is_stroke": [error_message],
                "timestamps": [first_timestamp_ms]}

    if len(instances) == 0:
        # not enough data error
        print("no prediction (len of instances = 0) for measurement: {}".format(measurement.measurement_id))
        error_message = "not enough data (yet)"
        return {"probabilities": [1],
                "labels": [None],
                "is_stroke": [error_message],
                "timestamps": [first_timestamp_ms]}

    return model.compute_prediction(instances, inference_ts_list)


def upload_predictions(prediction_for_measurement_dict: dict):
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


def upload_prediction(prediction_dict: dict, measurement_id: str):
    start = time()
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
    print("uploaded {} predictions with measurement id {} ({:.0}s)".format(len(predictions), measurement_id,
                                                                           time() - start))


def main_loop_old(model, configuration, config_dict):
    am_df = AllMeasurementsDF()
    ts_now = datetime.now()

    while True:
        from_ts = ts_now - timedelta(hours=config_dict["timedelta_from_now_h"])
        to_ts = from_ts + timedelta(milliseconds=config_dict["interval_milliseconds"])

        data_list = get_data_for_prediction(configuration,
                                            from_ts.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
                                            config_dict)
        print("\nget data for prediction ({}),"
              " timedelta_from_now_h: {:.2f}, from {} to {}".format(len(data_list),
                                                                    config_dict["timedelta_from_now_h"],
                                                                    from_ts,
                                                                    to_ts))

        if len(data_list) > 0:
            am_df.add_data(data_list)
            measurement_list = get_measurements(am_df)
            prediction_for_measurement_dict = get_instances_and_make_predictions(model, measurement_list,
                                                                                 config_dict)
            upload_predictions(prediction_for_measurement_dict)

        config_dict["timedelta_from_now_h"] += config_dict["interval_milliseconds"] / (1000 * 60 * 60)  # hours


def main_loop(model: MLP, configuration: Configuration, config_dict: dict):
    mm = MeasurementManager(config_dict)

    while True:
        now_ts = datetime.now()
        measurement_ids = get_measurement_ids(configuration,
                                              _from=to_str_timestamp(now_ts),
                                              _interval=min_to_millisec(config_dict["meas_length_to_keep_min"]))

        if measurement_ids is None:
            print("No measurements in the last {} minutes ({})".format(config_dict["meas_length_to_keep_min"],
                                                                       to_str_timestamp(now_ts)))
            sleep(5 * 60)
            continue

        print("Measurement ids to process: {}".format(measurement_ids))

        for measurement_id in measurement_ids:
            print("process measurement {}".format(measurement_id))
            start = time()
            mm.drop_old_data(measurement_id)
            from_ts = mm.get_last_timestamp(measurement_id)

            if from_ts is None:
                # measurement id is new
                now_ts = datetime.now()
                from_ts = now_ts - timedelta(minutes=config_dict["meas_length_to_keep_min"])

            while True:
                to_ts = from_ts + timedelta(minutes=config_dict["interval_min"])
                data_list, elapsed_time = get_data_for_prediction(configuration,
                                                                  to_str_timestamp(from_ts),
                                                                  measurement_id,
                                                                  min_to_millisec(config_dict["interval_min"]))
                print("\nget data for prediction ({}), from {} to {} ({:.2f}s)".format(len(data_list), from_ts, to_ts,
                                                                                       elapsed_time))

                if len(data_list) > 0:
                    mm.add_data(measurement_id, data_list)

                from_ts += timedelta(minutes=config_dict["interval_min"])

                if from_ts > now_ts:
                    # from_ts is in the future
                    break

            measurement = get_measurement(mm, measurement_id)
            prediction_dict = get_instances_and_make_predictions(model, measurement, config_dict)
            upload_prediction(prediction_dict, measurement_id)
            print("process measurement {} is done ({:.0}s)".format(measurement_id, time() - start))


if __name__ == "__main__":
    # TODO: time measurement
    _config_dict = get_config_dict()
    _configuration = get_configuration(_config_dict)
    _model = MLP(_config_dict)

    main_loop(_model, _configuration, _config_dict)

    # timestamp_now = datetime.now()
    # last_3_hours = timestamp_now - timedelta(hours=9)
    # last_3_hours = last_3_hours.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    # prediction_list = get_predictions_from_time_point(_configuration, _from=last_3_hours, _interval=3000000)
    # print(prediction_list)
