from typing import Tuple, Union
from utils.discord import DiscordBot

from ai_utils.mlp import MLP
from measurement_utils.measurement import Measurement, NotEnoughData, TimeStampTooHigh, SynchronizationError
from measurement_utils.measurement_manager import MeasurementManager
from utils.general_utils import to_str_timestamp, min_to_millisec, write_discord_log


def make_error_body(error_code: str, measurement_id: str, last_ts: int) -> dict:
    predictions = [{"prediction": error_code,
                    "probability": 1.0,
                    "timestamp": to_str_timestamp(last_ts)
                    }]

    body = {
        "predictions": predictions,
        "measurementId": measurement_id,
        "softwareVersion": "Predictor 1.0",
        "APIVersion": "MotionScan API 1.0"
    }
    return body


def make_body(prediction_dict: dict, measurement_id: str):
    predictions = list()
    for i in range(len(prediction_dict["is_stroke"])):
        prediction = "stroke" if prediction_dict["is_stroke"][i] else "ok"
        predictions.append({
            "prediction": prediction,
            "probability": float(prediction_dict["probabilities"][i]),
            "timestamp": to_str_timestamp(prediction_dict["timestamps"][i]),
        })

    body = {
        "predictions": predictions,
        "measurementId": measurement_id,
        "softwareVersion": "Predictor 1.0",
        "APIVersion": "MotionScan API 1.0"
    }
    return body


def get_measurement(mm: MeasurementManager, measurement_id: str) -> Union[Measurement, None]:
    meas = Measurement(measurement_id)
    meas_df = mm.get_df(measurement_id)

    if meas_df is None:
        return None
    else:
        meas.fill_from_df(meas_df)
        return meas


def check_and_synch_measurement(measurement: Measurement, config_dict: dict, discord: DiscordBot) -> Tuple[str, str]:
    # check if measurement in not None
    if measurement is None:
        return "raw_measurement_NOK", "Error 1"

    # check if measurement has data for each key
    missing_keys = measurement.get_missing_keys()
    if len(missing_keys) != 0:
        write_discord_log("missing keys: {}".format(missing_keys), discord)
        # print("missing keys: {}".format(missing_keys))
        return "missing_key", "Error 2"

    # check if frequency is okay for each key
    expected_delta_ms = (1 / config_dict["frequency"]) * 1000  # ms
    eps = config_dict["frequency_check_eps"]  # ms
    if not measurement.check_frequency(expected_delta_ms, eps=eps, discord=discord):
        return "frequency_NOK", "Error 3"

    # synchronize data along timestamp_ms
    synchron_ok = True
    try:
        measurement.synchronize_measurement_dict()
    except SynchronizationError:
        synchron_ok = False

    if not synchron_ok:
        return "synchron_NOK", "Error 4"

    # check if the measurement is long enough
    length_ms = min_to_millisec(config_dict["meas_length_min"])
    if not measurement.check_length(length_ms):
        return "length_NOK", "Error 5"

    return "OK", "NO Error"


def get_instances(measurement: Measurement,
                  config_dict: dict) -> Tuple[list, Union[list, None]]:
    first_timestamp_ms = measurement.get_first_timestamp_ms()
    length = config_dict["frequency"] * 60 * config_dict["meas_length_min"]
    inference_step_size_ms = config_dict["inference_step_size_sec"] * 1e3
    keys_in_order = (("arm", "acc"),
                     ("leg", "acc"),
                     ("arm", "gyr"),
                     ("leg", "gyr"))

    instance_list = list()
    inference_ts_list = list()
    i = 0
    # print("first_timestamp_ms: {}".format(to_str_timestamp(first_timestamp_ms)))
    # print("last_timestamp_ms: {}".format(to_str_timestamp(measurement.get_last_timestamp_ms())))
    while True:
        end_ts = int(first_timestamp_ms + inference_step_size_ms * i)
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
                # print(e)
                break
            except TimeStampTooHigh as e:
                # print(e)
                return instance_list, inference_ts_list

            _instance.append([diff_mean, ratio_mean_first, ratio_mean])

        if len(_instance) > 0:
            instance_list.append(sum(_instance, []))
            inference_ts_list.append(end_ts)


def get_instances_old(measurement: Measurement,
                      config_dict: dict) -> Tuple[list, Union[list, None]]:
    first_timestamp_ms = measurement.get_first_timestamp_ms()
    length = config_dict["frequency"] * 60 * config_dict["meas_length_min"]
    inference_step_size_ms = config_dict["inference_step_size_sec"] * 1e3
    keys_in_order = (("arm", "acc"),
                     ("leg", "acc"),
                     ("arm", "gyr"),
                     ("leg", "gyr"))

    def collect_instances() -> Tuple[list, list]:
        _instance_list = list()
        _inference_ts_list = list()
        i = 0
        while True:
            end_ts = int(first_timestamp_ms + inference_step_size_ms * i)
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
                    # print(e)
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
    last_timestamp_ms = measurement.get_last_timestamp_ms()
    try:
        instances, inference_ts_list = get_instances(measurement, config_dict)
    except SynchronizationError:
        # synchronization error
        print("synchronization error for measurement: {}".format(measurement.measurement_id))
        error_message = "Error 1"
        return {"probabilities": [1],
                "labels": [None],
                "is_stroke": [error_message],
                "timestamps": [last_timestamp_ms]}

    if inference_ts_list is None:
        # missing key error
        print("missing key error for measurement: {}".format(measurement.measurement_id))
        missing_keys = [[key[0] for key in instance] for instance in instances]
        error_message = "Error 2 {}".format(missing_keys)
        return {"probabilities": [1],
                "labels": [None],
                "is_stroke": [error_message],
                "timestamps": [last_timestamp_ms]}

    if len(instances) == 0:
        # not enough data error
        print("no prediction (len of instances = 0) for measurement: {}".format(measurement.measurement_id))
        error_message = "Error 3"
        return {"probabilities": [1],
                "labels": [None],
                "is_stroke": [error_message],
                "timestamps": [last_timestamp_ms]}

    return model.compute_prediction(instances, inference_ts_list)

# def upload_predictions(prediction_for_measurement_dict: dict):
#     for measurement_id, prediction_dict in prediction_for_measurement_dict.items():
#         predictions = list()
#         for i in range(len(prediction_dict["is_stroke"])):
#             predictions.append({
#                 "prediction": "stroke" if prediction_dict["is_stroke"][i] else "ok",
#                 "probability": float(prediction_dict["probabilities"][i]),
#                 "timestamp": to_str_timestamp(prediction_dict["timestamps"][i]),
#             })
#
#         _body = {
#             "predictions": predictions,
#             "measurementId": measurement_id,
#             "softwareVersion": "Predictor 1.0",
#             "APIVersion": "MotionScan API 1.0"
#         }
#
#         save_predictions(_configuration, _body)
#         print("uploaded {} predictions with measurement id {}".format(len(predictions), measurement_id))