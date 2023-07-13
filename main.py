import traceback
import zmq

from time import time, sleep
from datetime import datetime, timedelta
from ai_utils.mlp import MLP
from utils.discord import DiscordBot

from utils.api_utils import get_measurement_ids, get_configuration, get_data_for_prediction, upload_prediction, save_predictions
from utils.general_utils import to_str_timestamp, from_int_to_datetime, min_to_millisec, get_data_info
from utils.arg_parser_and_config import get_config_dict
from measurement_utils.measurement_manager import MeasurementManager
from openapi_client import Configuration
from utils.main_loop_utils import get_measurement, check_and_synch_measurement, make_error_body, get_instances, make_body


# # def get_measurements(am_df: AllMeasurementsDF) -> List[Measurement]:
# #     measurement_list = list()
# #     for measurement_id in am_df.all_data_df.measurementId.unique():
# #         meas = Measurement(measurement_id)
# #         meas.fill_from_df(am_df.all_data_df[am_df.all_data_df.measurementId == measurement_id])
# #         measurement_list.append(meas)
# #     return measurement_list
#
#
# def get_measurement(mm: MeasurementManager, measurement_id: str) -> Union[Measurement, None]:
#     meas = Measurement(measurement_id)
#     meas_df = mm.get_df(measurement_id)
#
#     if meas_df is None:
#         return None
#     else:
#         meas.fill_from_df(meas_df)
#         return meas
#
#
# def get_instances(measurement: Measurement,
#                   first_timestamp_ms: int,
#                   config_dict: dict) -> Tuple[list, Union[list, None]]:
#     expected_delta = (1 / config_dict["frequency"]) * 1000  # ms
#     eps = 3
#
#     missing_keys = measurement.get_missing_keys()
#
#     if len(missing_keys) != 0:
#         return missing_keys, None
#
#     measurement.check_frequency(expected_delta, eps=eps)
#     measurement.synchronize_measurement_dict()
#
#     length = config_dict["frequency"] * 60 * config_dict["meas_length_min"]
#     inference_delta_ms = config_dict["inference_delta_sec"] * 1e3
#     keys_in_order = (("arm", "acc"),
#                      ("leg", "acc"),
#                      ("arm", "gyr"),
#                      ("leg", "gyr"))
#
#     def collect_instances() -> Tuple[list, list]:
#         _instance_list = list()
#         _inference_ts_list = list()
#         i = 0
#         while True:
#             end_ts = int(first_timestamp_ms + inference_delta_ms * i)
#             i += 1
#             _instance = list()
#             for key in keys_in_order:
#                 try:
#                     diff_mean = measurement.get_limb_diff_mean(key[0], key[1], length, end_ts=end_ts)
#                     ratio_mean_first = measurement.get_limb_ratio_mean(key[0], key[1], length, end_ts=end_ts,
#                                                                        mean_first=True)
#                     ratio_mean = measurement.get_limb_ratio_mean(key[0], key[1], length, end_ts=end_ts,
#                                                                  mean_first=False)
#                 except NotEnoughData as e:
#                     # print(e)
#                     break
#                 except TimeStampTooHigh as e:
#                     return _instance_list, _inference_ts_list
#
#                 _instance.append([diff_mean, ratio_mean_first, ratio_mean])
#
#             if len(_instance) > 0:
#                 _instance_list.append(sum(_instance, []))
#                 _inference_ts_list.append(end_ts)
#
#     return collect_instances()
#
#
# # def get_instances_and_make_predictions(model: MLP,
# #                                        measurement_list: List[Measurement],
# #                                        config_dict: dict):
# #     prediction_for_measurement_dict = dict()
# #     for measurement in measurement_list:
# #         first_timestamp_ms = measurement.get_first_timestamp_ms()
# #         try:
# #             instances, inference_ts_list = get_instances(measurement, first_timestamp_ms, config_dict)
# #         except SynchronizationError:
# #             # synchronization error
# #             error_message = "problem during synchronizing the measurements"
# #             prediction_for_measurement_dict[measurement.measurement_id] = {"probabilities": [1],
# #                                                                            "labels": [None],
# #                                                                            "is_stroke": [error_message],
# #                                                                            "timestamps": [first_timestamp_ms]}
# #             continue
# #
# #         if inference_ts_list is None:
# #             # missing key error
# #             error_message = "keys are missing: {}".format(instances)
# #             prediction_for_measurement_dict[measurement.measurement_id] = {"probabilities": [1],
# #                                                                            "labels": [None],
# #                                                                            "is_stroke": [error_message],
# #                                                                            "timestamps": [first_timestamp_ms]}
# #             continue
# #
# #         if len(instances) == 0:
# #             # not enough data error
# #             print("no prediction (len of instances = 0) for measurement: {}".format(measurement.measurement_id))
# #             error_message = "not enough data (yet)"
# #             prediction_for_measurement_dict[measurement.measurement_id] = {"probabilities": [1],
# #                                                                            "labels": [None],
# #                                                                            "is_stroke": [error_message],
# #                                                                            "timestamps": [first_timestamp_ms]}
# #             continue
# #         prediction_dict = model.compute_prediction(instances, inference_ts_list)
# #         prediction_for_measurement_dict[measurement.measurement_id] = prediction_dict
# #     return prediction_for_measurement_dict
#
#
# def get_instances_and_make_predictions(model: MLP,
#                                        measurement: Measurement,
#                                        config_dict: dict):
#     last_timestamp_ms = measurement.get_last_timestamp_ms()
#     try:
#         instances, inference_ts_list = get_instances(measurement, last_timestamp_ms, config_dict)
#     except SynchronizationError:
#         # synchronization error
#         print("synchronization error for measurement: {}".format(measurement.measurement_id))
#         error_message = "Error 1"
#         return {"probabilities": [1],
#                 "labels": [None],
#                 "is_stroke": [error_message],
#                 "timestamps": [last_timestamp_ms]}
#
#     if inference_ts_list is None:
#         # missing key error
#         print("missing key error for measurement: {}".format(measurement.measurement_id))
#         missing_keys = [[key[0] for key in instance] for instance in instances]
#         error_message = "Error 2 {}".format(missing_keys)
#         return {"probabilities": [1],
#                 "labels": [None],
#                 "is_stroke": [error_message],
#                 "timestamps": [last_timestamp_ms]}
#
#     if len(instances) == 0:
#         # not enough data error
#         print("no prediction (len of instances = 0) for measurement: {}".format(measurement.measurement_id))
#         error_message = "Error 3"
#         return {"probabilities": [1],
#                 "labels": [None],
#                 "is_stroke": [error_message],
#                 "timestamps": [last_timestamp_ms]}
#
#     return model.compute_prediction(instances, inference_ts_list)
#
#
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
#
#
# def upload_prediction(prediction_dict: dict, measurement_id: str):
#     start = time()
#     predictions = list()
#     for i in range(len(prediction_dict["is_stroke"])):
#         if isinstance(prediction_dict["is_stroke"][i], str):
#             # error message
#             prediction = prediction_dict["is_stroke"][i]
#         else:
#             assert isinstance(prediction_dict["is_stroke"][i], bool)
#             prediction = "stroke" if prediction_dict["is_stroke"][i] else "ok"
#
#         predictions.append({
#             "prediction": prediction,
#             "probability": float(prediction_dict["probabilities"][i]),
#             "timestamp": to_str_timestamp(prediction_dict["timestamps"][i]),
#         })
#
#     _body = {
#         "predictions": predictions,
#         "measurementId": measurement_id,
#         "softwareVersion": "Predictor 1.0",
#         "APIVersion": "MotionScan API 1.0"
#     }
#     save_predictions(_configuration, _body)
#     print("uploaded {} prediction(s) with measurement id {} ({:.0}s)".format(len(predictions), measurement_id,
#                                                                              time() - start))


def main_loop(model: MLP, configuration: Configuration, config_dict: dict):
    timezone = config_dict["timezone"]
    mm = MeasurementManager(config_dict)

    while True:
        now_ts = datetime.now(timezone)
        measurement_ids = get_measurement_ids(configuration,
                                              _from=to_str_timestamp(
                                                  now_ts - timedelta(minutes=config_dict["meas_length_to_keep_min"])),
                                              _interval=min_to_millisec(config_dict["meas_length_to_keep_min"]))

        full_start = time()
        if measurement_ids is None:
            print("\nNo measurements in the last {} minutes ({})".format(config_dict["meas_length_to_keep_min"],
                                                                         to_str_timestamp(now_ts)))
            sleep(5 * 60)
            continue

        print("\nMeasurement ids to process: {} ({})".format(measurement_ids, to_str_timestamp(now_ts)))

        for measurement_id in measurement_ids:
            print("\nprocess measurement {}".format(measurement_id))
            start = time()
            from_ts = from_int_to_datetime(mm.get_last_timestamp(measurement_id))

            if from_ts is None:
                # measurement id is new
                now_ts = datetime.now(timezone)
                from_ts = now_ts - timedelta(minutes=config_dict["meas_length_to_keep_min"])
            else:
                from_ts = timezone.localize(from_ts)
            while True:
                to_ts = from_ts + timedelta(minutes=config_dict["interval_min"])
                data_list, elapsed_time = get_data_for_prediction(configuration,
                                                                  to_str_timestamp(from_ts),
                                                                  measurement_id,
                                                                  min_to_millisec(config_dict["interval_min"]))
                print("\nget data for prediction ({}), from {} to {} ({:.2f}s)".format(len(data_list), from_ts, to_ts,
                                                                                       elapsed_time))
                print("add_data")
                mm.add_data(measurement_id, data_list, datetime.now(timezone))

                from_ts += timedelta(minutes=config_dict["interval_min"])
                if from_ts > datetime.now(timezone):
                    # from_ts is in the future
                    break

            print("get_measurement")
            measurement = get_measurement(mm, measurement_id)
            # get_meas_info(measurement)

            # keys_ok, frequency_ok, synchron_ok, length_ok
            print("check")
            check_message, error_code = check_and_synch_measurement(measurement, config_dict)
            if check_message != "OK":
                print(error_code)
                if error_code != "Error 1":
                    body = make_error_body(error_code, measurement_id, measurement.get_last_timestamp_ms())
                else:
                    print("There is no measurement!")
                    continue
            else:
                print("check is ok")
                instances, inference_ts_list = get_instances(measurement, config_dict)
                prediction_dict = model.compute_prediction(instances, inference_ts_list)
                body = make_body(prediction_dict, measurement_id)

            #     if len(data_list) > 0:
            #         mm.add_data(measurement_id, data_list, datetime.now(timezone))
            #

            #
            # measurement = get_measurement(mm, measurement_id)
            #
            # if measurement is None:
            #     print("no prediction for measurement {}".format(measurement_id))
            #     continue
            #
            # prediction_dict = get_instances_and_make_predictions(model, measurement, config_dict)
            # upload_prediction(configuration, prediction_dict, measurement_id)
            save_predictions(configuration, body)
            print("uploaded {} prediction(s) with measurement id {} ({:.0f}s)".format(len(body["predictions"]),
                                                                                      measurement_id,
                                                                                      time() - start))
            print("process measurement {} is done ({:.0f}s)".format(measurement_id, time() - start))

        mm.drop_old_data()
        get_data_info(mm.all_measurement_dict, "all")
        if time() - full_start < 60:
            print("2 minutes sleep")
            sleep(2 * 60)


def main_loop_local_mode(model: MLP, config_dict: dict, *args, **kwargs):
    timezone = config_dict["timezone"]
    mm = MeasurementManager(config_dict)

    # get measurements
    context = zmq.Context()
    socket_req = context.socket(zmq.REQ)
    socket_req.connect("tcp://localhost:5555")

    # send predictions
    socket_push = context.socket(zmq.PUSH)
    socket_push.bind("tcp://*:5556")

    # (re)start run_measurement.py
    socket_req.send_string("restart")
    print(socket_req.recv_string())

    while True:
        full_start = time()

        # first pull is for getting the used ids
        socket_req.send_string("get_measurement_ids")
        measurement_ids = socket_req.recv_pyobj()
        print("\nMeasurement ids to process: {}".format(measurement_ids))

        for measurement_id in measurement_ids:
            print("\nprocess measurement {}".format(measurement_id))
            start = time()
            socket_req.send_string(str(measurement_id))
            data_list = socket_req.recv_pyobj()["measure"]
            print("\nget data for prediction ({}) {:.2f} sec".format(len(data_list), time() - start))

            print("add_data")
            mm.add_data(measurement_id, data_list, datetime.now(timezone))
            print("get_measurement")
            measurement = get_measurement(mm, measurement_id)
            # get_meas_info(measurement)

            # keys_ok, frequency_ok, synchron_ok, length_ok
            print("check")
            check_message, error_code = check_and_synch_measurement(measurement, config_dict)
            if check_message != "OK":
                print(error_code)
                body = make_error_body(error_code, measurement_id, measurement.get_last_timestamp_ms())
            else:
                print("check is ok")
                instances, inference_ts_list = get_instances(measurement, config_dict)
                prediction_dict = model.compute_prediction(instances, inference_ts_list)
                body = make_body(prediction_dict, measurement_id)

            socket_push.send_pyobj(body)
            print("process measurement {} is done ({:.0f}s)".format(measurement_id, time() - start))

        mm.drop_old_data()
        get_data_info(mm.all_measurement_dict, "all")
        if time() - full_start < 60:
            print("2 minutes sleep")
            sleep(1)


if __name__ == "__main__":
    # TODO: time measurement
    _config_dict = get_config_dict()
    _configuration = get_configuration(_config_dict)
    _model = MLP(_config_dict)

    discord = DiscordBot(active=_config_dict["discord"])

    if _config_dict["local_mode"]:
        current_main_loop = main_loop_local_mode
    else:
        current_main_loop = main_loop

    try:
        while True:
            discord.send_message(fields=[{"name": "stroke ai has started",
                                          "value": "new session has started (in an infinity loop)",
                                          "inline": True}])
            current_main_loop(_model, configuration=_configuration, config_dict=_config_dict)
            sleep(10)
    except Exception:
        print(traceback.format_exc())
        discord.send_message(fields=[{"name": "stroke ai has stopped",
                                      "value": "error: {}".format(traceback.format_exc()),
                                      "inline": True}])
    except KeyboardInterrupt:
        discord.send_message(fields=[{"name": "stroke ai has stopped",
                                      "value": "stopped by keyboard interrupt (infinity loop ends)",
                                      "inline": True}])
