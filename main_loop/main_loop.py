# import copy
# from concurrent.futures.thread import ThreadPoolExecutor
# from concurrent.futures import as_completed
from time import time, sleep
from datetime import datetime, timedelta

from measurement_utils.measurement_info import MeasurementInfoManager
from utils.api_utils import get_measurement_ids, get_data_for_prediction, save_predictions
from utils.general_utils import to_str_timestamp, from_int_to_datetime, min_to_millisec, get_data_info
from measurement_utils.measurement_manager import MeasurementManager
from openapi_client import Configuration
from main_loop.main_loop_utils import get_measurement, check_and_synch_measurement, make_error_body, make_body
from utils.log_maker import set_log_meas_id, write_log, send_image_to_discord
from ai_utils.model_abstract import Model


def run_main_loop(model: Model, configuration: Configuration, config_dict: dict):
    timezone = config_dict["timezone"]
    mim = MeasurementInfoManager(config_dict)
    mm = MeasurementManager(config_dict, mim)
    model.add_meas_info_manager(mim)

    while True:
        now_ts = datetime.now(timezone)
        if config_dict["start_date"] is None:
            measurement_ids_from = to_str_timestamp(now_ts - timedelta(minutes=config_dict["meas_length_to_keep_min"]))
        else:
            measurement_ids_from = config_dict["start_date"]

        measurement_ids = get_measurement_ids(configuration,
                                              _from=measurement_ids_from,
                                              _interval=min_to_millisec(config_dict["meas_length_to_keep_min"]))
        full_start = time()
        if measurement_ids is None:
            write_log("main_loop.txt",
                      "No measurements in the last {} minutes ({})".format(config_dict["meas_length_to_keep_min"],
                                                                           to_str_timestamp(now_ts)),
                      title="NoMeasurement", print_out=True, color="red", add_date=True, write_discord=False)
            sleep(5 * 60)
            continue

        write_log("main_loop.txt",
                  "Measurement ids to process: {} ({})".format(measurement_ids, to_str_timestamp(now_ts)),
                  title="MeasurementIds", print_out=True, color="blue", add_date=True, write_discord=False)

        for measurement_id in measurement_ids:
            set_log_meas_id(measurement_id)
            write_log("main_loop.txt", "Process measurement {}".format(measurement_id),
                      title="Process", print_out=True, color="blue", add_date=True, write_discord=False)
            start = time()
            from_ts = from_int_to_datetime(mm.get_last_timestamp(measurement_id))

            if from_ts is None:
                # measurement id is new
                if config_dict["start_date"] is None:
                    now_ts = datetime.now(timezone)
                    from_ts = now_ts - timedelta(minutes=config_dict["meas_length_to_keep_min"])
                else:
                    from_ts = datetime.strptime(config_dict["start_date"], '%Y-%m-%dT%H:%M:%S.%fZ')
                    from_ts = timezone.localize(from_ts)
            else:
                from_ts = timezone.localize(from_ts)

            # from_ts_list = [from_ts]
            # while from_ts + timedelta(minutes=config_dict["interval_min"]) <= datetime.now(timezone):
            #     # until from_ts is in the past
            #     from_ts += timedelta(minutes=config_dict["interval_min"])
            #     from_ts_list.append(copy.deepcopy(from_ts))
            #
            # def get_data(_from_ts):
            #     to_ts = _from_ts + timedelta(minutes=config_dict["interval_min"])
            #     data_list, elapsed_time = get_data_for_prediction(configuration,
            #                                                       to_str_timestamp(_from_ts),
            #                                                       measurement_id,
            #                                                       min_to_millisec(config_dict["interval_min"]))
            #     print("\nget data for prediction ({}), from {} to {} ({:.2f}s)".format(len(data_list), _from_ts, to_ts,
            #                                                                            elapsed_time))
            #
            #     # key_combinations = set([(item["side"], item["limb"], item["type"]) for item in data_list])
            #     # write_log("main_loop.txt",
            #     #           "key combinations: {} from {} to {}".format(key_combinations, _from_ts, to_ts),
            #     #           title="KeyCombinations", print_out=True, color="yellow", add_date=True, write_discord=True)
            #
            #     if config_dict["left_arm_only"]:
            #         # data_list = [item for item in data_list if item["limb"] == "a" and item["side"] == "l"]
            #         data_list = [item for item in data_list if item["limb"] == "a" and item["side"] == "r"]
            #     return data_list

            # data_collector = list()
            # with ThreadPoolExecutor(max_workers=10) as executor:
            #     for result in executor.map(get_data, from_ts_list):
            #         data_collector += result

            data_collector = list()
            while True:
                to_ts = from_ts + timedelta(minutes=config_dict["interval_min"])
                data_list, elapsed_time = get_data_for_prediction(configuration,
                                                                  to_str_timestamp(from_ts),
                                                                  measurement_id,
                                                                  min_to_millisec(config_dict["interval_min"]))
                print("\nget data for prediction ({}), from {} to {} ({:.2f}s)".format(len(data_list), from_ts, to_ts,
                                                                                       elapsed_time))

                key_combinations = set([(item["side"], item["limb"], item["type"]) for item in data_list])
                write_log("main_loop.txt",
                          "key combinations: {} from {} to {}".format(key_combinations, from_ts, to_ts),
                          title="KeyCombinations", print_out=True, color="yellow", add_date=True, write_discord=False)

                if config_dict["left_arm_only"]:
                    data_list = [item for item in data_list if item["limb"] == "a" and item["side"] == "l"]
                    # data_list = [item for item in data_list if item["limb"] == "a" and item["side"] == "r"]

                data_collector += data_list
                from_ts += timedelta(minutes=config_dict["interval_min"])
                if from_ts > datetime.now(timezone):
                    # from_ts is in the future
                    break

            if len(data_collector) > 0:
                print("add_data")
                mm.add_data(measurement_id, data_collector, datetime.now(timezone))

                print("get_measurement")
                measurement = get_measurement(mm, measurement_id)
                # get_meas_info(measurement)

                # keys_ok, frequency_ok, synchron_ok, length_ok
                print("check")
                check_message, error_code = check_and_synch_measurement(measurement, config_dict)
                if check_message != "OK":
                    write_log("main_loop.txt", "Error message: {}, {}".format(check_message, error_code),
                              title="Error", print_out=True, color="red", add_date=True, write_discord=True)
                    if error_code != "Error 1":
                        last_timestamp = measurement.get_last_timestamp_ms()
                        if last_timestamp is not None:
                            body = make_error_body(error_code, measurement_id, measurement.get_last_timestamp_ms())
                        else:
                            write_log("main_loop.txt", "Last timestamp is None (Zero length df)",
                                      title="ZeroLengthDf", print_out=True, color="red", add_date=True, write_discord=True)
                    else:
                        continue
                else:
                    write_log("main_loop.txt", "Check is OK",
                              title="CheckOK", print_out=True, color="green", add_date=True, write_discord=False)
                    prediction_dict = model.compute_prediction(measurement)
                    body = make_body(prediction_dict, measurement_id)

                if mm.is_time_to_save(measurement_id):
                    save_predictions(configuration, body)
                    mim.plot_timeline(measurement_id)
                    send_image_to_discord("./discord_plot.png")
                    write_log("main_loop.txt", "{} prediction with measurement id {}".format(body["predictions"],
                                                                                             measurement_id),
                              title="Prediction", print_out=True, color="blue", add_date=True, write_discord=True)
                    write_log("main_loop.txt",
                              "Uploaded {} prediction(s) with measurement id {} ({:.0f}s)".format(len(body["predictions"]),
                                                                                                  measurement_id,
                                                                                                  time() - start),
                              title="UploadInfo", print_out=True, color="blue", add_date=True, write_discord=False)
                else:
                    write_log("main_loop.txt",
                              "Prediction(s) did not uploaded with measurement id {},"
                              " because it is too early".format(measurement_id),
                              title="UploadInfo", print_out=True, color="blue", add_date=True, write_discord=False)

                write_log("main_loop.txt",
                          "Process measurement {} is done ({:.0f}s)".format(measurement_id, time() - start),
                          title="Done", print_out=True, color="green", add_date=True, write_discord=False)
            else:
                write_log("main_loop.txt",
                          "No new data for measurement {} ({:.0f}s)".format(measurement_id, time() - start),
                          title="NoNewData", print_out=True, color="yellow", add_date=True, write_discord=False)

        if config_dict["save_df"]:
            mm.save_each_measurement()

        mm.drop_old_data()
        get_data_info(mm.all_measurement_dict, "all")
        if time() - full_start < 60:
            print("2 minutes sleep")
            sleep(2 * 60)
