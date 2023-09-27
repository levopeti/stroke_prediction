import traceback
import zmq

from time import time, sleep
from datetime import datetime, timedelta
from ai_utils.mlp import MLP
from utils.discord import DiscordBot

from utils.api_utils import get_measurement_ids, get_configuration, get_data_for_prediction, save_predictions
from utils.general_utils import to_str_timestamp, from_int_to_datetime, min_to_millisec, get_data_info
from utils.arg_parser_and_config import get_config_dict
from measurement_utils.measurement_manager import MeasurementManager
from openapi_client import Configuration
from utils.main_loop_utils import get_measurement, check_and_synch_measurement, make_error_body, get_instances, \
    make_body
from utils.log_maker import start_log_maker, set_log_dir_path, write_log


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
            write_log("main_loop.txt", "No measurements in the last {} minutes ({})".format(config_dict["meas_length_to_keep_min"],
                                                                                            to_str_timestamp(now_ts)),
                      title="NoMeasurement", print_out=True, color="red", add_date=True, write_discord=True)
            sleep(5 * 60)
            continue

        write_log("main_loop.txt", "Measurement ids to process: {} ({})".format(measurement_ids, to_str_timestamp(now_ts)),
                  title="MeasurementIds", print_out=True, color="blue", add_date=True, write_discord=True)

        for measurement_id in measurement_ids:
            write_log("main_loop.txt", "Process measurement {}".format(measurement_id),
                      title="Process", print_out=True, color="blue", add_date=True, write_discord=True)
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
                write_log("main_loop.txt", "Error message: {}, {}".format(check_message, error_code),
                          title="Error", print_out=True, color="red", add_date=True, write_discord=True)
                if error_code != "Error 1":
                    body = make_error_body(error_code, measurement_id, measurement.get_last_timestamp_ms())
                else:
                    continue
            else:
                print("check is ok")
                instances, inference_ts_list = get_instances(measurement, config_dict)
                prediction_dict = model.compute_prediction(instances, inference_ts_list)
                body = make_body(prediction_dict, measurement_id)

            save_predictions(configuration, body)
            write_log("main_loop.txt", "Uploaded {} prediction(s) with measurement id {} ({:.0f}s)".format(len(body["predictions"]),
                                                                                                           measurement_id, time() - start),
                      title="UploadInfo", print_out=True, color="blue", add_date=True, write_discord=True)
            write_log("main_loop.txt", "Process measurement {} is done ({:.0f}s)".format(measurement_id, time() - start),
                      title="Done", print_out=True, color="green", add_date=True, write_discord=True)

        if config_dict["save_df"]:
            mm.save_each_measurement()

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

    _discord = DiscordBot(active=_config_dict["discord"])

    start_log_maker(_config_dict, _discord)
    set_log_dir_path(_config_dict["log_dir_path"])

    if _config_dict["local_mode"]:
        current_main_loop = main_loop_local_mode
    else:
        current_main_loop = main_loop

    while True:
        try:
            write_log("main_loop.txt", "Stroke ai has started. New session has started (in an infinity loop)",
                      title="UploadInfo", print_out=True, color="blue", add_date=True, write_discord=True)
            current_main_loop(_model, configuration=_configuration, config_dict=_config_dict)
            sleep(10)
        except Exception:
            print(traceback.format_exc())
            write_log("main_loop.txt", "Stroke ai has stopped. Error: {}".format(traceback.format_exc()),
                      title="UploadInfo", print_out=True, color="blue", add_date=True, write_discord=True)
        except KeyboardInterrupt:
            write_log("main_loop.txt", "Stroke ai has stopped. Stopped by keyboard interrupt (infinity loop ends)",
                      title="UploadInfo", print_out=True, color="blue", add_date=True, write_discord=True)
            break
