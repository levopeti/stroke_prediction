import zmq

from time import time, sleep
from datetime import datetime

from measurement_utils.measurement_info import MeasurementInfoManager
from utils.general_utils import get_data_info
from measurement_utils.measurement_manager import MeasurementManager
from main_loop.main_loop_utils import get_measurement, check_and_synch_measurement, make_error_body, make_body
from ai_utils.model_abstract import Model
from utils.log_maker import send_image_to_discord


def get_sockets():
    # get measurements
    context = zmq.Context()
    _socket_req = context.socket(zmq.REQ)
    _socket_req.connect("tcp://localhost:5555")

    # send predictions
    _socket_push = context.socket(zmq.PUSH)
    _socket_push.bind("tcp://*:5556")

    # (re)start run_measurement.py
    _socket_req.send_string("restart")
    print(_socket_req.recv_string())
    return _socket_req, _socket_push


def run_main_loop_local(model: Model, config_dict: dict, *args, **kwargs):
    timezone = config_dict["timezone"]
    mim = MeasurementInfoManager(config_dict)
    mm = MeasurementManager(config_dict, mim)
    model.add_meas_info_manager(mim)
    socket_req, socket_push = get_sockets()

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

            if config_dict["left_arm_only"]:
                # data_list = [item for item in data_list if item["limb"] == "a" and item["side"] == "l"]
                data_list = [item for item in data_list if item["limb"] == "a" and item["side"] == "r"]

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
                prediction_dict = model.compute_prediction(measurement)
                body = make_body(prediction_dict, measurement_id)

            mim.plot_timeline(measurement_id)
            send_image_to_discord("./discord_plot.png")
            socket_push.send_pyobj(body)
            print("process measurement {} is done ({:.0f}s)".format(measurement_id, time() - start))

        mm.drop_old_data()
        get_data_info(mm.all_measurement_dict, "all")
        if time() - full_start < 60:
            print("2 minutes sleep")
            sleep(10)

