import argparse
from random import random
from time import sleep

import pytz
import zmq

from datetime import datetime, timedelta

from openapi_client import ApiClient, ApiException
from openapi_client.apis.tags.motion_scan_restapi_end_points_api import MotionScanRESTAPIEndPointsApi
from utils.api_utils import get_configuration
from utils.general_utils import to_str_timestamp

key_list = [("l", "a", "a"),
            ("l", "a", "g"),
            ("l", "l", "a"),
            ("l", "l", "g"),
            ("r", "a", "a"),
            ("r", "a", "g"),
            ("r", "l", "a"),
            ("r", "l", "g")]


def normal_mode(id_list: list, timezone, only_zero_id_list, random_freq):
    time_delta_to_start = timedelta(minutes=2)
    start_timestamp = datetime.now(timezone) - time_delta_to_start
    print("start ts: {}".format(to_str_timestamp(start_timestamp)))
    time_stamp_dict = {m_id: {key: start_timestamp for key in key_list} for m_id in id_list}
    # steps_till_now = int(time_delta_to_start / time_delta_millis)
    # first_request = {m_id: True for m_id in id_list}

    # timestamp_data = datetime.now(timezone)
    # print(to_str_timestamp(timestamp_data))
    # time_stamp_dict = {m_id: timestamp_data for m_id in id_list}
    # time_delta_millis = 40
    num_of_steps = 100
    uploaded_data = 0
    while True:
        all_reach_the_future = True
        for measurement_id in id_list:
            measure = list()
            avg_time_delta_millis = timedelta(milliseconds=40)
            if min(time_stamp_dict[measurement_id].values()) + (avg_time_delta_millis * num_of_steps) > datetime.now(timezone):
                continue

            all_reach_the_future = False

            for i in range(num_of_steps):
                for key in key_list:
                    milliseconds = random.randrange(-random_freq, random_freq) if random_freq else 40
                    time_delta_millis = timedelta(milliseconds=milliseconds)
                    time_stamp_dict[measurement_id][key] += time_delta_millis
                    timestamp_data_string = to_str_timestamp(time_stamp_dict[measurement_id][key])

                    measure.append({
                        "limb": key[1],
                        "side": key[0],
                        "timestamp": timestamp_data_string,
                        "type": key[2],
                        "x": 0 if measurement_id in only_zero_id_list else random(),
                        "y": 0 if measurement_id in only_zero_id_list else random(),
                        "z": 0 if measurement_id in only_zero_id_list else random()
                    })

            test_body = {
                "measure": measure,
                "measurementId": "{:06d}".format(measurement_id),
                "softwareVersion": "Data Collector 1.0",
                "APIVersion": "MotionScan API 1.0"
            }

            with ApiClient(_configuration) as api_client:
                # Create an instance of the API class
                api_instance = MotionScanRESTAPIEndPointsApi(api_client)

                # example passing only required values which don't have defaults set
                header_params = {
                    'x-motionscan-name': 'motionscandemo',
                }
                body = test_body
                try:
                    api_instance.save_measurements(
                        header_params=header_params,
                        body=body,
                    )
                except ApiException as e:
                    print("Exception when calling MotionScanRESTAPIEndPointsApi->save_measurements: %s\n" % e)
            uploaded_data += 100
        print(uploaded_data / (25 * 60 * 60))  # number of hours
        print(to_str_timestamp(min(time_stamp_dict[id_list[0]].values())))
        print(id_list)
        print()
        if all_reach_the_future:
            # if each measurement is up-to-date (we do not want to step in the future)
            print("1 minutes sleep")
            sleep(60)


def local_mode(id_list, timezone, only_zero_id_list, random_freq):
    def get_measurements(number_of_steps, measurement_id, random_multiplier=100):
        measure = list()
        for i in range(number_of_steps):
            time_stamp_dict[measurement_id] += time_delta_millis
            timestamp_data_string = to_str_timestamp(time_stamp_dict[measurement_id])

            for key in key_list:
                measure.append({
                    "limb": key[1],
                    "side": key[0],
                    "timestamp": timestamp_data_string,
                    "type": key[2],
                    "x": 0 if measurement_id in only_zero_id_list else (random() - 0.5) * random_multiplier,
                    "y": 0 if measurement_id in only_zero_id_list else (random() - 0.5) * random_multiplier,
                    "z": 0 if measurement_id in only_zero_id_list else (random() - 0.5) * random_multiplier
                })

        _test_body = {
            "measure": measure,
            "measurementId": "{:06d}".format(measurement_id),
            "softwareVersion": "Data Collector 1.0",
            "APIVersion": "MotionScan API 1.0"
        }
        return _test_body

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")

    while True:
        time_delta_to_start = timedelta(minutes=2)
        start_timestamp = datetime.now(timezone) - time_delta_to_start
        print("start ts: {}".format(to_str_timestamp(start_timestamp)))
        time_stamp_dict = {m_id: start_timestamp for m_id in id_list}
        time_delta_millis = timedelta(milliseconds=40)
        steps_till_now = int(time_delta_to_start / time_delta_millis)
        first_request = {m_id: True for m_id in id_list}

        while True:
            message = socket.recv_string()
            print("\nget string: {}".format(message))
            if message == "restart":
                print("restart")
                socket.send_string("restart ok")
                break
            elif message == "get_measurement_ids":
                # first send the used ids
                print("send used ids: {}".format(id_list))
                socket.send_pyobj(id_list)
            elif message in [str(int_id) for int_id in id_list]:
                meas_id = int(message)
                if first_request[meas_id]:
                    # send measurements for the last "time_delta_to_start" minutes
                    print("send measurements for the last {} minutes".format(time_delta_to_start))
                    first_request[meas_id] = False
                else:
                    # update always until now()
                    last_timestamp = time_stamp_dict[meas_id]
                    time_delta_to_start = datetime.now(timezone) - last_timestamp
                    print("start ts: {}".format(to_str_timestamp(last_timestamp)))
                    steps_till_now = int(time_delta_to_start / time_delta_millis)

                test_body = get_measurements(steps_till_now, measurement_id=int(message))
                socket.send_pyobj(test_body)
            else:
                print("wrong message: {}".format(message))
                socket.send_string("wrong message: {}".format(message))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Provide data for main.py.")
    parser.add_argument("--local_mode", default=False, action="store_true", help="Local data flow through zmq.")
    parser.add_argument("--mocking_mode", default=False, action="store_true", help="First id in list gets only zero values.")
    parser.add_argument("--random_freq", default=None, type=int, help="Frequency is not constant.")
    args = parser.parse_args()

    _id_list = [5, 6]  # [8, 9, 10] [5, 6, 7]

    if args.mocking_mode:
        _only_zero_id_list = [_id_list[0]]
    else:
        _only_zero_id_list = list()

    _timezone = pytz.timezone("Europe/Budapest")

    if args.local_mode:
        local_mode(_id_list, _timezone, _only_zero_id_list, args.random_freq)
    else:
        _config_dict = {"host_url_and_token_path": "./host_url_and_token.json"}
        _configuration = get_configuration(_config_dict)
        normal_mode(_id_list, _timezone, _only_zero_id_list, args.random_freq)




