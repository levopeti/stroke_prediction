import random
import pytz
import zmq

from pprint import pprint
from datetime import datetime, timedelta

# from openapi_client import Configuration, ApiClient, ApiException
# from openapi_client.apis.tags.motion_scan_restapi_end_points_api import MotionScanRESTAPIEndPointsApi
# from api_utils import get_configuration


key_list = [("l", "a", "a"),
            ("l", "a", "g"),
            ("l", "l", "a"),
            ("l", "l", "g"),
            ("r", "a", "a"),
            ("r", "a", "g"),
            ("r", "l", "a"),
            ("r", "l", "g")]



def normal_mode(id_list: list):
    timestamp_data = datetime.now()
    print(timestamp_data.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z")
    time_stamp_dict = {m_id: timestamp_data for m_id in id_list}
    time_delta_millis = 40

    uploaded_data = 0
    while True:
        for measurement_id in id_list:
            measure = list()
            for i in range(100):
                time_stamp_dict[measurement_id] += timedelta(milliseconds=time_delta_millis)
                timestamp_data_string = time_stamp_dict[measurement_id].strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

                for key in key_list:
                    measure.append({
                        "limb": key[1],
                        "side": key[0],
                        "timestamp": timestamp_data_string,
                        "type": key[2],
                        "x": random.random(),
                        "y": random.random(),
                        "z": random.random()
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
        print(uploaded_data / (25 * 60 * 60))
        print(time_stamp_dict[id_list[0]].strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z")
        print(id_list)
        print()


def local_mode(id_list, timezone):
    def get_measurements(number_of_steps, measurement_id):
        measure = list()
        for i in range(number_of_steps):
            time_stamp_dict[measurement_id] += time_delta_millis
            timestamp_data_string = time_stamp_dict[measurement_id].strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

            for key in key_list:
                measure.append({
                    "limb": key[1],
                    "side": key[0],
                    "timestamp": timestamp_data_string,
                    "type": key[2],
                    "x": random.random(),
                    "y": random.random(),
                    "z": random.random()
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
        time_delta_to_start = timedelta(minutes=90)
        start_timestamp = datetime.now(timezone) - time_delta_to_start
        print("start ts: {}".format(start_timestamp.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"))
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
                    print("start ts: {}".format(last_timestamp.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"))
                    steps_till_now = int(time_delta_to_start / time_delta_millis)

                test_body = get_measurements(steps_till_now, measurement_id=int(message))
                socket.send_pyobj(test_body)
            else:
                print("wrong message: {}".format(message))
                socket.send_string("wrong message: {}".format(message))




if __name__ == "__main__":
    # _config_dict = {"host_url_and_token_path": "./host_url_and_token.json"}
    # _configuration = get_configuration(_config_dict)
    _timezone = pytz.timezone("Europe/Budapest")

    _id_list = [5, 6, 7]  # [8, 9, 10]

    # normal_mode(_id_list, _timezone)
    local_mode(_id_list, _timezone)

