import random
from pprint import pprint
from datetime import datetime, timedelta

from openapi_client import Configuration, ApiClient, ApiException
from openapi_client.apis.tags.motion_scan_restapi_end_points_api import MotionScanRESTAPIEndPointsApi
from api_utils import get_configuration


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


def local_mode(id_list):
    pass


if __name__ == "__main__":
    _config_dict = {"host_url_and_token_path": "./host_url_and_token.json"}
    _configuration = get_configuration(_config_dict)

    _id_list = [5, 6, 7]  # [8, 9, 10]

    # normal_mode(_id_list)
    local_mode(_id_list)

