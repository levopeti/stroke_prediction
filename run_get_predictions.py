import json
from pprint import pprint

import openapi_client

from time import time, sleep
from datetime import datetime, timedelta

from general_utils import min_to_millisec
from openapi_client import Configuration, ApiClient, ApiException
from openapi_client.apis.tags import motion_scan_restapi_end_points_api
from openapi_client.apis.tags.motion_scan_restapi_end_points_api import MotionScanRESTAPIEndPointsApi
from api_utils import get_configuration

def get_prediction(_timestamp_data_string, _interval):
    start = time()

    with openapi_client.ApiClient(configuration) as api_client:
        # Create an instance of the API class
        api_instance = motion_scan_restapi_end_points_api.MotionScanRESTAPIEndPointsApi(api_client)

        # example passing only optional values
        query_params = {
            'from': _timestamp_data_string,  # "2022-12-05T23:49:09.117Z",
            'interval': _interval,
        }
        header_params = {
            'x-motionscan-name': 'motionscandemo',
        }
        try:
            _api_response = api_instance.get_predictions_from_timepoint(
                query_params=query_params,
                header_params=header_params,
            )
            # pprint(api_response)
        except openapi_client.ApiException as e:
            print("Exception when calling MotionScanRESTAPIEndPointsApi->get_predictions_from_timepoint: %s\n" % e)

    end = time()
    return _api_response, end - start


_host_url = "https://api.test.ms.salusmo.euronetrt.hu"
_token = "nRYUakaQTdDQyy-PmYlVTIcZRwYvNmZsmGrD6YApvsxTniTghB8RsQZet3fIs95LUP1YSeCM-LQRsdhlrxRNx9ixk60mp" \
         "cH5CLp9wqUHiDPu2wxKDOZVCJqsach8B9H5"
_config_dict = {"host_url": _host_url,
                "token": _token}
configuration = get_configuration(_config_dict)

timestamp_data = datetime.now() - timedelta(minutes=120)
timestamp_data_string = timestamp_data.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
print(timestamp_data_string)
interval = min_to_millisec(120)

while True:
    api_response, elapsed_time = get_prediction(timestamp_data_string, interval)
    # pprint(json.loads(api_response.response.data))

    meas_ids = set()
    for pred_dict in json.loads(api_response.response.data):
        meas_ids.add(pred_dict["measurementId"])
        if pred_dict["measurementId"] == "1":
            print(pred_dict)

    print(meas_ids)
    print(f'{len(json.loads(api_response.response.data))} items: {elapsed_time} sec!\n')
    sleep(5)

