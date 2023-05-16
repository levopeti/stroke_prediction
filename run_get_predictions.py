import json
from pprint import pprint

import openapi_client

from time import time, sleep
from datetime import datetime, timedelta
from openapi_client import Configuration, ApiClient, ApiException
from openapi_client.apis.tags import motion_scan_restapi_end_points_api
from openapi_client.apis.tags.motion_scan_restapi_end_points_api import MotionScanRESTAPIEndPointsApi
from api_utils import get_configuration

_host_url = "https://api.test.ms.salusmo.euronetrt.hu"
_token = "nRYUakaQTdDQyy-PmYlVTIcZRwYvNmZsmGrD6YApvsxTniTghB8RsQZet3fIs95LUP1YSeCM-LQRsdhlrxRNx9ixk60mp" \
         "cH5CLp9wqUHiDPu2wxKDOZVCJqsach8B9H5"
_config_dict = {"host_url": _host_url,
                "token": _token}
configuration = get_configuration(_config_dict)

timestamp_data = datetime.now()
timestamp_data_string = timestamp_data.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
print(timestamp_data_string)
time_delta_millis = 40

uploaded_data = 0
while True:
    start = time()

    with openapi_client.ApiClient(configuration) as api_client:
        # Create an instance of the API class
        api_instance = motion_scan_restapi_end_points_api.MotionScanRESTAPIEndPointsApi(api_client)

        # example passing only optional values
        query_params = {
            'from': timestamp_data_string,  # "2022-12-05T23:49:09.117Z",
            'interval': 15000,
        }
        header_params = {
            'x-motionscan-name': 'motionscandemo',
        }
        try:
            api_response = api_instance.get_predictions_from_timepoint(
                query_params=query_params,
                header_params=header_params,
            )
            pprint(api_response)
        except openapi_client.ApiException as e:
            print("Exception when calling MotionScanRESTAPIEndPointsApi->get_predictions_from_timepoint: %s\n" % e)

    end = time()
    pprint(api_response.response.data)
    print(f'{len(json.loads(api_response.response.data))} items: {end - start} sec!\n')

