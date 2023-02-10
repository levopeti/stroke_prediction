### Example

# * Api Key Authentication (bearer):
# ```python

import random
from pprint import pprint
from datetime import datetime
from datetime import timedelta
from time import time

import openapi_client
from openapi_client.apis.tags import motion_scan_restapi_end_points_api
from openapi_client.model.open_api_endpoint_error_response import OpenAPIEndpointErrorResponse

# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = openapi_client.Configuration(
    # host="https://api.int.ms.salusmo.euronetrt.hu"
    host="https://api.test.ms.salusmo.euronetrt.hu"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure API key authorization: bearer
# configuration.api_key['bearer'] = '2H0w0soY8U3gX+UoujWB9p85ByCa5yDV/5iRpnlLVepB9lTLmX9XpP6jSlvRF17p'
configuration.api_key['bearer'] = 'nRYUakaQTdDQyy-PmYlVTIcZRwYvNmZsmGrD6YApvsxTniTghB8RsQZet3fIs95LUP1YSeCM-LQRsdhlrxRNx9ixk60mpcH5CLp9wqUHiDPu2wxKDOZVCJqsach8B9H5'

# Uncomment below to setup prefix (e.g. Bearer) for API key, if needed
configuration.api_key_prefix['bearer'] = 'Bearer'
# Enter a context with an instance of the API client


data_points = int(1e2)
timestamp_data = datetime.now()
timestamp_data_string = timestamp_data.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
print(timestamp_data_string)

start = time()

measure = list()
timestamp_data = datetime.now()
time_delta_ms = 40
for i in range(data_points):
    timestamp_data += timedelta(milliseconds=time_delta_ms)
    timestamp_data_string = timestamp_data.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    measure.append({
        "limb": random.choice("al"),
        "side": random.choice("lr"),
        "timestamp": timestamp_data_string,
        "type": random.choice("ag"),
        "x": random.random(),
        "y": random.random(),
        "z": random.random()
    })

test_body = {
    "measure": measure,
    "measurementId": "000001",
    "softwareVersion": "Data Collector 1.0",
    "APIVersion": "MotionScan API 1.0"
}

# save_measurements ###

with openapi_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = motion_scan_restapi_end_points_api.MotionScanRESTAPIEndPointsApi(api_client)

    # example passing only required values which don't have defaults set
    header_params = {
        'x-motionscan-name': 'motionscandemo',
    }
    body = test_body
    try:
        api_response = api_instance.save_measurements(
            header_params=header_params,
            body=body,
        )
        pprint(api_response)
        print("\n")
        pprint(api_response.response.reason)
        print("\n")
    except openapi_client.ApiException as e:
        print("Exception when calling MotionScanRESTAPIEndPointsApi->save_measurements: %s\n" % e)

end = time()
print(f'{data_points} db measure upload: {end - start} sec!\n')

start = time()

predictions = list()
for i in range(data_points):
    timestamp_data = datetime.now() + timedelta(milliseconds=i)
    timestamp_data_string = timestamp_data.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    predictions.append({
        "prediction": random.choice(["ok", "stroke"]),
        "probability": random.random(),
        "timestamp": timestamp_data_string,
    })

# print(predictions)

test_body = {
    "predictions": predictions,
    "measurementId": "000001",
    "softwareVersion": "Predictor 1.0",
    "APIVersion": "MotionScan API 1.0"
}

# save_predictions ###

with openapi_client.ApiClient(configuration) as api_client:
    # Create an instance of the AP
    # example passing only required values which don't have defaults set
    header_params = {
        'x-motionscan-name': 'motionscandemo',
    }
    body = test_body
    try:
        api_response = api_instance.save_predictions(
            header_params=header_params,
            body=body,
        )
        pprint(api_response)
    except openapi_client.ApiException as e:
        print("Exception when calling MotionScanRESTAPIEndPointsApi->save_predictions: %s\n" % e)

end = time()
print(f'{data_points} db prediction: {end - start} sec!\n')

start = time()

# get_predictions_from_timepoint ###

with openapi_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = motion_scan_restapi_end_points_api.MotionScanRESTAPIEndPointsApi(api_client)

    # example passing only required values which don't have defaults set
    query_params = {
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
        print("\n")
        pprint(api_response.response.reason)
        print("\n")
    except openapi_client.ApiException as e:
        print("Exception when calling MotionScanRESTAPIEndPointsApi->get_predictions_from_timepoint: %s\n" % e)

    # example passing only optional values
    query_params = {
        'from': "2022-12-05T23:49:09.117Z",
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
print(f'{data_points} db prediction request: {end - start} sec!\n')

start = time()

# get_data_for_prediction ###

with openapi_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = motion_scan_restapi_end_points_api.MotionScanRESTAPIEndPointsApi(api_client)

    # example passing only required values which don't have defaults set
    query_params = {
    }
    header_params = {
        'x-motionscan-name': 'motionscandemo',
    }
    try:
        api_response = api_instance.get_data_for_prediction(
            query_params=query_params,
            header_params=header_params,
        )
        pprint(api_response)
    except openapi_client.ApiException as e:
        print("Exception when calling MotionScanRESTAPIEndPointsApi->get_data_for_prediction: %s\n" % e)

    # example passing only optional values
    query_params = {
        # 'from': "2022-12-06T07:48:42.382Z",
        'from': "2023-01-28T23:14:01.327Z",
        'interval': 15000,
    }
    header_params = {
        'x-motionscan-name': 'motionscandemo',
    }
    try:
        api_response = api_instance.get_data_for_prediction(
            query_params=query_params,
            header_params=header_params,
        )
        pprint(api_response)
        print("\n")
        pprint(api_response.response.reason)
        print("\n")
    except openapi_client.ApiException as e:
        print("Exception when calling MotionScanRESTAPIEndPointsApi->get_data_for_prediction: %s\n" % e)

end = time()
print(f'{data_points} db data request for prediction: {end - start} sec!\n')
