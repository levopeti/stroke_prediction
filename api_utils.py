import json

from openapi_client import Configuration, ApiClient, ApiException
from openapi_client.apis.tags.motion_scan_restapi_end_points_api import MotionScanRESTAPIEndPointsApi


def get_configuration(config_dict: dict) -> Configuration:
    configuration = Configuration(host=config_dict["host_url"])
    configuration.api_key["bearer"] = config_dict["token"]
    configuration.api_key_prefix["bearer"] = "Bearer"
    return configuration


def get_measurement_ids(configuration: Configuration, _from: str, config_dict: dict) -> list:
    with ApiClient(configuration) as api_client:
        api_instance = MotionScanRESTAPIEndPointsApi(api_client)
        query_params = {
            'from': _from,
            'interval': config_dict["interval_milliseconds"],
        }
        header_params = {
            'x-motionscan-name': 'motionscandemo',
        }

        api_response = api_instance.get_measurementids(
            query_params=query_params,
            header_params=header_params,
        )
        return json.loads(api_response.response.data)["measurementids"]


def get_data_for_prediction(configuration: Configuration, _from: str, _meas_id: str, config_dict: dict) -> list:
    with ApiClient(configuration) as api_client:
        api_instance = MotionScanRESTAPIEndPointsApi(api_client)
        query_params = {
            'interval': config_dict["interval_milliseconds"],
        }

        if _from is not None:
            query_params["from"] = _from

        if _meas_id is not None:
            query_params["measurement-id"] = [_meas_id]

        header_params = {
            'x-motionscan-name': 'motionscandemo',
        }

        api_response = api_instance.get_data_for_prediction(
            query_params=query_params,
            header_params=header_params,
        )
        return json.loads(api_response.response.data)


def get_predictions_from_time_point(configuration: Configuration, _from: str, _interval: int = 15000) -> list:
    with ApiClient(configuration) as api_client:
        # Create an instance of the API class
        api_instance = MotionScanRESTAPIEndPointsApi(api_client)

        # example passing only optional values
        query_params = {
            'from': _from,
            'interval': _interval,
        }
        header_params = {
            'x-motionscan-name': 'motionscandemo',
        }
        # try:
        #     api_response = api_instance.get_data_for_prediction(
        #         query_params=query_params,
        #         header_params=header_params,
        #     )
        # except ApiException as e:
        #     print("Exception when calling MotionScanRESTAPIEndPointsApi->get_data_for_prediction: %s\n" % e)
        api_response = api_instance.get_predictions_from_timepoint(
            query_params=query_params,
            header_params=header_params,
        )
        return json.loads(api_response.response.data)


def save_predictions(configuration: Configuration, body: dict):
    with ApiClient(configuration) as api_client:
        # Create an instance of the API class
        api_instance = MotionScanRESTAPIEndPointsApi(api_client)
        # example passing only required values which don't have defaults set
        header_params = {
            'x-motionscan-name': 'motionscandemo',
        }

        # try:
        #     api_instance.save_predictions(
        #         header_params=header_params,
        #         body=body,
        #     )
        # except ApiException as e:
        #     print("Exception when calling MotionScanRESTAPIEndPointsApi->save_predictions: %s\n" % e)
        api_instance.save_predictions(
            header_params=header_params,
            body=body,
        )

