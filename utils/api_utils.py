import json

from time import time
from typing import Tuple

from openapi_client import Configuration, ApiClient, ApiException
from openapi_client.apis.tags.motion_scan_restapi_end_points_api import MotionScanRESTAPIEndPointsApi
from utils.general_utils import to_str_timestamp
from utils.log_maker import write_log


def get_configuration(config_dict: dict) -> Configuration:
    with open(config_dict["host_url_and_token_path"]) as f:
        host_url_and_token = json.load(f)

    configuration = Configuration(host=host_url_and_token["host_url"])
    configuration.api_key["bearer"] = host_url_and_token["token"]
    configuration.api_key_prefix["bearer"] = "Bearer"
    return configuration


def get_measurement_ids(configuration: Configuration, _from: str, _interval: int) -> list:
    with ApiClient(configuration) as api_client:
        api_instance = MotionScanRESTAPIEndPointsApi(api_client)
        query_params = {
            'from': _from,
            'interval': _interval,
            # 'interval': config_dict["interval_milliseconds"],
        }
        header_params = {
            'x-motionscan-name': 'motionscandemo',
        }

        api_response = api_instance.get_measurementids(
            query_params=query_params,
            header_params=header_params,
        )
        return json.loads(api_response.response.data)["measurementids"]


def get_data_for_prediction(configuration: Configuration,
                            _from: str,
                            _meas_id: str,
                            _interval: int) -> Tuple[list, float]:
    start = time()
    with ApiClient(configuration) as api_client:
        api_instance = MotionScanRESTAPIEndPointsApi(api_client)
        query_params = {
            'interval': _interval,
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
        return json.loads(api_response.response.data), time() - start


def get_predictions_from_time_point(configuration: Configuration,
                                    _from: str,
                                    _interval: int = 15000,
                                    _measurement_id: str = None) -> list:
    with ApiClient(configuration) as api_client:
        # Create an instance of the API class
        api_instance = MotionScanRESTAPIEndPointsApi(api_client)

        # example passing only optional values
        query_params = {
            'from': _from,
            'interval': _interval,
        }

        if _measurement_id is not None:
            query_params["measurement-id"] = _measurement_id

        header_params = {
            'x-motionscan-name': 'motionscandemo',
        }
        try:
            api_response = api_instance.get_predictions_from_timepoint(
                query_params=query_params,
                header_params=header_params,
            )
        except ApiException as e:
            # print("Exception when calling MotionScanRESTAPIEndPointsApi->get_data_for_prediction: %s\n" % e)
            write_log("python_api.txt", "Exception when calling MotionScanRESTAPIEndPointsApi->get_predictions_from_timepoint: %s\n" % e,
                      title="GetPred", print_out=True, color="red", add_date=True)

        return json.loads(api_response.response.data)


def save_predictions(configuration: Configuration, body: dict):
    with ApiClient(configuration) as api_client:
        api_instance = MotionScanRESTAPIEndPointsApi(api_client)
        header_params = {
            'x-motionscan-name': 'motionscandemo',
        }

        try:
            x = api_instance.save_predictions(
                header_params=header_params,
                body=body,
            )
            # print(x)
        except ApiException as e:
            # print("Exception when calling MotionScanRESTAPIEndPointsApi->get_data_for_prediction: %s\n" % e)
            write_log("python_api.txt", "Exception when calling MotionScanRESTAPIEndPointsApi->get_data_for_prediction: %s\n" % e,
                      title="SavePred", print_out=True, color="red", add_date=True)
            raise Exception("Exception when calling MotionScanRESTAPIEndPointsApi->get_data_for_prediction: %s\n" % e)


def upload_prediction(configuration: Configuration, prediction_dict: dict, measurement_id: str):
    start = time()
    predictions = list()
    for i in range(len(prediction_dict["is_stroke"])):
        if isinstance(prediction_dict["is_stroke"][i], str):
            # error message
            prediction = prediction_dict["is_stroke"][i]
        else:
            assert isinstance(prediction_dict["is_stroke"][i], bool)
            prediction = "stroke" if prediction_dict["is_stroke"][i] else "ok"

        predictions.append({
            "prediction": prediction,
            "probability": float(prediction_dict["probabilities"][i]),
            "timestamp": to_str_timestamp(prediction_dict["timestamps"][i]),
        })

    _body = {
        "predictions": predictions,
        "measurementId": measurement_id,
        "softwareVersion": "Predictor 1.0",
        "APIVersion": "MotionScan API 1.0"
    }
    save_predictions(configuration, _body)
    print("uploaded {} prediction(s) with measurement id {} ({:.0}s)".format(len(predictions), measurement_id,
                                                                             time() - start))
