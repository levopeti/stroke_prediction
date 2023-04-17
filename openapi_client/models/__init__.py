# coding: utf-8

# flake8: noqa

# import all models into this package
# if you have many models here with many references from one model to another this may
# raise a RecursionError
# to avoid this, import only the models that you directly need like:
# from openapi_client.model.pet import Pet
# or import this package, but before doing it, use:
# import sys
# sys.setrecursionlimit(n)

from openapi_client.model.cleaned_data_item import CleanedDataItem
from openapi_client.model.data_item import DataItem
from openapi_client.model.date_time import DateTime
from openapi_client.model.get_measurements_result import GetMeasurementsResult
from openapi_client.model.get_predictions_result import GetPredictionsResult
from openapi_client.model.metadata import Metadata
from openapi_client.model.motionscan_endpoint_get_cleaned_measurements_content import MotionscanEndpointGetCleanedMeasurementsContent
from openapi_client.model.motionscan_endpoint_get_data_for_prediction_result import MotionscanEndpointGetDataForPredictionResult
from openapi_client.model.motionscan_endpoint_measurementids_result import MotionscanEndpointMeasurementidsResult
from openapi_client.model.motionscan_endpoint_measurements_content import MotionscanEndpointMeasurementsContent
from openapi_client.model.motionscan_endpoint_predictions_result import MotionscanEndpointPredictionsResult
from openapi_client.model.motionscan_endpoint_save_cleaned_measurements_content import MotionscanEndpointSaveCleanedMeasurementsContent
from openapi_client.model.motionscan_endpoint_save_predictions_content import MotionscanEndpointSavePredictionsContent
from openapi_client.model.open_api_endpoint_error_response import OpenAPIEndpointErrorResponse
from openapi_client.model.ping_endpoint_content import PingEndpointContent
from openapi_client.model.ping_endpoint_list_item import PingEndpointListItem
from openapi_client.model.ping_endpoint_result import PingEndpointResult
from openapi_client.model.prediction_item import PredictionItem
