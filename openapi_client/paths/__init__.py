# do not import all endpoints into this module because that uses a lot of memory and stack frames
# if you need the ability to import all endpoints from this module, import them with
# from openapi_client.apis.path_to_api import path_to_api

import enum


class PathValues(str, enum.Enum):
    SAVEMEASUREMENTS = "/save-measurements"
    GETMEASUREMENTS = "/get-measurements"
    GETMEASUREMENTIDS = "/get-measurementids"
    GETPREDICTIONSFROMTIMEPOINT = "/get-predictions-from-timepoint"
    GETDATAFORPREDICTION = "/get-data-for-prediction"
    SAVEPREDICTIONS = "/save-predictions"
    GETPREDICTIONS = "/get-predictions"
    GETCLEANEDMEASUREMENTS = "/get-cleanedmeasurements"
    SAVECLEANEDMEASUREMENTS = "/save-cleanedmeasurements"
    PING = "/ping"
