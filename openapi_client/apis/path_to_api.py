import typing_extensions

from openapi_client.paths import PathValues
from openapi_client.apis.paths.save_measurements import SaveMeasurements
from openapi_client.apis.paths.get_predictions_from_timepoint import GetPredictionsFromTimepoint
from openapi_client.apis.paths.get_data_for_prediction import GetDataForPrediction
from openapi_client.apis.paths.save_predictions import SavePredictions
from openapi_client.apis.paths.ping import Ping

PathToApi = typing_extensions.TypedDict(
    'PathToApi',
    {
        PathValues.SAVEMEASUREMENTS: SaveMeasurements,
        PathValues.GETPREDICTIONSFROMTIMEPOINT: GetPredictionsFromTimepoint,
        PathValues.GETDATAFORPREDICTION: GetDataForPrediction,
        PathValues.SAVEPREDICTIONS: SavePredictions,
        PathValues.PING: Ping,
    }
)

path_to_api = PathToApi(
    {
        PathValues.SAVEMEASUREMENTS: SaveMeasurements,
        PathValues.GETPREDICTIONSFROMTIMEPOINT: GetPredictionsFromTimepoint,
        PathValues.GETDATAFORPREDICTION: GetDataForPrediction,
        PathValues.SAVEPREDICTIONS: SavePredictions,
        PathValues.PING: Ping,
    }
)
