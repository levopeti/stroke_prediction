import typing_extensions

from openapi_client.apis.tags import TagValues
from openapi_client.apis.tags.motion_scan_restapi_end_points_api import MotionScanRESTAPIEndPointsApi
from openapi_client.apis.tags.restapi_tools_api import RESTAPIToolsApi

TagToApi = typing_extensions.TypedDict(
    'TagToApi',
    {
        TagValues.MOTION_SCAN_REST_API_END_POINTS: MotionScanRESTAPIEndPointsApi,
        TagValues.REST_API__TOOLS: RESTAPIToolsApi,
    }
)

tag_to_api = TagToApi(
    {
        TagValues.MOTION_SCAN_REST_API_END_POINTS: MotionScanRESTAPIEndPointsApi,
        TagValues.REST_API__TOOLS: RESTAPIToolsApi,
    }
)
