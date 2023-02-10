# do not import all endpoints into this module because that uses a lot of memory and stack frames
# if you need the ability to import all endpoints from this module, import them with
# from openapi_client.apis.tag_to_api import tag_to_api

import enum


class TagValues(str, enum.Enum):
    MOTION_SCAN_REST_API_END_POINTS = "MotionScan REST API EndPoints"
    REST_API__TOOLS = "REST API - Tools"
