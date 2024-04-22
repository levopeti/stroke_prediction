from typing import List
from measurement_utils.measurement import Measurement


class MeasurementInfoManager(object):
    def __init__(self, config_dict: dict):
        self.config_dict = config_dict
        self.measurement_dict = dict()

    @staticmethod
    def collect_info(measurement: Measurement, keys: List[tuple]) -> None:
        for key in keys:
            # "timestamp_ms", "x", "y", "z"
            df = measurement.measurement_dict[key]

    def add_new_measurement(self, measurement_id: str) -> None:
        assert measurement_id not in self.measurement_dict
        self.measurement_dict[measurement_id] = dict()

    def add_info(self, measurement_id: str, info_key: str, measurement_info) -> None:
        self.measurement_dict[measurement_id][info_key] = measurement_info

    def del_measurement(self, measurement_id: str) -> None:
        del self.measurement_dict[measurement_id]



