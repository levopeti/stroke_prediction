from abc import ABC, abstractmethod
from measurement_utils.measurement import Measurement
from measurement_utils.measurement_info import MeasurementInfoManager


class Model(ABC):
    # @abstractmethod
    # def get_instances(self, measurement: Measurement, config_dict: dict):
    #     pass

    @abstractmethod
    def compute_prediction(self, measurement: Measurement):
        """
        return dict: {"probabilities": list of floats,
                      "labels": list of ints,
                      "is_stroke": list of bools,
                      "timestamps": list of ints}
        """
        pass

    @abstractmethod
    def add_meas_info_manager(self, meas_info_manager: MeasurementInfoManager):
        pass


