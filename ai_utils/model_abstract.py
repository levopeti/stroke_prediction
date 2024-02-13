import numpy as np
from abc import ABC, abstractmethod

from measurement_utils.measurement import Measurement


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


