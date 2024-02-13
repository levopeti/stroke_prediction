import numpy as np
from tensorflow import keras


class MLP(object):
    def __init__(self, config_dict):
        if config_dict["mocked_model"]:
            def mocked_prediction(input_array: np.ndarray) -> np.ndarray:
                """ If every input value is zero it returns with STROKE otherwise OK"""
                output_array = np.zeros_like(input_array)

                if input_array.sum() == input_array.shape[0] * 8:
                    # mocked input for one timestamp:[0. 1. 1. 0. 1. 1. 0. 1. 1. 0. 1. 1.]
                    # stroke
                    output_array[:, 0] = 1
                else:
                    # not stroke (OK) -> class 5
                    output_array[:, 5] = 1
                return output_array

            # create object with one prediction method defined above
            self.model = type("mocked_model", (object,), dict(predict=mocked_prediction))

        else:
            self.model = keras.models.load_model(config_dict["model_path"])
            self.model.summary()

    @staticmethod
    def preprocessing(input_data):
        if not isinstance(input_data, np.ndarray):
            input_data = np.array(input_data)
        return input_data

    def predict(self, input_data):
        y_pred_list = self.model.predict(input_data)
        labels = np.array(y_pred_list).argmax(axis=1)
        probabilities = [y_pred_list[i][labels[i]] for i in range(len(labels))]

        threshold = 4.5
        is_stroke = labels < threshold
        return {"probabilities": probabilities, "labels": labels.tolist(), "is_stroke": is_stroke.tolist()}

    def compute_prediction(self, input_data, timestamps):
        preprocessed_data = self.preprocessing(input_data)
        result_dict = self.predict(preprocessed_data)
        result_dict["timestamps"] = timestamps
        return result_dict

