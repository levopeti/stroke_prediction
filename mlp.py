import numpy as np
from tensorflow import keras


class MLP(object):
    def __init__(self, model_path):
        self.model = keras.models.load_model(model_path)
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

