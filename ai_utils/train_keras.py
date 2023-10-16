import gc
import json
import os
import pickle

import numpy as np

from pprint import pprint
from datetime import datetime
# from pympler.asizeof import asizeof

# import tensorflow as tf

# tf.compat.v1.disable_eager_execution()

# from tensorflow.keras import backend as k
# from tensorflow.keras.callbacks import Callback
# from tensorflow.keras.utils import Sequence
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Dense, ReLU
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# from tensorflow.keras.utils import to_categorical

from keras import backend as k
from keras.callbacks import Callback
from keras.utils import Sequence
from keras.models import Model
from keras.layers import Input, Dense, ReLU
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical

from ai_utils.training_utils.clear_measurements import ClearMeasurements
from ai_utils.training_utils.func_utils import get_input_from_df
from ai_utils.training_utils.loss_and_accuracy import stroke_loss_reg, stroke_loss_clas, \
    stroke_accuracy_reg, stroke_accuracy_clas
from measurement_utils.measure_db import MeasureDB


# tf.config.run_functions_eagerly(True)


def define_model(input_shape, output_shape, layer_sizes, learning_rate, stroke_loss_factor, **kwargs):
    assert len(layer_sizes) > 0, layer_sizes

    ip = Input(shape=(input_shape,), name="input")
    x = Dense(units=layer_sizes[0], name="hidden_layer", activation=None)(ip)  # "relu"
    x = ReLU()(x)

    for i, layer_size in enumerate(layer_sizes[1:]):
        x = Dense(units=layer_size, name="hidden_layer_{}".format(i + 2), activation=None)(x)  # "relu"
        x = ReLU()(x)

    if output_shape == 1:
        op = Dense(units=output_shape, name="prediction", activation="sigmoid")(x)
        custom_loss = stroke_loss_reg(stroke_loss_factor)
        stroke_accuracy = stroke_accuracy_reg
    else:
        op = Dense(units=output_shape, name="prediction", activation="softmax")(x)
        custom_loss = stroke_loss_clas(stroke_loss_factor)
        stroke_accuracy = stroke_accuracy_clas

    _model = Model(inputs=ip, outputs=op, name="full_model")
    _model.summary()

    optimizer = Adam(learning_rate, amsgrad=True)
    _model.compile(loss=custom_loss,
                   optimizer=optimizer,
                   # run_eagerly=True,
                   metrics=["accuracy", stroke_accuracy])  # "categorical_crossentropy"
    return _model


def save_params(_params: dict):
    os.makedirs(_params["model_base_path"], exist_ok=True)
    with open(os.path.join(_params["model_base_path"], "params.json"), "w") as f:
        json.dump(_params, f)


class DataGenerator(Sequence):
    def __init__(self,
                 data_type: str,  # train or test
                 clear_measurements: ClearMeasurements,
                 batch_size: int,
                 n_classes: int,
                 length: int,
                 sample_per_meas: int) -> None:
        self.batch_size = batch_size

        self.meas_id_list = clear_measurements.get_meas_id_list(data_type)
        self.clear_measurements = clear_measurements
        self.sample_per_meas = sample_per_meas
        self.length = length
        self.n_classes = n_classes

    def __len__(self):
        return int(len(self.meas_id_list) * self.sample_per_meas / self.batch_size)

    def __getitem__(self, batch_idx):
        batch_array = list()
        labels = list()
        for idx in range(batch_idx * self.batch_size, (batch_idx + 1) * self.batch_size):
            meas_idx = idx // self.sample_per_meas
            meas_id = self.meas_id_list[meas_idx]
            meas_df = self.clear_measurements.get_measurement(meas_id)

            class_value_dict = self.clear_measurements.get_class_value_dict(meas_id=meas_id)
            input_array = get_input_from_df(meas_df, self.length, class_value_dict)
            label = self.clear_measurements.get_min_class_value(meas_id)

            batch_array.append(input_array)
            labels.append(label)

        if self.n_classes != 1:
            labels = to_categorical(labels, num_classes=self.n_classes)
        else:
            labels = np.expand_dims(np.array(labels), axis=1)

        # print(" {}: {:.2f}".format(type(self).__name__, asizeof(self) / 1e6))
        return np.concatenate(batch_array, axis=0), labels


class ClearMemory(Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        k.clear_session()


if __name__ == "__main__":
    params = {"accdb_path": "./data/WUS-v4measure202307311.accdb",
              "ucanaccess_path": "./ucanaccess/",
              "folder_path": "./data/clear_data/",
              "clear_json_path": "./data/clear_train_test_ids.json",
              "model_base_path": "./models/{}".format(datetime.now().strftime('%Y-%m-%d-%H-%M')),
              "length": int(1.5 * 60 * 60 * 25),  # 1.5 hours, 25 Hz
              "train_sample_per_meas": 10,
              "test_sample_per_meas": 100,
              "train_batch_size": 100,
              "test_batch_size": 100,
              "input_shape": 12,
              "output_shape": 1,
              "layer_sizes": [1024, 512, 256],
              "patience": 20,
              "learning_rate": 0.001,
              "wd": 0,
              "num_epoch": 1000,
              "steps_per_epoch": 100,
              "stroke_loss_factor": 0.1,
              "cache_size": 1
              }

    pprint(params)
    measDB = MeasureDB(params["accdb_path"], params["ucanaccess_path"])
    clear_measurements = ClearMeasurements(measDB, params["folder_path"], params["clear_json_path"],
                                           cache_size=params["cache_size"])
    clear_measurements.print_stat()

    params["train_id_list"] = clear_measurements.get_meas_id_list("train")
    params["test_id_list"] = clear_measurements.get_meas_id_list("test")
    save_params(params)

    # Generators
    training_generator = DataGenerator("train", clear_measurements, params["train_batch_size"], params["output_shape"],
                                       params["length"], params["train_sample_per_meas"])
    test_generator = DataGenerator("test", clear_measurements, params["test_batch_size"], params["output_shape"],
                                   params["length"], params["test_sample_per_meas"])

    model = define_model(**params)

    cp = ModelCheckpoint(
        filepath=params["model_base_path"],
        save_weights_only=False,
        monitor='loss',
        mode='auto',
        save_best_only=True)

    es = EarlyStopping(monitor='loss', patience=params["patience"])
    # cm = ClearMemory()

    # Train model on dataset
    history = model.fit_generator(generator=training_generator,
                                  validation_data=test_generator,
                                  steps_per_epoch=params["steps_per_epoch"],
                                  epochs=params["num_epoch"],
                                  callbacks=[es, cp],  # , cm
                                  shuffle=False,
                                  use_multiprocessing=False,
                                  workers=6)

    # save model
    # model.save(os.path.join(params["model_base_path"], "model.keras"))

    # save history
    with open(os.path.join(params["model_base_path"], "history.pkl"), "wb") as file:
        pickle.dump(history.history, file)
