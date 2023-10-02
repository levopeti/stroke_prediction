from datetime import datetime

import numpy as np

from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

from ai_utils.training_utils.clear_measurements import ClearMeasurements
from ai_utils.training_utils.func_utils import get_input_from_df
from measurement_utils.measure_db import MeasureDB


def define_model(input_shape, output_shape, layer_sizes):
    assert len(layer_sizes) > 0, layer_sizes

    ip = Input(shape=(input_shape,), name="input")
    x = Dense(units=layer_sizes[0], name="hidden_layer", activation="relu")(ip)

    for i, layer_size in enumerate(layer_sizes[1:]):
        x = Dense(units=layer_size, name="hidden_layer_{}".format(i + 2), activation="relu")(x)

    op = Dense(units=output_shape, name="prediction", activation="softmax")(x)
    _model = Model(inputs=ip, outputs=op, name="full_model")
    _model.summary()

    learning_rate = 0.001
    optimizer = Adam(learning_rate, amsgrad=True)
    _model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return _model


class DataGenerator(Sequence):
    def __init__(self,
                 data_type: str,  # train or test
                 clear_measurements: ClearMeasurements,
                 measDB: MeasureDB,
                 batch_size: int,
                 n_classes: int,
                 length: int,
                 sample_per_meas: int) -> None:
        self.batch_size = batch_size

        self.meas_id_list = clear_measurements.get_meas_id_list(data_type)
        self.clear_measurements = clear_measurements
        self.measDB = measDB
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

            class_value_dict = self.measDB.get_class_value_dict(meas_id=meas_id)
            input_array = get_input_from_df(meas_df, self.length, class_value_dict)
            label = min(class_value_dict.values())

            batch_array.append(input_array)
            labels.append(label)
        return np.concatenate(batch_array, axis=0), to_categorical(labels, num_classes=self.n_classes)


if __name__ == "__main__":
    db_path = "./data/WUS-v4measure202307311.accdb"
    ucanaccess_path = "./ucanaccess/"
    folder_path = "./data/clear_data/"
    clear_json_path = "./data/clear_train_test_ids.json"
    now = datetime.now().strftime('%Y-%m-%d-%H-%M')
    model_base_path = "./models/{}".format(now)
    print(model_base_path)

    length = int(1.5 * 60 * 60 * 25)  # 1.5 hours, 25 Hz
    sample_per_meas = 3
    batch_size = 12
    input_size = 12
    layer_sizes = [1024, 512, 128]
    output_size = 6

    measDB = MeasureDB(db_path, ucanaccess_path)
    clear_measurements = ClearMeasurements(folder_path, clear_json_path, cache_size=18)

    # Generators
    training_generator = DataGenerator("train", clear_measurements, measDB, batch_size, output_size, length, sample_per_meas)
    test_generator = DataGenerator("test", clear_measurements, measDB, batch_size, output_size, length, sample_per_meas)

    lr = 0.001
    wd = 0
    num_epoch = 1000

    # Design model
    model = define_model(input_size, output_size, layer_sizes)

    cp = ModelCheckpoint(
        filepath=model_base_path,
        save_weights_only=False,
        monitor='val_loss',
        mode='auto',
        save_best_only=True)

    es = EarlyStopping(monitor='val_loss', patience=20)

    # Train model on dataset
    model.fit_generator(generator=training_generator,
                        validation_data=test_generator,
                        epochs=num_epoch,
                        callbacks=[es, cp],
                        shuffle=False,
                        use_multiprocessing=False,
                        workers=6)
