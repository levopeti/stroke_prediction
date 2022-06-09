import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

from ..measurement.measurement_collector import MeasurementCollector
from ..utils.confusion_matrix_keras import plot_cm_keras
from ..utils.cache_utils import cache

# milliseconds between two measurements
TIME_DELTA = 25


def get_data(_mc, _param_dict):
    minutes = _param_dict["minutes"]
    length = TIME_DELTA * 60 * minutes
    sample_size = _param_dict["sample_size"]
    limb = _param_dict["limb"]

    keys_in_order = (("arm", "acc"),
                     ("leg", "acc"),
                     ("arm", "gyr"),
                     ("leg", "gyr"))

    X = list()
    y = list()
    for _ in tqdm(range(sample_size)):
        random_diff_dict, class_value = _mc.get_random_mean_with_class_all(mean_type='all', limb=limb,
                                                                           length=length, type_of_set="train")
        instance = list()
        for key in keys_in_order:
            if limb != "all" and key[0] != limb:
                continue
            instance.append(random_diff_dict[key])
        instance = sum(instance, [])

        X.append(instance)
        y.append(class_value)

    # if save_data:
    #     pickle.dump({"X": X, "y": y},
    #                 open("./training_data_dict_{}_{}_{}.pkl".format(minutes, sample_size, limb), "wb"))

    y_cat = to_categorical(y)
    X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y_cat),
                                                        test_size=0.3, stratify=y,
                                                        random_state=42)

    return X_train, X_test, y_train, y_test


def define_model(input_shape):
    ip = Input(shape=(input_shape,), name="input")
    x = Dense(units=512, name="hidden_layer", activation="relu")(ip)
    x = Dense(units=256, name="hidden_layer_2", activation="relu")(x)
    x = Dense(units=128, name="hidden_layer_3", activation="relu")(x)
    op = Dense(units=6, name="prediction", activation="softmax")(x)
    _model = Model(inputs=ip, outputs=op, name="full_model")
    _model.summary()

    learning_rate = 0.001
    optimizer = Adam(learning_rate, amsgrad=True)
    _model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return _model


def fit_model(_model, X_train, X_test, y_train, y_test):
    checkpoint_filepath = '/tmp/checkpoint'
    cp = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='auto',
        save_best_only=True)

    es = EarlyStopping(monitor='val_loss', patience=20)
    history = _model.fit(X_train,
                         y_train,
                         batch_size=100,
                         epochs=1500,
                         callbacks=[es, cp],
                         verbose=1,
                         validation_data=(X_test, y_test))

    return _model, history


def get_accuracy_and_cm(_model, X_train, X_test, y_train, y_test):
    y_pred_train = _model.predict(X_train)
    print("Train accuracy: {}".format(accuracy_score(np.argmax(y_train, axis=1),
                                                     np.argmax(y_pred_train, axis=1))))
    y_pred_test = _model.predict(X_test)
    print("Test accuracy: {}".format(accuracy_score(np.argmax(y_test, axis=1),
                                                    np.argmax(y_pred_test, axis=1))))
    plot_cm_keras(_model, X_train, y_train)
    plot_cm_keras(_model, X_test, y_test)


def start_training(_param_dict):
    _db_path = "/home/levcsi/projects/stroke_prediction/data/WUS-v4m.accdb"
    _m_path = "/home/levcsi/projects/stroke_prediction/data/biocal.xlsx"
    mc = MeasurementCollector('/home/levcsi/projects/stroke_prediction/data', _db_path, _m_path)

    X_train, X_test, y_train, y_test = get_data(mc, _param_dict)
    model = define_model(input_shape=X_train.shape[1])
    model, history = fit_model(model, X_train, X_test, y_train, y_test)
    get_accuracy_and_cm(model, X_train, X_test, y_train, y_test)

    return model, history


if __name__ == "__main__":
    param_dict = {
        "minutes": 90,
        "sample_size": 1000000,
        "limb": "all",
    }

    start_training(param_dict)
