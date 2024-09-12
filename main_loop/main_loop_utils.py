from typing import Union, Tuple

import numpy as np

from measurement_utils.measurement import Measurement, SynchronizationError
from measurement_utils.measurement_manager import MeasurementManager
from utils.general_utils import to_str_timestamp, min_to_millisec


def make_error_body(error_code: str, measurement_id: str, last_ts: int) -> dict:
    predictions = [{"prediction": error_code,
                    "probability": 1.0,
                    "timestamp": to_str_timestamp(last_ts)
                    }]

    body = {
        "predictions": predictions,
        "measurementId": measurement_id,
        "softwareVersion": "Predictor 1.0",
        "APIVersion": "MotionScan API 1.0"
    }
    return body


def make_body(prediction_dict: dict, measurement_id: str):
    predictions = list()
    for i in range(len(prediction_dict["is_stroke"])):
        prediction = "stroke" if prediction_dict["is_stroke"][i] else "ok"
        predictions.append({
            "prediction": prediction,
            "probability": float(prediction_dict["probabilities"][i]),
            "timestamp": to_str_timestamp(prediction_dict["timestamps"][i]),
        })

    body = {
        "predictions": predictions,
        "measurementId": measurement_id,
        "softwareVersion": "Predictor 1.0",
        "APIVersion": "MotionScan API 1.0"
    }
    return body


def get_measurement(mm: MeasurementManager, measurement_id: str) -> Union[Measurement, None]:
    meas = Measurement(measurement_id)
    meas_df = mm.get_df(measurement_id)

    if meas_df is None:
        return None
    else:
        meas.fill_from_df(meas_df)
        return meas


def interpolate_measurements(meas: Measurement, min_diff: int, max_diff: int) -> None:
    for key, df in meas.measurement_dict.items():
        if df is not None:
            is_it_done = False
            while not is_it_done:
                timestamps = df["timestamp_ms"].values
                x_axis = df["x"].values
                y_axis = df["y"].values
                z_axis = df["z"].values
                diff_array = np.diff(timestamps)
                mask = np.logical_and(diff_array > min_diff, diff_array < max_diff)
                indices = np.where(mask)[0]  # get indices of true values
                # print("{}: {}".format(key, len(indices)))
                # if len(indices) > 1:
                #     print("more then one diff are found with key {}".format(key))
                #     ind = indices[0]
                # elif len(indices) == 1:
                #     ind = indices[0]
                # else:
                #     is_it_done = True
                #     continue
                if len(indices) > 0:
                    ind = indices[0]
                    interpolated_ts = int((timestamps[ind] + timestamps[ind + 1]) / 2)
                    interpolated_x = (x_axis[ind] + x_axis[ind + 1]) / 2
                    interpolated_y = (y_axis[ind] + y_axis[ind + 1]) / 2
                    interpolated_z = (z_axis[ind] + z_axis[ind + 1]) / 2

                    df.loc[df.index[ind] + 0.5] = [interpolated_ts, interpolated_x, interpolated_y, interpolated_z]
                    df = df.sort_index().reset_index(drop=True)
                    df["timestamp_ms"] = df["timestamp_ms"].astype("int")
                    meas.measurement_dict[key] = df
                else:
                    is_it_done = True


def check_and_synch_measurement(measurement: Measurement, config_dict: dict) -> Tuple[str, str]:
    """
    Error 1: Raw measurement error (wrong format, etc.). You can get it only in the log, not in the prediction.
    Error 2: Missing keys error. You can see specific missing keys in the log.
    Error 3: Frequency error. There is a too-large or too-short time gap in the data.
    Error 4: Synchronization error. The 8 measurements can not be synchronized.
    Error 5: Length error. The measurement is not long enough (expected 90 min).
    """
    # check if measurement in not None
    if measurement is None:
        return "raw_measurement_NOK", "Error 1"

    # check if measurement has data for each key
    missing_keys = measurement.get_missing_keys()
    if len(missing_keys) != 0:
        return "missing keys: {}".format(missing_keys), "Error 2"

    # interpolate larger time gaps
    interpolate_measurements(measurement, min_diff=config_dict["frequency_check_eps_error"] * 2 - 10,
                             max_diff=config_dict["interpolation_max_diff"])

    # check if frequency is okay for each key
    # warning
    expected_delta_ms = (1 / config_dict["frequency"]) * 1000  # ms
    eps = config_dict["frequency_check_eps_warning"]  # ms
    measurement.check_frequency(expected_delta_ms, eps=eps)

    # error
    expected_delta_ms = (1 / config_dict["frequency"]) * 1000  # ms
    eps = config_dict["frequency_check_eps_error"]  # ms
    if not measurement.check_frequency(expected_delta_ms, eps=eps):
        return "frequency_NOK", "Error 3"

    # synchronize data along timestamp_ms
    synchron_ok = True
    try:
        measurement.synchronize_measurement_dict()
    except SynchronizationError:
        synchron_ok = False

    if not synchron_ok:
        return "synchron_NOK", "Error 4"

    # check if the measurement is long enough
    length_ms = min_to_millisec(config_dict["meas_length_min"])
    if not measurement.check_length(length_ms):
        return "length_NOK", "Error 5"

    return "OK", "NO Error"
