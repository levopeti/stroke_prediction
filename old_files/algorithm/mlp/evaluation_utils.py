import os
import numpy as np
import pandas as pd
import seaborn as sns

from tensorflow import keras
from datetime import datetime
from sklearn.metrics import confusion_matrix

from ..measurement.measurement_collector import MeasurementCollector
from ..utils.cache_utils import cache

import matplotlib.pyplot as plt
import matplotlib.dates as md
import matplotlib

matplotlib.rcParams['figure.figsize'] = [15, 15]

# seconds between two measurements
TIME_DELTA_SEC = 0.04


@cache
def generate_infer_data(_mc, length, step_size, limb, type_of_set, use_cache, key):
    keys_in_order = (("arm", "acc"),
                     ("leg", "acc"),
                     ("arm", "gyr"),
                     ("leg", "gyr"))

    cur_meas_name = None
    result_dict = dict()

    for mean_dict, class_value, meas_name in _mc.sweep_mean_with_class_generator(mean_type='all',
                                                                                 limb=limb,
                                                                                 length=length,
                                                                                 step_size=step_size,
                                                                                 type_of_set=type_of_set):
        if cur_meas_name != meas_name:
            print(meas_name)
            result_dict[meas_name] = {
                "X": list(),
                "y": list()
            }
            cur_meas_name = meas_name

        instance = list()
        for key in keys_in_order:
            if limb != "all" and key[0] != limb:
                continue
            instance.append(mean_dict[key])
        instance = sum(instance, [])

        result_dict[meas_name]["X"].append(instance)
        result_dict[meas_name]["y"].append(class_value)

    return result_dict


@cache
def make_prediction(_model, _data_dict, use_cache, key):
    result_dict = dict()
    cur_meas_name = None

    for meas_name, x_y_dict in _data_dict.items():
        if cur_meas_name != meas_name:
            print(meas_name)
            result_dict[meas_name] = {
                "class_values": x_y_dict["y"],
                "y_pred_list": list()
            }
            cur_meas_name = meas_name

        y_pred_list = _model.predict(x_y_dict["X"])
        result_dict[meas_name]["y_pred_list"] = y_pred_list

    return result_dict


def make_plot(result_dict, minutes, step_size, save_path=None, type_of_set="train", plot=False):
    print("make plot")
    plt.ion()

    pred_is_healthy_list = list()
    is_healty_list = list()
    for meas_name, pred_dict in result_dict.items():
        print(meas_name)
        print("class value: {}".format(set(pred_dict["class_values"])))
        print("len of y pred list: {}".format(len(pred_dict["y_pred_list"])))

        pred_array = np.array(pred_dict["y_pred_list"]).argmax(axis=1)
        percentage_list = [len(pred_array[pred_array == value]) / len(pred_array) * 100 for value in range(6)]
        fig, axs = plt.subplots(4, 1, facecolor="w")

        color_list = ['blue'] * 6
        for i in set(pred_dict["class_values"]):
            color_list[i] = "red"

        graph = axs[0].bar(list(range(6)),
                           percentage_list,
                           color=color_list)

        for p in graph:
            height = p.get_height()
            axs[0].text(x=p.get_x() + p.get_width() / 2, y=height + 1,
                        s="{:.3} %".format(height),
                        ha='center')

        axs[0].set_ylim(-5, 105)
        axs[0].legend(["Ratio of detections during the measurment"], loc='best')
        axs[0].grid(True)

        pred_is_healthy = np.array(pred_dict["y_pred_list"]).argmax(axis=1) > 4.5
        is_healty = np.array(pred_dict["class_values"]) > 4.5
        ratio = np.sum(pred_is_healthy == is_healty) / len(pred_is_healthy)

        pred_is_healthy_list.append(pred_is_healthy)
        is_healty_list.append(np.ones_like(pred_is_healthy) * is_healty)

        axs[1].pie([ratio, 1 - ratio], explode=(0, 0.1), labels=["True", "False"], autopct='%1.1f%%',
                   shadow=True, startangle=90)
        axs[1].legend(["Prediction in terms of is it healty or not"], loc='best')

        axs[2].plot(np.array(pred_dict["y_pred_list"]).argmax(axis=1), label=meas_name)
        axs[2].plot(pred_dict["class_values"], label="class_values ({})".format(set(pred_dict["class_values"])))
        axs[2].axis([None, None, -0.5, 5.5])
        axs[2].legend(loc='best')
        axs[2].grid()

        xformatter = md.DateFormatter('%H:%M')
        # xlocator = md.MinuteLocator(interval=8)
        xlocator = md.HourLocator(interval=100)
        axs[2].xaxis.set_major_formatter(xformatter)
        axs[2].xaxis.set_major_locator(xlocator)

        axs[3].plot(np.array(pred_dict["y_pred_list"]).max(axis=1) * 100, label="confidence of prediction")
        axs[3].plot([a[i] * 100 for a, i in zip(pred_dict["y_pred_list"], pred_dict["class_values"])],
                    label="confidence of prediction for true label")
        axs[3].axis([None, None, -5, 105])
        axs[3].legend(loc='best')
        axs[3].grid()

        if save_path is not None:
            os.makedirs(
                os.path.join(save_path, "plots/plots_{}m_{}step_{}/{}/".format(minutes, step_size,
                                                                                datetime.now().strftime('%Y-%m-%d-%H'),
                                                                                type_of_set)), exist_ok=True)
            plt.savefig(os.path.join(save_path,
                                     "plots/plots_{}m_{}step_{}/{}/{}.png".format(minutes, step_size,
                                                                                   datetime.now().strftime(
                                                                                       '%Y-%m-%d-%H'),
                                                                                   type_of_set, meas_name)))

        if plot:
            plt.show()

    tn_fp_fn_tp = confusion_matrix(~np.concatenate(is_healty_list), ~np.concatenate(pred_is_healthy_list)).ravel()

    if len(tn_fp_fn_tp) == 4:
        print("tn_fp_fn_tp ", tn_fp_fn_tp)
        tn, fp, fn, tp = tn_fp_fn_tp
        # Sensitivity = TP / (TP + FN)
        sensitivity = tp / (tp + fn)
        print(sensitivity)
        # Specificity = TN / (TN + FP)
        specificity = tn / (tn + fp)
        print(specificity)

        color_list = ['blue', 'red']

        fig = plt.figure(facecolor="w")
        ax = fig.add_axes([0, 0, 1, 1])
        langs = ['sensitivity', 'specificity']
        students = [sensitivity * 100, specificity * 100]
        graph = ax.bar(langs, students, color=color_list)

        for p in graph:
            height = p.get_height()
            ax.text(x=p.get_x() + p.get_width() / 2, y=height + 1,
                    s="{:.2f} %".format(height),
                    ha='center')

        ax.legend(["sensitivity - specificity"], loc='best')
        ax.grid(True)

        if save_path is not None:
            os.makedirs(os.path.join(save_path, "plots/plots_{}m_{}step_{}/{}/".format(minutes, step_size,
                                                                                        datetime.now().strftime(
                                                                                            '%Y-%m-%d-%H'),
                                                                                        type_of_set)), exist_ok=True)
            plt.savefig(os.path.join(save_path, "plots/plots_{}m_{}step_{}/{}/sens_spec.png".format(minutes, step_size,
                                                                                                     datetime.now().strftime(
                                                                                                         '%Y-%m-%d-%H'),
                                                                                                     type_of_set)))
    if plot:
        plt.show()


def sens_spec(result_dict: dict, step_size: int, save_path: str, minutes: int, type_of_set: str):
    new_time_delta_sec = step_size * TIME_DELTA_SEC
    sens_spec_dict = {"threshold": list(),
                      "window (s)": list(),
                      "sensitivity": list(),
                      "specificity": list()
                      }

    for avg_prob_threshold in [0.7, 0.8, 0.85, 0.9, 0.95]:
        for window_length_sec in [20, 60, 120, 300, 600]:
            window_length = int(window_length_sec / new_time_delta_sec)

            pred_is_stroke_list = list()
            is_stroke_list = list()
            for meas_name, pred_dict in result_dict.items():
                pred_is_stroke = np.array(pred_dict["y_pred_list"]).argmax(axis=1) < 4.5
                is_stroke = np.array(pred_dict["class_values"]) < 4.5

                avg_pred = np.lib.stride_tricks.sliding_window_view(pred_is_stroke, window_length).mean(axis=1)
                avg_pred_is_stroke = avg_pred > avg_prob_threshold

                pred_is_stroke_list.append(avg_pred_is_stroke)
                is_stroke_list.append(is_stroke[-len(avg_pred_is_stroke):])

            tn_fp_fn_tp = confusion_matrix(np.concatenate(is_stroke_list), np.concatenate(pred_is_stroke_list)).ravel()
            if len(tn_fp_fn_tp) == 4:
                tn, fp, fn, tp = tn_fp_fn_tp
                # Sensitivity = TP / (TP + FN)
                sensitivity = tp / (tp + fn)
                # Specificity = TN / (TN + FP)
                specificity = tn / (tn + fp)
            else:
                sensitivity = np.nan
                specificity = 1

            sens_spec_dict["threshold"].append(avg_prob_threshold)
            sens_spec_dict["window (s)"].append(window_length_sec)
            sens_spec_dict["sensitivity"].append(sensitivity)
            sens_spec_dict["specificity"].append(specificity)

    sens_spec_df = pd.DataFrame.from_dict(sens_spec_dict)
    sens_df = pd.pivot_table(sens_spec_df, values="sensitivity", index=["threshold"], columns=["window (s)"])
    spec_df = pd.pivot_table(sens_spec_df, values="specificity", index=["threshold"], columns=["window (s)"])
    print(sens_df)
    print(spec_df)

    if len(sens_df) > 0 and len(spec_df) > 0:
        fig, axs = plt.subplots(2, 1, facecolor="w")
    else:
        fig, axs = plt.subplots(1, 1, facecolor="w")
        axs = [axs]

    axs_id = 0

    if len(sens_df) > 0:
        sns.heatmap(sens_df, vmax=1, cmap="coolwarm", linewidths=0.30, annot=True, ax=axs[axs_id])
        axs[axs_id].title.set_text("Sensitivity")
        axs_id += 1

    if len(spec_df) > 0:
        sns.heatmap(spec_df, vmax=1, cmap="coolwarm", linewidths=0.30, annot=True, ax=axs[axs_id])
        axs[axs_id].title.set_text("Specificity")

    if save_path is not None:
        plt.savefig(os.path.join(save_path, "plots/plots_{}m_{}step_{}/{}/sens_spec_hm.png".format(minutes, step_size,
                                                                                                datetime.now().strftime(
                                                                                                    '%Y-%m-%d-%H'),
                                                                                                type_of_set)))


def write_prediction_to_csv(_prediction_dict, length, step_size, save_path):
    dict_to_df = dict()
    max_length = 0

    for k, v in _prediction_dict.items():
        probs = np.array([x for x in v["y_pred_list"] if x is not None])
        predicted_classes = probs.argmax(axis=1)

        dict_to_df[str(k) + "(" + str(v["class_values"][0]) + ")"] = predicted_classes.tolist()
        max_length = max(max_length, len(predicted_classes))

    for l in dict_to_df.values():
        l += (max_length - len(l)) * [None]

    result_df = pd.DataFrame.from_dict(dict_to_df)

    path_to_save = os.path.join(save_path, "result_class_{}_{}.csv".format(length, step_size))
    result_df.to_csv(path_to_save, index=False)
    print("prediction saved into csv ({})".format(path_to_save))


def load_model(_model_path):
    _model = keras.models.load_model(_model_path)
    _model.summary()

    return _model


def start_evaluation(_param_dict):
    _db_path = _param_dict["db_path"]
    _m_path = _param_dict["m_path"]
    _base_path = _param_dict["base_path"]
    _ucanaccess_path = _param_dict["ucanaccess_path"]
    _class_values_to_use = _param_dict["class_values_to_use"]
    mc = MeasurementCollector(_base_path, _db_path, _m_path, _ucanaccess_path, class_values_to_use=_class_values_to_use)

    length = int(TIME_DELTA_SEC * 60 * _param_dict["minutes"])
    step_size = _param_dict["step_size"]
    limb = _param_dict["limb"]
    type_of_set = _param_dict["type_of_set"]
    save_path = _param_dict["save_path"]
    model_path = _param_dict["model_path"]

    _model = load_model(model_path)

    key = "{}".format([length, step_size, limb, type_of_set, len(mc.measurement_dict[type_of_set]),
                       _class_values_to_use])
    infer_data = generate_infer_data(mc, length, step_size, limb, type_of_set, use_cache=True, key=key)
    prediction_dict = make_prediction(_model, infer_data, use_cache=False, key=key)
    make_plot(prediction_dict, _param_dict["minutes"], step_size, save_path=save_path, type_of_set=type_of_set)
    sens_spec(prediction_dict, step_size, save_path, _param_dict["minutes"], type_of_set)
    write_prediction_to_csv(prediction_dict, length, step_size, save_path)


if __name__ == "__main__":
    param_dict = {
        "minutes": 90,
        "limb": "all",
        "class_values_to_use": None,  # [0, 1, 2]
        "step_size": 500,  # x * TIME_DELTA_SEC in sec
        "type_of_set": "train",  # train, test, mixed
        "base_path": '/home/levcsi/projects/stroke_prediction/old_files/data',
        "db_path": "/home/levcsi/projects/stroke_prediction/old_files/data/WUS-v4m.accdb",
        "m_path": "/home/levcsi/projects/stroke_prediction/old_files/data/biocal.xlsx",
        "ucanaccess_path": "/home/levcsi/projects/stroke_prediction/ucanaccess",
        "save_path": ".",
        "model_path": "/home/levcsi/projects/stroke_prediction/models/model_90_1000000_all",
    }

    start_evaluation(param_dict)
