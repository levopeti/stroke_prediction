import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from matplotlib.collections import PolyCollection
from typing import List, Union
from measurement_utils.measurement import Measurement

stroke_threshold = 1.5  # [0, 1] -> stroke, [2] -> healthy


class MeasurementInfoManager(object):
    """
    drop_ts
    init_delta_too_large_ts (list)
    init_ts_min / init_ts_max
    added_df_ts_min / added_df_ts_max
    current_ts_min / current_ts_miax

    inference_dict: inference_dict["non-inverted"/"inverted", length_min (30, 60, 90)] = predictions [n, 3]
    inference_timestamp_ms [n,]
    pred_is_stroke_dict: pred_is_stroke_dict["non-inverted"/"inverted", length_min (30, 60, 90)] = pred_is_stroke [n,]
    non_inverted_array / inverted_array [n,], index_for_length = {30: 0, 60: 1, 90: 2}
    """
    def __init__(self, config_dict: dict):
        self.config_dict = config_dict
        self.measurement_dict = dict()

    # @staticmethod
    # def collect_info(measurement: Measurement, keys: List[tuple]) -> None:
    #     for key in keys:
    #         # "timestamp_ms", "x", "y", "z"
    #         df = measurement.measurement_dict[key]

    def add_new_measurement(self, measurement_id: str) -> None:
        assert measurement_id not in self.measurement_dict
        self.measurement_dict[measurement_id] = dict()

    def add_info(self, measurement_id: str, info_key: str, measurement_info) -> None:
        self.measurement_dict[measurement_id][info_key] = measurement_info

    def del_measurement(self, measurement_id: str) -> None:
        del self.measurement_dict[measurement_id]

    def plot_all(self) -> None:
        for measurement_id in self.measurement_dict.keys():
            self.plot_timeline(measurement_id)

    def plot_timeline(self, measurement_id: str) -> None:
        def ts_to_dt(key: Union[str, int]) -> dt.datetime:
            if isinstance(key, str):
                return dt.datetime.fromtimestamp(self.measurement_dict[measurement_id][key] / 1000)
            elif isinstance(key, int):
                return dt.datetime.fromtimestamp(key / 1000)

        # timeline boxes
        data = [
            (ts_to_dt("init_ts_min"), ts_to_dt("init_ts_max"), "init"),
            (ts_to_dt("added_df_ts_min"), ts_to_dt("added_df_ts_max"), "added"),
            (ts_to_dt("current_ts_min"), ts_to_dt("current_ts_max"), "current")
        ]

        cats = {"init": 1, "added": 2, "current": 3}
        color_mapping = {"init": "C0", "added": "C1", "current": "C2"}

        vertices = list()
        colors = list()
        for d in data:
            v = [(mdates.date2num(d[0]), cats[d[2]] - .4),
                 (mdates.date2num(d[0]), cats[d[2]] + .4),
                 (mdates.date2num(d[1]), cats[d[2]] + .4),
                 (mdates.date2num(d[1]), cats[d[2]] - .4),
                 (mdates.date2num(d[0]), cats[d[2]] - .4)]
            vertices.append(v)
            colors.append(color_mapping[d[2]])

        bars = PolyCollection(vertices, facecolors=colors)

        fig, axes = plt.subplots(7, 1, figsize=(10, 12))
        axes[0].add_collection(bars)

        # vertical lines
        if "drop_ts" in self.measurement_dict[measurement_id]:
            axes[0].vlines(ts_to_dt("drop_ts"), ymin=-0.05, ymax=3.55, linestyles="dashed", colors="k", label="drop_ts")

        for idtl_ts in self.measurement_dict[measurement_id]["init_delta_too_large_ts"]:
            axes[0].vlines(ts_to_dt(idtl_ts), ymin=-0.05, ymax=3.55, linestyles="dashed", colors="r", label="gap")

        axes[0].legend(loc="upper left")
        axes[0].autoscale()
        loc = mdates.MinuteLocator(byminute=[0, 15, 30, 45])
        axes[0].xaxis.set_major_locator(loc)
        formatter = mdates.AutoDateFormatter(loc)
        formatter.scaled[1 / (24. * 60.)] = "%H:%M"
        axes[0].xaxis.set_major_formatter(formatter)

        axes[0].set_yticks([1, 2, 3])
        axes[0].set_yticklabels(["init", "added", "current"])
        axes[0].set_title(measurement_id)

        # predictions
        ax_id = 1
        for (inverted, length_min), pred_is_stroke in self.measurement_dict[measurement_id]["pred_is_stroke_dict"].items():
            # raw predictions
            pred_timestamp_ms = self.measurement_dict[measurement_id]["inference_timestamp_ms"].tolist()
            pred_dt_list = [ts_to_dt(ts) for ts in pred_timestamp_ms]
            predictions = self.measurement_dict[measurement_id]["inference_dict"][(inverted, length_min)].argmax(axis=1)
            pred_is_stroke_ori = (predictions < stroke_threshold).astype(int)
            color_list = ["red" if x else "blue" for x in pred_is_stroke_ori.astype(bool)]
            axes[ax_id].scatter(pred_dt_list, pred_is_stroke_ori, c=color_list, s=10, marker="x")

            # averaged predictions
            timestamp_ms = pred_timestamp_ms[-len(pred_is_stroke):]
            dt_list = [ts_to_dt(ts) for ts in timestamp_ms]
            color_list = ["red" if x else "blue" for x in pred_is_stroke.astype(bool)]
            axes[ax_id].scatter(dt_list, pred_is_stroke, c=color_list, s=50)

            axes[ax_id].set_title("{}, {} min".format(inverted, length_min))
            axes[ax_id].axis([None, None, -0.25, 1.25])
            axes[ax_id].set_yticks([0, 1])
            axes[ax_id].set_yticklabels(["stroke" if x == 1 else "healthy" for x in [0, 1]])
            axes[ax_id].grid()
            axes[ax_id].xaxis.set_major_formatter(formatter)
            ax_id += 1

        plt.subplots_adjust(hspace=0.6)
        plt.savefig("./discord_plot.png")
        plt.close(fig)



