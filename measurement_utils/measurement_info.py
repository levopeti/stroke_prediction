import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from matplotlib.collections import PolyCollection
from typing import List
from measurement_utils.measurement import Measurement


class MeasurementInfoManager(object):
    """
    drop_ts
    init_delta_too_large_ts (list)
    init_ts_min / init_ts_max
    added_df_ts_min / added_df_ts_max
    current_ts_min / current_ts_miax

    inference_dict
    inference_timestamp_ms
    pred_is_stroke_dict
    inference_timestamp_ms
    non_inverted_array
    inverted_array
    """
    def __init__(self, config_dict: dict):
        self.config_dict = config_dict
        self.measurement_dict = dict()

    @staticmethod
    def collect_info(measurement: Measurement, keys: List[tuple]) -> None:
        for key in keys:
            # "timestamp_ms", "x", "y", "z"
            df = measurement.measurement_dict[key]

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
        def ts_to_dt(key: str) -> dt.datetime:
            return dt.datetime.fromtimestamp(self.measurement_dict[measurement_id][key] / 1000)

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

        fig, ax = plt.subplots()
        ax.add_collection(bars)

        if "drop_ts" in self.measurement_dict[measurement_id]:
            ax.vlines(ts_to_dt("drop_ts"), ymin=0, ymax=3, linestyles="dashed", label="drop_ts")

        ax.autoscale()
        loc = mdates.MinuteLocator(byminute=[0, 15, 30, 45])
        ax.xaxis.set_major_locator(loc)
        formatter = mdates.AutoDateFormatter(loc)
        formatter.scaled[1 / (24. * 60.)] = "%H:%M"
        ax.xaxis.set_major_formatter(formatter)

        ax.set_yticks([1, 2, 3])
        ax.set_yticklabels(["init", "added", "current"])
        ax.set_title(measurement_id)

        plt.show()



