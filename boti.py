from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import matplotlib as mpl

# mpl.use('Qt5Agg')

from termcolor import colored
from measurement_utils.measurement import key_list_short, Measurement
from utils.general_utils import to_str_timestamp, to_int_timestamp

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def get_length_from_timestamps(start_ts: int, end_ts: int) -> str:
    start = datetime.fromtimestamp(start_ts / 1000)
    end = datetime.fromtimestamp(end_ts / 1000)
    length = end - start
    mm, ss = divmod(length.total_seconds(), 60)
    hh, mm = divmod(mm, 60)
    return "{}:{}".format(int(hh), int(mm))


path = "/home/ad.adasworks.com/levente.peto/projects/stroke_prediction/98727214-2B8F-471B-A4D4-4176DB276EF8_2024-03-04 15:30:30.004253+01:00.csv"
df = pd.read_csv(path)
min_ts = df["timestamp_ms"].min()
time_series = df["timestamp_ms"].apply(lambda x: get_length_from_timestamps(min_ts, x))
df["time"] = time_series.values
print(df.head())

for key in [('r', 'a', 'a'), ('r', 'a', 'g')]:
    meas_df = df[df["keys_tuple"] == str(key)]
    meas_df[:].plot(x="time", y=["x", "y", "z"], grid=True, title="{}-{}-{}".format(*key))

plt.show()

exit()

# ts_ms_list = df[df["keys_tuple"] == str(('r', 'a', 'g'))]["timestamp_ms"]
# ts_list = df[df["keys_tuple"] == str(('r', 'a', 'g'))]["timestamp"]
# print(np.diff(ts_ms_list)[np.diff(ts_ms_list) > 80])
# print(ts_list[:-1][np.diff(ts_ms_list) > 80])

# print(df[df["keys_tuple"] == str(("r", "a", "a"))].head(20))
# print(df[df["keys_tuple"] == str(("r", "a", "a"))].tail(20))
# exit()
time_of_requests = df["time_of_request"].unique()

for key in key_list_short:
    print(key)
    ts_list = df[df["keys_tuple"] == str(key)]["timestamp_ms"]
    plt.plot(range(len(ts_list)), ts_list)
    plt.title(key)
    ax = plt.gca()
    ax.get_yaxis().set_visible(False)
    plt.show()
    plt.plot(range(len(ts_list) - 1), np.diff(ts_list))
    plt.title(str(key) + " diff")
    plt.show()
    continue

    # if key not in [('left', 'leg', 'acc'), ('left', 'leg', 'gyr')]:
    #     continue

    prev_max = 0
    for time_of_requ in time_of_requests:
        print(colored(time_of_requ, "red"))
        print(sorted(df[df["time_of_request"] == time_of_requ]["keys_tuple"].unique()))
        filtered_df = df[(df["time_of_request"] == time_of_requ) & (df["keys_tuple"] == str(key))]
        min_ts = filtered_df["timestamp"].min()
        max_ts = filtered_df["timestamp"].max()

        min_ts_ms = filtered_df["timestamp_ms"].min()
        max_ts_ms = filtered_df["timestamp_ms"].max()

        diff_ts_ms = np.diff(filtered_df["timestamp_ms"])
        min_diff_ts = diff_ts_ms.min()
        max_diff_ts = diff_ts_ms.max()
        # print(max(filtered_df["timestamp_ms"]), prev_max)
        shift = filtered_df["timestamp_ms"].min() - prev_max
        prev_max = filtered_df["timestamp_ms"].max()

        print(colored(key, "blue"),
              "min: {} ({}), max: {} ({}), min diff: {}, max diff: {}, shift: {}".format(min_ts, min_ts_ms, max_ts,
                                                                                         max_ts_ms, min_diff_ts,
                                                                                         max_diff_ts, shift))
    print()

exit()
meas = Measurement("8")
meas.fill_from_df(df)
print(meas.check_frequency(40, eps=5))
print(meas.log_list)

filtered_acc_df = df[
    (df["time_of_request"] == "2023-10-30 18:13:13.654069+01:00") & (df["keys_tuple"] == str(('left', 'leg', 'acc')))]
filtered_gyr_df = df[
    (df["time_of_request"] == "2023-10-30 18:13:13.654069+01:00") & (df["keys_tuple"] == str(('left', 'leg', 'gyr')))]

print(filtered_acc_df.head())
print(filtered_gyr_df.head())

start, end = None, None

plt.plot(np.diff(filtered_acc_df["timestamp_ms"])[start:end])
plt.show()

plt.plot(filtered_acc_df[["x", "y", "z"]][start:end], marker='o', linestyle='dashed', markersize=1)
plt.show()

plt.plot(filtered_acc_df["x"][start:end], marker='o', linestyle='dashed', markersize=1)
plt.show()
