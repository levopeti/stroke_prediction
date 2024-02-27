import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import matplotlib as mpl

# mpl.use('Qt5Agg')

from termcolor import colored
from measurement_utils.measurement import key_list_short, Measurement

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

path = "/home/ad.adasworks.com/levente.peto/projects/stroke_prediction/FA7D8946-A5D9-4D8D-9819-697E8EA9B2C9_2024-02-26 14:16:42.803824+01:00.csv"
df = pd.read_csv(path)
print(df.head())
timestamp_ms = df["timestamp_ms"].values
too_large_diffs = np.diff(timestamp_ms) > 1000
limit_ts_ms = timestamp_ms[1:][too_large_diffs].max()
df = df[df["timestamp_ms"] >= limit_ts_ms]

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

        print(colored(key, "blue"), "min: {} ({}), max: {} ({}), min diff: {}, max diff: {}, shift: {}".format(min_ts, min_ts_ms, max_ts, max_ts_ms, min_diff_ts, max_diff_ts, shift))
    print()

exit()
meas = Measurement("8")
meas.fill_from_df(df)
print(meas.check_frequency(40, eps=5))
print(meas.log_list)

filtered_acc_df = df[(df["time_of_request"] == "2023-10-30 18:13:13.654069+01:00") & (df["keys_tuple"] == str(('left', 'leg', 'acc')))]
filtered_gyr_df = df[(df["time_of_request"] == "2023-10-30 18:13:13.654069+01:00") & (df["keys_tuple"] == str(('left', 'leg', 'gyr')))]

print(filtered_acc_df.head())
print(filtered_gyr_df.head())

start, end = None, None

plt.plot(np.diff(filtered_acc_df["timestamp_ms"])[start:end])
plt.show()

plt.plot(filtered_acc_df[["x", "y", "z"]][start:end], marker='o', linestyle='dashed', markersize=1)
plt.show()

plt.plot(filtered_acc_df["x"][start:end], marker='o', linestyle='dashed', markersize=1)
plt.show()
