import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import matplotlib as mpl

# mpl.use('Qt5Agg')

from termcolor import colored
from measurement_utils.measurement import key_list, Measurement

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

"""
frequency is not correct, with key: ('left', 'leg', 'acc') min: 35, max: 44, avg: 39.79
less timestamps: 2023-10-30T18:35:24.632Z (35)
more timestamps: 

frequency is not correct, with key: ('left', 'leg', 'gyr') min: 35, max: 44, avg: 39.79
less timestamps: 2023-10-30T18:35:24.632Z (35)
more timestamps: 
Error message: frequency_NOK, Error 3
"""

path = "/home/ad.adasworks.com/levente.peto/projects/stroke_prediction/D586D987-C74A-4D00-933B-485553D1E3CC_2023-10-31 15:57:08.153042+01:00.csv"
df = pd.read_csv(path)
time_of_requests = df["time_of_request"].unique()

for key in key_list:
    # if key not in [('left', 'leg', 'acc'), ('left', 'leg', 'gyr')]:
    #     continue

    prev_max = 0
    for time_of_requ in time_of_requests:
        print(colored(time_of_requ, "red"))
        filtered_df = df[(df["time_of_request"] == time_of_requ) & (df["keys_tuple"] == str(key))]
        min_ts = filtered_df["timestamp"].min()
        max_ts = filtered_df["timestamp"].max()

        min_ts_ms = filtered_df["timestamp_ms"].min()
        max_ts_ms = filtered_df["timestamp_ms"].max()

        diff_ts_ms = np.diff(filtered_df["timestamp_ms"])
        min_diff_ts = min(diff_ts_ms)
        max_diff_ts = max(diff_ts_ms)
        # print(max(filtered_df["timestamp_ms"]), prev_max)
        shift = filtered_df["timestamp_ms"].min() - prev_max
        prev_max = filtered_df["timestamp_ms"].max()

        print(colored(key, "blue"), "min: {} ({}), max: {} ({}), min diff: {}, max diff: {}, shift: {}".format(min_ts, min_ts_ms, max_ts, max_ts_ms, min_diff_ts, max_diff_ts, shift))
    print()

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
