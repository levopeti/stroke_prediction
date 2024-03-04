from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

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


path = "./98727214-2B8F-471B-A4D4-4176DB276EF8_2024-03-04 15:30:30.004253+01:00.csv"
df = pd.read_csv(path)
min_ts = df["timestamp_ms"].min()
time_series = df["timestamp_ms"].apply(lambda x: get_length_from_timestamps(min_ts, x))
df["time"] = time_series.values
print(df.head())

for key in [('r', 'a', 'a'), ('r', 'a', 'g')]:
    meas_df = df[df["keys_tuple"] == str(key)]
    meas_df[:].plot(x="time", y=["x", "y", "z"], grid=True, title="{}-{}-{}".format(*key))

plt.show()

