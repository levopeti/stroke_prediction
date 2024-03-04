import pandas as pd

from ai_utils.six_models import SixModels
from measurement_utils.measurement import Measurement

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

config_dict = {
        "model_folder": "./trained_models",
        "host_url_and_token_path": "./host_url_and_token.json",
        "log_dir_path": "./log",
        "init_data": "./init_data/init_data.csv",
        "frequency": 25,  # Hz, T = 40 ms
        "frequency_check_eps_warning": 3,  # ms
        "frequency_check_eps_error": 40,  # ms
        "batch_size": 100,
        "step_size_sec": 20,
        "left_arm_only": True,
        "length_of_init_data_min": 90,
        "init_time_diff_threshold": 1000,
        "start_date": None,  # "2024-02-028T13:29:39.362Z", None
    }

path = "./98727214-2B8F-471B-A4D4-4176DB276EF8_2024-03-04 15:30:30.004253+01:00.csv"
df = pd.read_csv(path)
df["keys_tuple"] = df["keys_tuple"].apply(lambda x: eval(x))

print(df.head())
print(df.tail())
print(df.keys_tuple.unique())

meas = Measurement("00001")
meas.fill_from_df(df)

model = SixModels(config_dict, to_cuda=False)
prediction_dict = model.compute_prediction(meas, debug_print=True)

print(prediction_dict)



