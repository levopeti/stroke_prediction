import pytz
from datetime import timedelta, datetime

from utils.api_utils import get_measurement_ids, get_configuration, get_predictions_from_time_point
from utils.general_utils import min_to_millisec, to_str_timestamp

_config_dict = {"host_url_and_token_path": "./host_url_and_token.json"}
configuration = get_configuration(_config_dict)

timezone = pytz.timezone("Europe/Budapest")
now_ts = datetime.now(timezone)

_from = to_str_timestamp(now_ts - timedelta(minutes=90))  # to_str_timestamp(now_ts - timedelta(minutes=90))  "2023-10-05T13:29:39.362Z"
_interval = min_to_millisec(90)

measurement_ids = get_measurement_ids(configuration,
                                      _from=_from,
                                      _interval=_interval)

print(measurement_ids)

# predictions = get_predictions_from_time_point(configuration,
#                                               _from=_from,
#                                               _interval=7000000)

# print(predictions)

for meas_id in measurement_ids:
    print(meas_id)
    predictions = get_predictions_from_time_point(configuration,
                                                  _from=_from,
                                                  _interval=_interval,
                                                  _measurement_id=meas_id
                                                  )

    print(predictions)
