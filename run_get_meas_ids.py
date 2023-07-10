import pytz
from datetime import timedelta, datetime

from api_utils import get_measurement_ids, get_configuration
from general_utils import min_to_millisec, to_str_timestamp

_config_dict = {"host_url_and_token_path": "./host_url_and_token.json"}
configuration = get_configuration(_config_dict)

timezone = pytz.timezone("Europe/Budapest")
now_ts = datetime.now(timezone)
measurement_ids = get_measurement_ids(configuration,
                                      _from=to_str_timestamp(
                                          now_ts - timedelta(minutes=90)),
                                      _interval=min_to_millisec(90))

print(measurement_ids)
