import traceback
from time import sleep

from utils.api_utils import get_configuration
from utils.arg_parser_and_config import get_config_dict
from utils.discord import DiscordBot
from utils.log_maker import start_log_maker, set_log_dir_path, write_log
from main_loop.main_loop import run_main_loop
from main_loop.main_loop_local import run_main_loop_local
from ai_utils.six_models import SixModels


if __name__ == "__main__":
    # TODO: time measurement
    _config_dict = get_config_dict()
    _configuration = get_configuration(_config_dict)

    # TODO
    _model = SixModels(_config_dict)

    _discord = DiscordBot(active=_config_dict["discord"])

    start_log_maker(_config_dict, _discord)
    set_log_dir_path(_config_dict["log_dir_path"])

    if _config_dict["local_mode"]:
        current_main_loop = run_main_loop_local
    else:
        current_main_loop = run_main_loop

    while True:
        try:
            write_log("main_loop.txt", "Stroke ai has started. New session has started (in an infinity loop)",
                      title="UploadInfo", print_out=True, color="blue", add_date=True, write_discord=True)
            current_main_loop(_model, configuration=_configuration, config_dict=_config_dict)
            sleep(10)
        except Exception:
            print(traceback.format_exc())
            write_log("main_loop.txt", "Stroke ai has stopped. Error: {}".format(traceback.format_exc()),
                      title="UploadInfo", print_out=True, color="blue", add_date=True, write_discord=True)
        except KeyboardInterrupt:
            write_log("main_loop.txt", "Stroke ai has stopped. Stopped by keyboard interrupt (infinity loop ends)",
                      title="UploadInfo", print_out=True, color="blue", add_date=True, write_discord=True)
            break
