import argparse
import git
import sys
import subprocess
import yaml
import os


def show_versions():
    print("SYSTEM INFO")
    print("-----------")
    print("python:", sys.version)

    print("\nPYTHON DEPENDENCIES")
    print("-------------------")
    from pip._internal.operations import freeze
    for p in freeze.freeze():
        print(p)
    print("")


def get_git_hash() -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "git is not available"


def get_command() -> str:
    return " ".join(sys.argv)


def get_args() -> argparse.Namespace:
    repo = git.Repo(search_parent_directories=True)
    branch_name = "detached" if repo.head.is_detached else repo.active_branch.name

    if branch_name == "master":
        print(f"GIT HASH\n--------\n{get_git_hash()}\n")
        print(f"COMMAND\n-------\n{get_command()}\n")
        show_versions()

    parser = argparse.ArgumentParser(description='Run AI to predict strokes.')
    parser.add_argument('--meas_length_to_keep_min', default=100, type=int, help='Only keep this time range until now.')
    parser.add_argument('--interval_min', default=30, type=int, help='Time chunk length to get the data.')


    # parser.add_argument('--timedelta_from_now_h', default=0, type=float,
    #                     help='Hours from now in the past to get the from timestamp.')
    # parser.add_argument('--interval_milliseconds', default=300000, type=int, help='Time interval in milliseconds.')
    parser.add_argument('--meas_length_min', default=90, type=int, help='Considered time for prediction in minutes.')
    parser.add_argument('--inference_delta_sec', default=30, type=int,
                        help='Time delay between two measurements in seconds.')
    parser.add_argument('--test_mode', default=False, action='store_true',  help='Run in test mode.')
    parser.add_argument('--discord', default=False, action='store_true', help='Run discord webhook.')
    parser.add_argument('--local_mode', default=False, action='store_true', help='Local data flow through zmq.')

    args = parser.parse_args()
    return args


def get_other_config() -> dict:
    other_config_dict = {
        "model_path": "./models/model_90_1000000_all",
        "host_url_and_token_path": "./host_url_and_token.json",
        "frequency": 25,  # Hz, T = 40 ms
        "frequency_check_eps": 3
    }
    return other_config_dict


def get_config_dict() -> dict:
    config_dict = dict()

    args = get_args()
    config_dict.update(vars(args))
    config_dict.update(get_other_config())

    return config_dict


def save_config_dict(config_dict: dict, log_dir_path: str):
    os.makedirs(os.path.join(log_dir_path), exist_ok=True)

    with open(os.path.join(log_dir_path, 'config.yaml'), 'w') as outfile:
        yaml.dump(config_dict, outfile, default_flow_style=False)
