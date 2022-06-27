from ..measurement.measurement_collector import MeasurementCollector


def start_checking(_param_dict):
    _db_path = _param_dict["db_path"]
    _m_path = _param_dict["m_path"]
    _base_path = _param_dict["base_path"]
    _ucanaccess_path = _param_dict["ucanaccess_path"]
    mc = MeasurementCollector(_base_path, _db_path, _m_path, _ucanaccess_path, check=True)


if __name__ == "__main__":
    param_dict = {
        "base_path": "/home/levcsi/projects/stroke_prediction/data",
        "db_path": "/home/levcsi/projects/stroke_prediction/data/WUS-v4m.accdb",
        "m_path": "/home/levcsi/projects/stroke_prediction/data/biocal.xlsx",
        "ucanaccess_path": "/home/levcsi/projects/stroke_prediction/ucanaccess",
    }

    start_checking(param_dict)

