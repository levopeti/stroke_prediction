import os
import jaydebeapi
import pandas as pd
import warnings

warnings.simplefilter(action='ignore', category=Warning)

class MeasureDB(object):
    def __init__(self, accdb_path: str, ucanaccess_path: str) -> None:
        self.accdb_path = accdb_path
        self.ucanaccess_path = ucanaccess_path

        self.dict_of_measureDB_df = self.get_measure_df()

    def get_class_value_dict(self, meas_id: int) -> dict:
        neurology_df = self.dict_of_measureDB_df["Z_3NEUROLÓGIA"]
        neurology_df = neurology_df[neurology_df["VizsgAz"] == meas_id]
        class_value_dict = {
            ("left", "arm"): int(eval(neurology_df["ParStatBK"].values[0]) * 5),
            ("left", "leg"): int(eval(neurology_df["ParStatBL"].values[0]) * 5),
            ("right", "arm"): int(eval(neurology_df["ParStatJK"].values[0]) * 5),
            ("right", "leg"): int(eval(neurology_df["ParStatJL"].values[0]) * 5),
        }
        return class_value_dict

    def get_measure_df(self, write=False) -> dict:
        "https://stackoverflow.com/questions/70716540/how-do-i-use-jaydebeapi-to-read-a-access-db-file-on-databricks"
        ucanaccess_jars = [
            os.path.join(self.ucanaccess_path, "ucanaccess-5.0.1.jar"),
            os.path.join(self.ucanaccess_path, "lib/commons-lang3-3.8.1.jar"),
            os.path.join(self.ucanaccess_path, "lib/commons-logging-1.2.jar"),
            os.path.join(self.ucanaccess_path, "lib/hsqldb-2.5.0.jar"),
            os.path.join(self.ucanaccess_path, "lib/jackcess-3.0.1.jar"),
        ]
        classpath = ":".join(ucanaccess_jars)
        cnxn = jaydebeapi.connect(
            "net.ucanaccess.jdbc.UcanaccessDriver",
            f"jdbc:ucanaccess://{self.accdb_path}",
            ["", ""],
            classpath,
        )

        table_names = pd.read_sql_query("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA='PUBLIC'", cnxn)

        if write:
            with pd.ExcelWriter("./data/accdb.xlsx") as writer:
                for table_name in table_names["TABLE_NAME"]:
                    df = pd.read_sql_query("SELECT * FROM {}".format(table_name), cnxn)
                    df.to_excel(writer, index=False, sheet_name=table_name)

        dict_of_df = dict()
        for table_name in table_names["TABLE_NAME"]:
            dict_of_df[table_name] = pd.read_sql_query("SELECT * FROM {}".format(table_name), cnxn)

        return dict_of_df

