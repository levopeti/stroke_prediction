import jaydebeapi
import pandas as pd
import os

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def get_measure_df(ucanaccess_path, db_path, write=False):
    "https://stackoverflow.com/questions/70716540/how-do-i-use-jaydebeapi-to-read-a-access-db-file-on-databricks"
    ucanaccess_jars = [
        os.path.join(ucanaccess_path, "ucanaccess-5.0.1.jar"),
        os.path.join(ucanaccess_path, "lib/commons-lang3-3.8.1.jar"),
        os.path.join(ucanaccess_path, "lib/commons-logging-1.2.jar"),
        os.path.join(ucanaccess_path, "lib/hsqldb-2.5.0.jar"),
        os.path.join(ucanaccess_path, "lib/jackcess-3.0.1.jar"),
    ]
    classpath = ":".join(ucanaccess_jars)
    cnxn = jaydebeapi.connect(
        "net.ucanaccess.jdbc.UcanaccessDriver",
        f"jdbc:ucanaccess://{db_path}",
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


if __name__ == "__main__":
    _db_path = "/home/levcsi/projects/stroke_prediction/data/WUS-v4meresek 20220202.accdb"
    _dict_of_df = get_measure_df(_db_path, write=True)




