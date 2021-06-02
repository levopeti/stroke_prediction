import jaydebeapi
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

db_path = "./data/WUS-v4gdprpoz.accdb"

ucanaccess_jars = [
    "/home/levcsi/ucanaccess/ucanaccess-5.0.1.jar",
    "/home/levcsi/ucanaccess/lib/commons-lang3-3.8.1.jar",
    "/home/levcsi/ucanaccess/lib/commons-logging-1.2.jar",
    "/home/levcsi/ucanaccess/lib/hsqldb-2.5.0.jar",
    "/home/levcsi/ucanaccess/lib/jackcess-3.0.1.jar",
]
classpath = ":".join(ucanaccess_jars)
cnxn = jaydebeapi.connect(
    "net.ucanaccess.jdbc.UcanaccessDriver",
    f"jdbc:ucanaccess://{db_path}",
    ["", ""],
    classpath,
)

table_names = pd.read_sql_query("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA='PUBLIC'", cnxn)

with pd.ExcelWriter("./data/accdb.xlsx") as writer:
    for table_name in table_names["TABLE_NAME"]:
        df = pd.read_sql_query("SELECT * FROM {}".format(table_name), cnxn)
        df.to_excel(writer, index=False, sheet_name=table_name)

