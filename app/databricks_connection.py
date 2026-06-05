from databricks import sql
from dotenv import load_dotenv
import pandas as pd
import os

load_dotenv()


def get_connection():
    return sql.connect(
        server_hostname=os.getenv("DATABRICKS_SERVER_HOSTNAME"),
        http_path=os.getenv("DATABRICKS_HTTP_PATH"),
        access_token=os.getenv("DATABRICKS_TOKEN")
    )


def run_query(query):
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)

            result = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]

            return pd.DataFrame(result, columns=columns)