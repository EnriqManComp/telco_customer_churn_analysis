from databricks import sql
import os
from dotenv import load_dotenv

load_dotenv()

with sql.connect(
    server_hostname=os.getenv("DATABRICKS_SERVER_HOSTNAME"),
    http_path=os.getenv("DATABRICKS_HTTP_PATH"),
    access_token=os.getenv("DATABRICKS_TOKEN")
) as connection:
    
    with connection.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM workspace.telco.ml_silver_data")
        row_count = cursor.fetchone()[0]

print("Row count:", row_count)
