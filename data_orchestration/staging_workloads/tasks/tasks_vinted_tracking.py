from prefect import task, flow
import pandas as pd
from tasks.pyVinted.vinted import Vinted
from sqlalchemy import create_engine
from .utils import *
import time
from datetime import datetime
from prefect.tasks import exponential_backoff
from prefect.states import Failed

@task(name="Polling 'user_ids' from samples table.")
def load_data_from_postgres(conn) -> pd.DataFrame:
    """
    """
    cursor = conn.cursor()
    query = 'SELECT * FROM samples'  
    cursor.execute(query)
    rows = cursor.fetchall()
    return pd.DataFrame(rows, 
                       columns = ["user_id"])


@task(name="Batch API calls.",
      description= "Executes API calls in batches. Fails if dataframe is empty (all calls return None).",
      retries=3, 
      retry_delay_seconds=exponential_backoff(backoff_factor=7),
      retry_jitter_factor=2,
      log_prints= True)
def fetch_sample_data(data) -> pd.DataFrame:
    """
    requests.exceptions.HTTPError 429: https://www.rfc-editor.org/rfc/rfc6585#page-3
    """
    # Specify your transformation logic here
    vinted = Vinted()
    _tracking_list = []
    for index, row in data.iterrows():
        try:
            _item = vinted.items.search_item(user_id = row["user_id"])
            _tracking_list.append(_item)
        except:
            pass
        
    if _tracking_list == []:
        #prefect.engine.signals.SKIP()
        return Failed(message="Dataframe is empty.")

    df = pd.concat(_tracking_list, 
                   axis=0, 
                   ignore_index=True)

    return df

@task(name="Drops and type asserts the columns fetched.")
def transform_data(df: pd.DataFrame, **kwargs) -> None:
    """
    """
    cols = ["id", "brand", "size", "catalog_id", "color1_id", "favourite_count", 
            "view_count", "created_at_ts", "original_price_numeric", "price_numeric", "description", "package_size_id", "service_fee", "city", "country", "color1", "status", "item_closing_action"]
    df = df[cols]

    df = df.rename(columns={'id': 'product_id', 
                            "brand": "brand_title", 
                            "size": "size_title", 
                            "created_at_ts": "created_at"})
    
    df["original_price_numeric"] = df["original_price_numeric"].astype(float)
    df["price_numeric"] = df["price_numeric"].astype(float)
    df["date"] = datetime.now().strftime("%Y-%m-%d")
    return df

@task(name= "Export data to 'tracking'.",
      description= "Export tracking data to staging table: 'tracking'",
      timeout_seconds = 360,
      retries= 2)
def export_data_to_postgres(df: pd.DataFrame, **kwargs) -> None:
    """
    """

    table_name = 'tracking'  # Specify the name of the table to export data to
    engine = create_engine('postgresql://user:4202@localhost:5432/vinted-ai')
    df.to_sql(table_name, 
              engine, 
              if_exists = "append", 
              index = False, 
              method= insert_on_conflict_nothing_tracking,
              schema= "public")


def load_balancer(df: pd.DataFrame, chunk_size = 10, interval = 600) -> None:
    # total bandwidth = 50*1*24 = 1200
    for start in range(0, df.shape[0], chunk_size):
        tracking_subflow(df = df.iloc[start:start + chunk_size], 
                         name = f"Tracking subflow for: {str(start-chunk_size)}-{str(start)}")
        time.sleep(interval)

@flow(flow_run_name= "{name}", 
      log_prints= True)
def tracking_subflow(df, name):
    df = fetch_sample_data(df)
    df = transform_data(df)
    print(df)
    export_data_to_postgres(df)
