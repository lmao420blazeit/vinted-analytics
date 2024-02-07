from prefect import task, flow
import pandas as pd
from pyVinted.vinted import Vinted
from sqlalchemy import create_engine
from ..utils import *
import requests
import time
#from prefect.client.schemas.schedules import IntervalSchedule
from datetime import timedelta, datetime

@task(name="load_data_from_postgres")
def load_data_from_postgres(conn) -> pd.DataFrame:
    """
    """

    cursor = conn.cursor()
    query = 'SELECT * FROM samples'  
    cursor.execute(query)
    rows = cursor.fetchall()
    return pd.DataFrame(rows, 
                       columns = ["user_id"])


@task(name="fetch_sample_data_with_backoff", log_prints= True)
def fetch_sample_data(data) -> pd.DataFrame:
    """
    requests.exceptions.HTTPError 429: https://www.rfc-editor.org/rfc/rfc6585#page-3
    """
    # Specify your transformation logic here
    vinted = Vinted()
    _tracking_list = []
    retry = 1 # retry doesnt reset   
    for index, row in data.iterrows():
        _item = vinted.items.search_item(user_id = row["user_id"])
        _tracking_list.append(_item)
        print(_item)

    df = pd.concat(_tracking_list, 
                   axis=0, 
                   ignore_index=True)

    return df

@task(name="transform_data")
def transform_data(df: pd.DataFrame, **kwargs) -> None:
    """
    """
    df = df.rename(columns={'id': 'product_id', 
                            "brand": "brand_title", 
                            "size": "size_title", 
                            "created_at_ts": "created_at"})
    
    df["original_price_numeric"] = df["original_price_numeric"].astype(float)
    df["price_numeric"] = df["price_numeric"].astype(float)
    df["date"] = datetime.now().strftime("%Y-%m-%d")
    return df

@task(name= "export_data_to_postgres")
def export_data_to_postgres(df: pd.DataFrame, **kwargs) -> None:
    """
    """

    table_name = 'tracking'  # Specify the name of the table to export data to
    engine = create_engine('postgresql://user:4202@localhost:5432/vinted-ai')
    df.to_sql(table_name, engine, if_exists = "append", index = False, method= insert_on_conflict_nothing_tracking)


@task(name= "get_missing_values")
def get_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    """
    df = df[df.isna().any(axis=1)]

    return (df)


@task(name= "remove_sold_from_sample")
def remove_sold_from_sample(df: pd.DataFrame, **kwargs) -> None:
    """
    """
    engine = create_engine('postgresql://user:4202@localhost:5432/vinted-ai')
    if df.empty:
        return
    
    ids = tuple(df["product_id"])
    print(ids)
    query = f"DELETE FROM samples WHERE product_id IN {ids}"
    engine.execute(query)
    return

def load_balancer(df: pd.DataFrame, chunk_size = 50, interval = 600) -> None:
    # total bandwidth = 50*1*24 = 1200
    for start in range(0, df.shape[0], chunk_size):
        #start_time = datetime.utcnow() + timedelta(seconds=interval*i)
        #child_flow = tracking_subflow(df.iloc[start:start + chunk_size])
        df = fetch_sample_data(df.iloc[start:start + chunk_size])
        df = transform_data(df)
        export_data_to_postgres(df)
        time.sleep(interval)

@flow(name = "Tracking load balancer subflows.")
def tracking_subflow(df):
    df = fetch_sample_data(df)
    df = transform_data(df)
    export_data_to_postgres(df)
    mvalues = get_missing_values(df)
    remove_sold_from_sample(mvalues)
