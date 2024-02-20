from prefect import task
import pandas as pd
from sqlalchemy import create_engine
from utils import upsert_color_dim, upsert_tracking_fact

@task(name="Load tracking_staging for backfilling")
def load_from_tracking_staging(engine):
    try:
        # try delta load
        latest_record = "SELECT MAX(date) FROM tracking_fact"
        date = pd.read_sql(latest_record, engine)["date"]
        sql_query = f"SELECT * FROM tracking_staging WHERE date > '{date}'"
        df = pd.read_sql(sql_query, engine)
        
    except:
        # full load
        sql_query = "SELECT * FROM tracking_staging"
        df = pd.read_sql(sql_query, engine)

    return df

@task(name = "Transform to 'color_dim'")
def color_dim_transform(data):
    # ensure only the latest date is inserted
    data = data[data['date'] == data["date"].max()]
    data = data[["color1_id", "color1"]]
    data.columns = ["color_id", "color_title"]

    # drop rows where color_id = empty
    data = data[~data["color_id"].isin(["", " "])]
    data["color_id"] = data["color_id"].fillna(-100)
    data["color_id"] = data["color_id"].astype(float).astype(int)
    return (data)

@task(name = "Transform to 'tracking_fact'")
def tracking_fact_transform(data):
    # ensure only the latest date is inserted
    #data = data[data['date'] == data["date"].max()]
    data = data[["product_id", "catalog_id", "brand_title", "date", "size_title", "color1_id", 
                 "favourite_count", "view_count", "created_at", "original_price_numeric",
                 "price_numeric", "package_size_id", "service_fee", "user_id", "status", "description"]]
    
    data = data.rename(columns={'color1_id': 'color_id'})
    data = data.dropna(subset = ["product_id", "date"])
    data["color_id"] = data["color_id"].fillna(-100)
    data["color_id"] = data["color_id"].astype(float).astype(int)
    return (data)

@task(name="Export color to 'color_dim'")
def export_color_dim(data: pd.DataFrame, engine) -> None:
    """
    Exports metadata to a PostgreSQL table.

    Args:
        data (pd.DataFrame): Input DataFrame containing metadata.
        engine: The SQLAlchemy engine for the PostgreSQL database.

    Returns:
        None
    """
    #schema_name = 'public'  # Specify the name of the schema to export data to
    table_name = 'color_dim'  # Specify the name of the table to export data to
    data.to_sql(table_name, 
                engine, 
                if_exists = "append", 
                index = False,
                method=upsert_color_dim
                )
    
    return

@task(name="Export color to 'tracking_fact'")
def export_tracking_fact(data: pd.DataFrame, engine) -> None:
    """
    Exports metadata to a PostgreSQL table.

    Args:
        data (pd.DataFrame): Input DataFrame containing metadata.
        engine: The SQLAlchemy engine for the PostgreSQL database.

    Returns:
        None
    """
    #schema_name = 'public'  # Specify the name of the schema to export data to
    table_name = 'tracking_fact'  # Specify the name of the table to export data to
    data.to_sql(table_name, 
                engine, 
                if_exists = "append", 
                index = False,
                method=upsert_tracking_fact
                )
    
    return