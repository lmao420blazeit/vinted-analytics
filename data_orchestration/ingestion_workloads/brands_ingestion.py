from prefect import task
import pandas as pd
from sqlalchemy import create_engine
from ..utils import upsert_brands_dim

@task(name="Load data for backfilling")
def load_from_brands_interest(engine):
    try:
        # try delta load
        latest_record = "SELECT MAX(date) FROM brands_fact"
        date = pd.read_sql(latest_record, engine)["date"]
        sql_query = f"SELECT * FROM brands_interest WHERE date > '{date}'"
        df = pd.read_sql(sql_query, engine)
        
    except:
        # full load
        sql_query = "SELECT * FROM brands_interest"
        df = pd.read_sql(sql_query, engine)

    return df

@task(name = "Transform to brands_dim")
def brands_dim_transform(data):
    # ensure only the latest date is inserted
    data = data[data['date'] == data["date"].max()]
    data = data[["brand_id", "title", "is_visible_in_listings", "is_luxury", "is_hvf"]]
    return (data)

@task(name="Export brands to brands_dim")
def brands_interest_to_dim(data: pd.DataFrame, engine) -> None:
    """
    Exports metadata to a PostgreSQL table.

    Args:
        data (pd.DataFrame): Input DataFrame containing metadata.
        engine: The SQLAlchemy engine for the PostgreSQL database.

    Returns:
        None
    """
    #schema_name = 'public'  # Specify the name of the schema to export data to
    table_name = 'brands_dim'  # Specify the name of the table to export data to
    data.to_sql(table_name, 
                engine, 
                if_exists = "append", 
                index = False,
                method=upsert_brands_dim
                )
    
    return

@task(name="Export brands to brands_fact")
def brands_interest_to_fact(data: pd.DataFrame, engine) -> None:
    """
    Exports metadata to a PostgreSQL table.

    Args:
        data (pd.DataFrame): Input DataFrame containing metadata.
        engine: The SQLAlchemy engine for the PostgreSQL database.

    Returns:
        None
    """
    #schema_name = 'public'  # Specify the name of the schema to export data to
    data = data[["brand_id", "date", "favourite_count", "item_count"]]
    table_name = 'brands_fact'  # Specify the name of the table to export data to
    data.to_sql(table_name, 
                engine, 
                if_exists = "replace", 
                index = False
                )
    
    return