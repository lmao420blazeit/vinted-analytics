from prefect import task
import pandas as pd
from sqlalchemy import create_engine
from utils import upsert_catalog_dim, upsert_catalog_fact

@task(name="Load catalog_staging for backfilling")
def load_from_catalog_staging(engine):
    try:
        # try delta load
        latest_record = "SELECT MAX(date) FROM catalog_fact"
        date = pd.read_sql(latest_record, engine)["date"]
        sql_query = f"SELECT * FROM catalog_staging WHERE date > '{date}'"
        df = pd.read_sql(sql_query, engine)
        
    except:
        # full load
        sql_query = "SELECT * FROM catalog_staging"
        df = pd.read_sql(sql_query, engine)

    return df

@task(name = "Transform to catalog_dim")
def catalog_dim_transform(data):
    # ensure only the latest date is inserted
    data = data[data['date'] == data["date"].max()]
    data = data[["catalog_id", "code", "title", "parent_id", "parent_title"]]
    return (data)

@task(name = "Transform to catalog_fact")
def catalog_fact_transform(data):
    # ensure only the latest date is inserted
    data = data[["catalog_id", "date", "item_count"]]
    return (data)

@task(name="Export data to catalog_dim")
def export_catalog_dim(data: pd.DataFrame, engine) -> None:
    """
    Exports metadata to a PostgreSQL table.

    Args:
        data (pd.DataFrame): Input DataFrame containing metadata.
        engine: The SQLAlchemy engine for the PostgreSQL database.

    Returns:
        None
    """
    #schema_name = 'public'  # Specify the name of the schema to export data to
    table_name = 'catalog_dim'  # Specify the name of the table to export data to
    data.to_sql(table_name, 
                engine, 
                if_exists = "append", 
                index = False,
                method=upsert_catalog_dim
                )
    
    return

@task(name="Export data to catalog_fact")
def export_catalog_fact(data: pd.DataFrame, engine) -> None:
    """
    Exports metadata to a PostgreSQL table.

    Args:
        data (pd.DataFrame): Input DataFrame containing metadata.
        engine: The SQLAlchemy engine for the PostgreSQL database.

    Returns:
        None
    """
    #schema_name = 'public'  # Specify the name of the schema to export data to
    #data = data[["catalog_id", "date", "item_count"]]
    table_name = 'catalog_fact'  # Specify the name of the table to export data to
    data.to_sql(table_name, 
                engine, 
                if_exists = "append", 
                index = False,
                method= upsert_catalog_fact
                )
    
    return