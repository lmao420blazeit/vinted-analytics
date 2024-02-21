from prefect import task
import pandas as pd
from utils import (insert_on_conflict_city_dim, 
                   insert_on_conflict_country_dim, 
                   insert_on_conflict_nothing_user,
                   insert_on_conflict_users_fact)

@task(name="Load users_staging for backfilling")
def load_from_users_staging(engine):
    try:
        # try delta load
        latest_record = "SELECT MAX(date) FROM users_fact"
        date = pd.read_sql(latest_record, engine)["date"]
        sql_query = f"SELECT * FROM users_staging WHERE date > '{date}'"
        df = pd.read_sql(sql_query, engine)
        
    except:
        # full load
        sql_query = "SELECT * FROM users_staging"
        df = pd.read_sql(sql_query, engine)

    return df

@task(name = "Transform to 'city_dim'")
def city_dim_transform(data):
    # ensure only the latest date is inserted
    data = data[data['date'] == data["date"].max()]
    data = data[["city", "country_id", "city_id"]]
    data = data.dropna(subset = ["city_id"])

    return (data)

@task(name = "Transform to 'country_dim'")
def country_dim_transform(data):
    # ensure only the latest date is inserted
    # data = data[data['date'] == data["date"].max()]
    data = data[["country_id", "country_title"]]
    data = data.dropna(subset = ["country_id"])

    return (data)

@task(name = "Transform to 'users_dim'")
def users_dim_transform(data):
    # ensure only the latest date is inserted
    #data = data[data['date'] == data["date"].max()]
    data = data[["user_id", "gender", "profile_url"]]
    data = data.dropna(subset = ["user_id"])

    return (data)

@task(name = "Transform to 'users_fact'")
def users_fact_transform(data):
    # ensure only the latest date is inserted
    #data = data[data['date'] == data["date"].max()]
    data = data[["user_id", "item_count", "given_item_count", "taken_item_count", "followers_count", 
                 "following_count", "positive_feedback_count", "negative_feedback_count", "feedback_count", 
                 "date", "city_id", "country_id"]]
    data = data.dropna(subset = ["user_id", "date"])

    return (data)

@task(name="Export to 'city_dim'")
def export_city_dim(data: pd.DataFrame, engine) -> None:
    """
    Exports metadata to a PostgreSQL table.

    Args:
        data (pd.DataFrame): Input DataFrame containing metadata.
        engine: The SQLAlchemy engine for the PostgreSQL database.

    Returns:
        None
    """
    #schema_name = 'public'  # Specify the name of the schema to export data to
    table_name = 'city_dim'  # Specify the name of the table to export data to
    data.to_sql(table_name, 
                engine, 
                if_exists = "append", 
                index = False,
                method= insert_on_conflict_city_dim
                )
    
    return

@task(name="Export to 'users_fact'")
def export_users_fact(data: pd.DataFrame, engine) -> None:
    """
    Exports metadata to a PostgreSQL table.

    Args:
        data (pd.DataFrame): Input DataFrame containing metadata.
        engine: The SQLAlchemy engine for the PostgreSQL database.

    Returns:
        None
    """
    #schema_name = 'public'  # Specify the name of the schema to export data to
    table_name = 'users_fact'  # Specify the name of the table to export data to
    data.to_sql(table_name, 
                engine, 
                if_exists = "append", 
                index = False,
                method= insert_on_conflict_users_fact
                )
    
    return

@task(name="Export to 'country_dim'")
def export_country_dim(data: pd.DataFrame, engine) -> None:
    """
    Exports metadata to a PostgreSQL table.

    Args:
        data (pd.DataFrame): Input DataFrame containing metadata.
        engine: The SQLAlchemy engine for the PostgreSQL database.

    Returns:
        None
    """
    #schema_name = 'public'  # Specify the name of the schema to export data to
    table_name = 'country_dim'  # Specify the name of the table to export data to
    data.to_sql(table_name, 
                engine, 
                if_exists = "append", 
                index = False,
                method= insert_on_conflict_country_dim
                )
    
    return

@task(name="Export to 'users_dim'")
def export_users_dim(data: pd.DataFrame, engine) -> None:
    """
    Exports metadata to a PostgreSQL table.

    Args:
        data (pd.DataFrame): Input DataFrame containing metadata.
        engine: The SQLAlchemy engine for the PostgreSQL database.

    Returns:
        None
    """
    #schema_name = 'public'  # Specify the name of the schema to export data to
    table_name = 'users_dim'  # Specify the name of the table to export data to
    data.to_sql(table_name, 
                engine, 
                if_exists = "append", 
                index = False,
                method= insert_on_conflict_nothing_user
                )
    
    return
