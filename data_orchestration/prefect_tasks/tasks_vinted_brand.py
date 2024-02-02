from pyVinted.vinted import Vinted
import pandas as pd
from typing import List
from sqlalchemy import create_engine, exc
from prefect import task
from sqlalchemy.dialects.postgresql import insert

def insert_on_conflict_nothing(table, conn, keys, data_iter):
     data = [dict(zip(keys, row)) for row in data_iter]
     stmt = insert(table.table).values(data).on_conflict_do_nothing(index_elements=["id"])
     result = conn.execute(stmt)
     return result.rowcount

@task(name="Load from api")
def load_data_from_api() -> List[pd.DataFrame]:
    """
    Template for loading data from API
    """
    vinted = Vinted()
    brands = vinted.items.search_all_brands()
    
    return (brands)

@task(name="Transform data into brands dim.")
def transform_date_dim(data: pd.DataFrame):
    """
    Template code for a transformer block.

    Add more parameters to this function if this block has multiple parent blocks.
    There should be one parameter for each output variable from each parent block.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    # Specify your transformation logic here

    data = data.drop(["path", "url", "favourite_count", "pretty_favourite_count", "item_count", "pretty_item_count"],
            axis = 1)

    return (data)

@task(name="Transform data into brands fact.")
def transform_date_fact(data: pd.DataFrame):
    """
    Template code for a transformer block.

    Add more parameters to this function if this block has multiple parent blocks.
    There should be one parameter for each output variable from each parent block.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    # Specify your transformation logic here
    data = data[["id", "favourite_count", "item_count"]]

    return (data)


@task(name="Export data to brands dim.")
def export_data_to_dim_postgres(df: pd.DataFrame) -> None:
    """
    """
    #schema_name = 'public'  # Specify the name of the schema to export data to
    table_name = 'brands_dim'  # Specify the name of the table to export data to
    engine = create_engine('postgresql://user:4202@localhost:5432/vinted-ai')
    df.to_sql(table_name, engine, if_exists = "append", index = False, method = insert_on_conflict_nothing)

    return

@task(name="Export data to brands fact.")
def export_data_to_fact_postgres(df: pd.DataFrame) -> None:
    """
    """
    #schema_name = 'public'  # Specify the name of the schema to export data to
    table_name = 'brands_fact'  # Specify the name of the table to export data to
    engine = create_engine('postgresql://user:4202@localhost:5432/vinted-ai')
    df.to_sql(table_name, engine, if_exists = "append", index = False, method = insert_on_conflict_nothing)

    return