from pyVinted.vinted import Vinted
from prefect import task
import pandas as pd


@task(name="Load brands data from api.")
def load_brands_from_api():
    vinted = Vinted()
    items = vinted.items.search_brands()
    return(items)


@task(name="Drop columns and rename.")
def transform(data: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms the input DataFrame by dropping columns, renaming columns, and converting data types.

    Args:
        data (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Transformed DataFrame.
    """
    # Specify your transformation logic here

    data = data.drop(["path", "url", "is_favourite", "slug", "pretty_item_count", "requires_authenticity_check"],
            axis = 1)
    
    data = data.rename(columns={'id': 'brand_id'})
    data['date'] = pd.to_datetime(data['date']).dt.strftime('%Y-%m-%d %H:%M')
    return (data)

@task(name="Export sample to pg")
def export_metadata_to_postgres(data: pd.DataFrame, engine) -> None:
    """
    Exports metadata to a PostgreSQL table.

    Args:
        data (pd.DataFrame): Input DataFrame containing metadata.
        engine: The SQLAlchemy engine for the PostgreSQL database.

    Returns:
        None
    """
    #schema_name = 'public'  # Specify the name of the schema to export data to
    table_name = 'brands_interest'  # Specify the name of the table to export data to
    data.to_sql(table_name, 
                engine, 
                if_exists = "append", 
                index = False)
    
    return