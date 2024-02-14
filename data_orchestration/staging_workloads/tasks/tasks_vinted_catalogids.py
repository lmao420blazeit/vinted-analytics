from tasks.pyVinted.vinted import Vinted
import pandas as pd
from typing import List
import datetime
from os import path
from prefect import task
from prefect.context import FlowRunContext
from .utils import upsert_catalog_staging
#from operator import itemgetter
#import re
#from prefect.artifacts import create_table_artifact
from prefect.tasks import exponential_backoff

@task(name="Parse HTML catalogs.", 
      log_prints= True,
      retries=3, 
      retry_delay_seconds=exponential_backoff(backoff_factor=6),
      retry_jitter_factor=2)
def parse_html_catalog() -> pd.DataFrame:
    """

    """
    vinted = Vinted()
    date = datetime.datetime.now()
    items = vinted.items.get_catalog_ids()
    items["date"] = date
    
    return (items)

@task(name="Export data to pg", 
      log_prints=True)
def export_data_to_postgres(data: pd.DataFrame, engine) -> None:
    """
    Exports a pandas DataFrame to a specified PostgreSQL table using the provided database engine.

    Args:
        data (pd.DataFrame): The DataFrame to be exported to PostgreSQL.
        engine: The SQLAlchemy engine for the PostgreSQL database.

    Returns:
        None
    """
    #schema_name = 'public'  # Specify the name of the schema to export data to
    table_name = 'catalog_staging'  # Specify the name of the table to export data to
    #engine = create_engine('postgresql://user:4202@localhost:5432/vinted-ai')
    data.to_sql(table_name, 
                engine, 
                if_exists = "append", 
                index = False, 
                method = upsert_catalog_staging,
                schema= "public")

    return