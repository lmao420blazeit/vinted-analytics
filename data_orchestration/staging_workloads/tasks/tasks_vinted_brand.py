from tasks.pyVinted.vinted import Vinted
from prefect import task, flow
import pandas as pd
from prefect.tasks import exponential_backoff
import time
from sqlalchemy import create_engine
import datetime
from .utils import insert_on_conflict_nothing_brands


# orchestrator pattern
def load_brands(brand_ids, chunk_size):
    for chunk in range(chunk_size, brand_ids, chunk_size):
        #start_time = datetime.utcnow() + timedelta(seconds=interval*i)
        chunk_list = [i for i in range(1, chunk + 1)]
        # decouple executions
        brands_subflow(name = f"Subflow for chunk: {str(chunk-chunk_size)}-{str(chunk)}",
                       brand_ids = chunk_list)
        time.sleep(600)

@flow(flow_run_name= "{name}")
def brands_subflow(name, brand_ids):
    vinted_obj = Vinted()
    engine = create_engine('postgresql://user:4202@localhost:5432/vinted-ai')
    data = load_brands_from_api(vinted_obj = vinted_obj, 
                                brand_ids = brand_ids)
    data = transform(data)
    export_metadata_to_postgres(data, 
                                engine= engine)

@task(name="Load brands data from api.",
      retries=3, 
      retry_delay_seconds=exponential_backoff(backoff_factor=7),
      retry_jitter_factor=2)
def load_brands_from_api(vinted_obj, brand_ids):
    df_list = []
    for brand in brand_ids:
        items = vinted_obj.items.search_brands(brand)
        df_list.append(items)
        time.sleep(5)

    return(pd.concat(df_list, 
                     axis = 0, 
                     ignore_index= True))


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
    data = data.drop([data.columns[0], "path", "url", "is_favourite", "slug", "pretty_item_count", "requires_authenticity_check", "pretty_favourite_count"],
            axis = 1)
    
    data = data.rename(columns={'id': 'brand_id'})
    data['date'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    data = data.dropna(subset=['brand_id'])
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
                index = False,
                method = insert_on_conflict_nothing_brands)
    
    return