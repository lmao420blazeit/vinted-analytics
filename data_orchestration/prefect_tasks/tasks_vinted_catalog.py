from pyVinted.vinted import Vinted
import pandas as pd
from typing import List
import datetime
from os import path
from sqlalchemy import create_engine
from prefect import task
from prefect.context import FlowRunContext
from ..utils import insert_on_conflict_nothing
from operator import itemgetter
import re
from prefect.artifacts import create_table_artifact

@task(name="Load from api", log_prints= True)
def load_data_from_api(nbrRows, batch_size, item) -> List[pd.DataFrame]:
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
    vinted = Vinted()
    date = datetime.datetime.now()
    items = vinted.items.search_all(nbrRows = nbrRows, 
                                batch_size = batch_size, 
                                url = f"https://www.vinted.pt/catalog/items?catalog_ids[]={item}&order=newest_first")
    # cant process latin characters
    items["user_id"] = items.user.apply(pd.Series)["id"]
    items["color"] = items.photo.apply(pd.Series)["dominant_color"]
    items["catalog_id"] = item
    items["date"] = date
    
    return (items)

@task(name="Drop columns and rename.")
def transform(data: pd.DataFrame):
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
    data["price"] = data["price"].astype(float)

    data = data.drop(["is_for_swap", "user", "photo", "is_favourite", "discount", "badge", "conversion", "service_fee", 
            "total_item_price_rounded", "icon_badges", "is_visible", "search_tracking_params", "favourite_count",
            "total_item_price", "view_count", "content_source"],
            axis = 1)
    
    data = data.rename(columns={'id': 'product_id'})
    data['date'] = pd.to_datetime(data['date']).dt.strftime('%Y-%m-%d %H:%M')
    return (data)

@task(name="Parse size_title into unique sizes (S, M, XL, XXL).")
def parse_size_title(data: pd.DataFrame):
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
    def check_string(string_to_check):
        # check size_title and parse sizes into unique sizes
        # M / 36 = M
        if re.match(r'^[a-zA-Z]', string_to_check):
            return (''.join(re.findall(r'[a-zA-Z]', string_to_check)))

        else:
            # first letter starts with a number
            return(string_to_check)
        
    data["size_title"] = data["size_title"].apply(check_string)

    return (data)

@task(name="Transform metadata.")
def transform_metadata(data: pd.DataFrame):
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
    flow_meta = FlowRunContext.get().flow_run.dict()
    flow_meta.update({"missing_values": data.isna().sum().sum(),
                      "total_rows": len(data.index)})
    flow_meta = itemgetter(*["name", "parameters", "missing_values", "total_rows"])(flow_meta)
    data = pd.DataFrame.from_dict(flow_meta).T
    data.columns = ["name", "missing_values", "total_rows"]

    return (data)

@task(name = "Create artifacts.")
def create_artifacts(df):
    create_table_artifact(key = "output-describe",
                          table = df.describe().reset_index().to_dict(orient='records'),
                          description= "output describe pandas")


@task(name="Export sample to pg")
def export_metadata_to_postgres(df: pd.DataFrame) -> None:
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
    #schema_name = 'public'  # Specify the name of the schema to export data to
    table_name = 'flow_metadata'  # Specify the name of the table to export data to
    engine = create_engine('postgresql://user:4202@localhost:5432/vinted-ai')
    df.to_sql(table_name, engine, if_exists = "append", index = False, method = insert_on_conflict_nothing)
    
    return

@task(name="Export sample to pg")
def export_sample_to_postgres(df: pd.DataFrame, sample_frac) -> None:
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
    #schema_name = 'public'  # Specify the name of the schema to export data to
    table_name = 'samples'  # Specify the name of the table to export data to
    engine = create_engine('postgresql://user:4202@localhost:5432/vinted-ai')

    df = df[["product_id", "catalog_id", "user_id", "date"]].groupby("catalog_id", group_keys=False).apply(lambda x: x.sample(frac = sample_frac))
    df.to_sql(table_name, engine, if_exists = "append", index = False, method = insert_on_conflict_nothing)
    
    return

@task(name="Export data to pg", log_prints=True)
def export_data_to_postgres(df: pd.DataFrame) -> None:
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
    #schema_name = 'public'  # Specify the name of the schema to export data to
    table_name = 'products_catalog'  # Specify the name of the table to export data to
    engine = create_engine('postgresql://user:4202@localhost:5432/vinted-ai')
    df["flow_name"] = FlowRunContext.get().flow_run.dict().get('name')
    df.to_sql(table_name, engine, if_exists = "append", index = False, method = insert_on_conflict_nothing)

    return