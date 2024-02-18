from prefect import task
import pandas as pd
from sqlalchemy import create_engine
from gsheets_api.iowrite import GoogleSheetsClient
import datetime
import numpy as np
#from ..utils import upsert_brands_dim

# name, catalog_id, unique products, total volume, median price, 
# date, median service fee, qq1, q2, q3, unique users, unique sizes

@task(name="Load from products_catalog.")
def load_from_catalog(engine):

    sql_query = """
                SELECT products_catalog.*, catalog_staging.title as catalog_title
                FROM products_catalog 
                LEFT JOIN catalog_staging 
                    ON products_catalog.catalog_id = catalog_staging.catalog_id
                """
    df = pd.read_sql(sql_query, engine)

    return df

@task(name="Transform to summary table.")
def transform_summary(data):
    transformed_data = (
        data.groupby("date")
            .agg(
                price_median=("price", "median"),
                price_std=("price", "std"),
                product_unique=("product_id", "nunique"),
                brand_unique=("brand_title", "nunique"),
                size_unique=("size_title", "nunique"),
                service_fee=("service_fee", "median"),
                users_unique=("user_id", "nunique"),
                total_volume=("price", "sum"),
                catalog_unique=("catalog_id", "nunique")
            )
            .reset_index()
    )
    transformed_data["products_per_user"] = transformed_data["product_unique"] / transformed_data["users_unique"]
    transformed_data["products_per_catalog"] = transformed_data["product_unique"] / transformed_data["catalog_unique"]
    return transformed_data


@task(name="Transform to brand sales table.")
def transform_brand_sale_performance(data):
    
    filtered_data = data.groupby(["date", "brand_title"]).filter(lambda group: group["product_id"].count() > 10)

    transformed_data = (
        filtered_data.groupby(["date", "brand_title"])
            .agg(
                price_median=("price", "median"),
                price_std=("price", "std"),
                product_unique=("product_id", "nunique"),
                service_fee=("service_fee", "median"),
                users_unique=("user_id", "nunique"),
                catalog_unique=("catalog_id", "nunique")
            )
            .reset_index()
    )
    # Filter brands with more than 10 unique products for each day
    # transformed_data["products_per_user"] = transformed_data["product_unique"] / transformed_data["users_unique"]
    #transformed_data["normalized_price_std"] = transformed_data["price_std"] / transformed_data["price_median"]
    return transformed_data


@task(name="Transform to catalog sales table.")
def transform_catalog_sale_performance(data):

    catalog_keys = data[["catalog_id", "catalog_title"]].drop_duplicates()

    transformed_data = (
        data.groupby(["date", "catalog_id"])
            .agg(
                price_median=("price", "median"),
                price_std=("price", "std"),
                product_unique=("product_id", "nunique"),
                service_fee=("service_fee", "median"),
                users_unique=("user_id", "nunique"),
                brand_unique=("brand_title", "nunique")
            )
            .reset_index()
    ).merge(
            catalog_keys, 
            on="catalog_id", 
            how="left"
            )
    # Filter brands with more than 10 unique products for each day
    # transformed_data["products_per_user"] = transformed_data["product_unique"] / transformed_data["users_unique"]
    #transformed_data["normalized_price_std"] = transformed_data["price_std"] / transformed_data["price_median"]
    return transformed_data

@task(name="Transform to products_table. (1 days)")
def transform_products_table(data):

    transformed_data = data[["title", "price", "brand_title", "catalog_title", "url", "size_title", "promoted", "status", "catalog_id", "date", "service_fee", "product_id", "user_id"]]
    transformed_data["date"] = pd.to_datetime(transformed_data["date"])
    transformed_data = transformed_data[transformed_data["date"] > datetime.datetime.now() - datetime.timedelta(1)]
    # Filter brands with more than 10 unique products for each day
    # transformed_data["products_per_user"] = transformed_data["product_unique"] / transformed_data["users_unique"]
    #transformed_data["normalized_price_std"] = transformed_data["price_std"] / transformed_data["price_median"]
    return transformed_data


@task(name="Asserts types and replaces NaN.")
def assert_types(data):
    # google sheets doesnt have native support for datetime objects, but strings
    columns = list(data.columns)
    data.fillna("", inplace=True)
    data.replace([np.inf, -np.inf], "")
    data = data.values.tolist()
    data = [[str(cell) if isinstance(cell, datetime.date) else cell for cell in row] for row in data]
    data = [columns] + data
    return data

@task(name="Export to googlesheets.",
      task_run_name= "Export to {sheet_name}")
def export_to_gsheets(data, sheet_name):
    GSHEET_CREDENTIALS_FILE = 'data_orchestration\credentials.json'
    GSHEET_ID = "12cd88ZmYvH-_LOQ4475XUFZ1NVelUivRa4VrzQdmuZM"

    client = GoogleSheetsClient(GSHEET_CREDENTIALS_FILE, GSHEET_ID)
    client.write(data = data, 
                 sheet_name = sheet_name)
    

if __name__ == "__main__":
    #engine = create_engine('postgresql://user:4202@localhost:5432/vinted-ai')
    #data = load_from_catalog(engine= engine)
    #data = transform_summary(data)
    #data = assert_types(data)
    #export_to_gsheets(data, sheet_name = "summary!A1")
    pass