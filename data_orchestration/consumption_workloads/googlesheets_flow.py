from prefect import flow
from sqlalchemy import create_engine
from googlesheets_ingestion_tasks import *

@flow(name = "Feed locker studio reports.",
      log_prints= True)
def serve_googlesheets():
    engine = create_engine('postgresql://user:4202@localhost:5432/vinted-ai')

    data = load_from_catalog(engine = engine)

    # transforms
    summary = transform_summary(data)
    brand_sale_performance = transform_brand_sale_performance(data)
    catalog_sale_performance = transform_catalog_sale_performance(data)
    products_records = transform_products_table(data)

    # asserts types and returns dataframe as list
    # has to be executed after data manipulation
    summary = assert_types(summary)
    brand_sale_performance = assert_types(brand_sale_performance)
    catalog_sale_performance = assert_types(catalog_sale_performance)
    products_records = assert_types(products_records)
    #print(products_records)

    # exports
    export_to_gsheets(data = summary,
                      sheet_name= "summary!A1")
    export_to_gsheets(data = brand_sale_performance,
                      sheet_name= "brand_sales!A1")
    export_to_gsheets(data = catalog_sale_performance,
                      sheet_name= "catalog_sales!A1")
    export_to_gsheets(data = products_records,
                      sheet_name= "products_records!A1")

if __name__ == "__main__":
    serve_googlesheets.serve(name="google-sheets summary",
            tags=["ingestion", "google-sheets"],
            pause_on_shutdown=False,
            interval=60*60*6)