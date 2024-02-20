from prefect import flow
from sqlalchemy import create_engine
from catalog_staging_tasks import *

@flow(name = "Normalize catalog_staging")
def normalize_catalog_staging():
    engine = create_engine('postgresql://user:4202@localhost:5432/vinted-ai')
    data = load_from_catalog_staging(engine = engine)
    data_dim = catalog_dim_transform(data)
    data_fact = catalog_fact_transform(data)
    export_catalog_dim(data_dim, engine = engine)
    export_catalog_fact(data_fact, engine= engine)

if __name__ == "__main__":
    normalize_catalog_staging.serve(name="catalog-normalize-tables",
            tags=["ingestion", "postgresql"],
            pause_on_shutdown=False,
            interval=60*60*6)