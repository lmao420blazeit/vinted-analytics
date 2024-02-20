from prefect import flow
from sqlalchemy import create_engine
from data_orchestration.transformation_workloads.catalog_staging_tasks import *

@flow(name = "Normalize catalog_staging")
def normalize_catalog_staging():
    engine = create_engine('postgresql://user:4202@localhost:5432/vinted-ai')
    data = load_from_catalog_staging(engine = engine)
    data_dim = catalog_dim_transform(data)
    catalog_to_dim(data_dim, engine = engine)
    catalog_to_fact(data, engine= engine)

if __name__ == "__main__":
    normalize_catalog_staging.serve(name="brands-ingestion-pg",
            tags=["ingestion", "postgresql"],
            pause_on_shutdown=False,
            interval=60*60*6)