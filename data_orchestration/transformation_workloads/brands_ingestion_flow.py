from prefect import flow
from sqlalchemy import create_engine
from data_orchestration.transformation_workloads.brands_ingestion import *

@flow(name = "Normalize brands staging")
def normalize_brands_staging():
    engine = create_engine('postgresql://user:4202@localhost:5432/vinted-ai')
    data = load_from_brands_interest(engine = engine)
    data_dim = brands_dim_transform(data)
    brands_interest_to_dim(data_dim, engine = engine)
    brands_interest_to_fact(data, engine= engine)

if __name__ == "__main__":
    normalize_brands_staging.serve(name="brands-ingestion-pg",
            tags=["ingestion", "postgresql"],
            pause_on_shutdown=False,
            interval=60*60*6)