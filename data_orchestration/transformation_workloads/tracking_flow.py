from prefect import flow
from sqlalchemy import create_engine
from data_orchestration.transformation_workloads.tasks.tracking_ingestion_tasks import *

@flow(name = "Normalize tracking_staging")
def normalize_tracking_staging():
    engine = create_engine('postgresql://user:4202@localhost:5432/vinted-ai')
    data = load_from_tracking_staging(engine = engine)
    color_dim = color_dim_transform(data)
    fact_data = tracking_fact_transform(data)
    export_color_dim(color_dim, engine = engine)
    export_tracking_fact(fact_data, engine= engine)

if __name__ == "__main__":
    normalize_tracking_staging.serve(name="tracking-normalize-tables",
            tags=["ingestion", "postgresql"],
            pause_on_shutdown=False,
            interval=60*60*6)