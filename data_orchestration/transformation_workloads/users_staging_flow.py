from prefect import flow
from sqlalchemy import create_engine
from data_orchestration.transformation_workloads.tasks.users_staging_tasks import *

@flow(name = "Normalize users_staging")
def normalize_users_staging():
    engine = create_engine('postgresql://user:4202@localhost:5432/vinted-ai')
    data = load_from_users_staging(engine = engine)
    city_dim = city_dim_transform(data)
    country_dim = country_dim_transform(data)
    users_dim = users_dim_transform(data)
    users_fact = users_fact_transform(data)
    export_country_dim(country_dim, 
                       engine = engine)    
    export_city_dim(city_dim, 
                    engine = engine)
    export_users_dim(users_dim, 
                     engine = engine)
    export_users_fact(users_fact, 
                     engine = engine)

if __name__ == "__main__":
    normalize_users_staging.serve(name="users-normalize-tables",
            tags=["ingestion", "postgresql"],
            pause_on_shutdown=False,
            interval=60*60*6)