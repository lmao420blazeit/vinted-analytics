from prefect import flow
from sqlalchemy import create_engine
from data_orchestration.prefect_tasks.tasks_vinted_brand import *

@flow(name= "Fetch from vinted", 
      log_prints= True)
def fetch_data_from_vinted():
    engine = create_engine('postgresql://user:4202@localhost:5432/vinted-ai')
    data = load_brands_from_api()
    data = transform(data)
    export_metadata_to_postgres(data)
    return

if __name__ == "__main__":
    # update brands https://www.vinted.pt/api/v2/brands
    # Run the flow interactively (this would typically be run by the Prefect agent in production)
    fetch_data_from_vinted.serve(name="vinted-brands",
                        tags=["onboarding"],
                        pause_on_shutdown=False,
                        interval=3600)