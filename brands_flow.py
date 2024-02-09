from prefect import flow
from data_orchestration.prefect_tasks.tasks_vinted_brand import *

@flow(name= "Fetch from vinted", 
      log_prints= True)
def fetch_brands_from_vinted():
    brand_ids = 1000
    chunk_size = 25
    load_brands(brand_ids= brand_ids,
                chunk_size= chunk_size)
    return

if __name__ == "__main__":
    # update brands https://www.vinted.pt/api/v2/brands
    # Run the flow interactively (this would typically be run by the Prefect agent in production)
    fetch_brands_from_vinted.serve(name="vinted-brands",
                        tags=["brands", "staging"],
                        pause_on_shutdown=False,
                        interval=60*60*24)