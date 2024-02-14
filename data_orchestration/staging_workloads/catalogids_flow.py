from prefect import flow
from sqlalchemy import create_engine
from tasks.tasks_vinted_catalogids import *

@flow(name= "Fetch catalog data.", 
      log_prints= True)
def fetch_catalog_data():
    """
    Fetches data from the vinted/items endpoint, preprocesses it, and exports it to a PostgreSQL staging table.

    Parameters:
    - sample_frac (float): Fraction of data to sample.
    - item_ids (list): List of item IDs to fetch data for.
    - batch_size (int): Size of each batch to fetch.
    - nbrRows (int): Number of rows to fetch.

    Returns:
    None
    """
    engine = create_engine('postgresql://user:4202@localhost:5432/vinted-ai')
    data = parse_html_catalog()
    export_data_to_postgres(data = data,
                            engine= engine)

    return

if __name__ == "__main__":
    # update brands https://www.vinted.pt/api/v2/brands
    # Run the flow interactively (this would typically be run by the Prefect agent in production)
    #fetch_catalog_data.serve(name="vinted-catalog_ids",
    #                    tags=["staging", "extraction", "api", "catalog_ids"],
    #                    pause_on_shutdown=False,
    #                    interval=60*60*24)
    pass