from prefect import flow, serve
from sqlalchemy import create_engine
from tasks.tasks_vinted_catalog import *
from tests.catalog_unit_test import test_fetch_data
from tasks.pyVinted.vinted import Vinted
import time

@flow(name= "Fetch from vinted", 
      log_prints= True,
      description= """
      Main flow: 
      start node: fetch vinted/items endpoint 
      -> simple preprocessing 
      -> dumps into postgres staging table""")
def fetch_data_from_vinted(sample_frac = 0.01, 
                           item_ids = [], 
                           batch_size = 500, 
                           nbrRows = 1000):
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
    vinted = Vinted()
    engine = create_engine('postgresql://user:4202@localhost:5432/vinted-ai')
    for __item in item_ids:
        catalog_subflow(item = __item, 
                        nbrRows= nbrRows,
                        batch_size= batch_size,
                        vinted = vinted, 
                        engine= engine, 
                        sample_frac= sample_frac)
        time.sleep(60)

    return

if __name__ == "__main__":
    # update brands https://www.vinted.pt/api/v2/brands
    # Run the flow interactively (this would typically be run by the Prefect agent in production)
    """
    deployment = fetch_data_from_vinted.to_deployment(name="vinted-v1",
                        tags=["staging", "extraction", "api"],
                        parameters={"sample_frac": 0.001,
                                    "batch_size": 500,
                                    "nbrRows": 500,
                                    "item_ids": [221, 1242, 2320, 1811, 267, 1812, 98, 246, 287, 2964] 
                                    # [t shirts, trainers, sweaters, books, hoodies and sweaters, zip hoodies, sunglasses, backpacks, caps, gorros]
                                    },
                        interval=3600)
    serve(deployment)"""