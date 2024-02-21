from brands_flow import *
from catalog_flow import *
from catalogids_flow import *
from tracking_flow import *
from prefect import serve

if __name__ == "__main__":
    vinted_brands = fetch_brands_from_vinted.to_deployment(name="staging-brands",
                    tags=["brands", "staging", "extraction", "api"],
                    interval=60*60*24) # 24h interval
    
    vinted_main = fetch_data_from_vinted.to_deployment(name="staging-main",
                        tags=["staging", "extraction", "api"],
                        parameters={"sample_frac": 0.001,
                                    "batch_size": 500,
                                    "nbrRows": 500,
                                    "item_ids": [221, 1242, 2320, 1811, 267, 1812, 98, 246, 287, 2964, 1806] 
                                    # [t shirts women, trainers, sweaters, books, hoodies and sweaters, zip hoodies, sunglasses, backpacks, caps, gorros, t-shirts men]
                                    },
                        interval=3600) # 1h interval
    
    vinted_catalog_ids = fetch_catalog_data.to_deployment(name="staging-catalog_ids",
                        tags=["staging", "extraction", "api", "catalog_ids"],
                        interval=60*60*24)
    
    vinted_tracking = tracking.to_deployment(name="staging-tracking",
            tags=["tracking", "api", "staging"],
            interval=60*20) # 30 mins interval
    
    serve(vinted_brands, vinted_main, vinted_catalog_ids, vinted_tracking,
          pause_on_shutdown=False)