from data_orchestration.prefect_tasks.tasks_vinted_catalog import *
from prefect import flow

@flow(name= "Fetch from vinted", log_prints= True)
def fetch_data_from_vinted(sample_frac = 0.01, 
                           item_ids = [221, 1231, 76], 
                           batch_size = 100, 
                           nbrRows = 300):
    df_list = []
    for __item in [221, 1231, 76]:
        df = load_data_from_api(nbrRows = nbrRows,
                        batch_size = batch_size,
                        item = __item)
        df_list.append(df)

    df = pd.concat(df_list, 
                   ignore_index= True)
    #df2 = transform_metadata(df)
    #export_metadata_to_postgres(df2)
    df = transform(df)
    df = parse_size_title(df)
    create_artifacts(df)
    export_data_to_postgres(df)     # upload first to products due to FK referencing
    export_sample_to_postgres(df, 
                              sample_frac= sample_frac)
    return

if __name__ == "__main__":
    # update brands https://www.vinted.pt/api/v2/brands
    # Run the flow interactively (this would typically be run by the Prefect agent in production)
    fetch_data_from_vinted.serve(name="vinted-v1",
                        tags=["onboarding"],
                        parameters={"sample_frac": 0.01,
                                    "item_ids": 200,
                                    "batch_size": 100,
                                    "nbrRows": 300,
                                    "item_ids": [221, 1231, 76]
                                    },
                        pause_on_shutdown=False,
                        interval=3600)