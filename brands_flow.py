from data_orchestration.prefect_tasks.tasks_vinted_brand import *
from prefect import flow

@flow(name = "Brands", log_prints= True)
def brands():
    data = load_data_from_api()
    data2 = transform_date_dim(data)
    data = transform_date_fact(data)
    res = export_data_to_dim_postgres(data2)
    export_data_to_fact_postgres(data,
                                 wait_for=[res])
    return


if __name__ == "__main__":
    # Replace these values with your actual database connection details
    db_params = {
        "host": "localhost",
        "database": "vinted-ai",
        "user": "user",
        "password": "4202",
    }

    brands.serve(name="vinted-brands",
                    tags=["brands"],
                    pause_on_shutdown=False,
                    interval=60*60*24)