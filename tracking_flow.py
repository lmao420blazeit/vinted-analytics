from data_orchestration.prefect_tasks.tasks_vinted_tracking import *
from prefect import flow
import psycopg2

@flow(name = "Tracking", log_prints= True)
def tracking():
    db_params = {
        "host": "localhost",
        "database": "vinted-ai",
        "user": "user",
        "password": "4202",
    }

    # Establish a connection to the PostgreSQL database
    conn = psycopg2.connect(**db_params)
    data = load_data_from_postgres(conn)
    load_balancer(data)


if __name__ == "__main__":
    # Replace these values with your actual database connection details
    tracking.serve(name="vinted-tracking",
            tags=["tracking"],
            pause_on_shutdown=False,
            interval=60*60*24)