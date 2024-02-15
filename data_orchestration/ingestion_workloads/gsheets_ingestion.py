from gsheets_api.iowrite import GoogleSheetsClient
import pandas as pd
import datetime
import psycopg2


def load_data_from_postgres(conn):
    yesterday = datetime.datetime.now() - datetime.timedelta(days = 1)
    cursor = conn.cursor()
    query = f"SELECT * FROM products_catalog WHERE date = '{yesterday}'"
    cursor.execute(query)
    data = cursor.fetchall()

    # Query to fetch column names and data types
    cursor.execute("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public'
        AND table_name = 'products_catalog'
    """)
    columns_info = cursor.fetchall()
    column_names = [column_info[0] for column_info in columns_info]

    data = [column_names] + data
    #df = pd.DataFrame(rows)
    return data

def assert_types(data):
    # google sheets doesnt have native support for datetime objects, but strings
    data = [[str(cell) if isinstance(cell, datetime.date) else cell for cell in row] for row in data]
    return data

def export_to_gsheets(data):
    GSHEET_CREDENTIALS_FILE = 'data_orchestration\credentials.json'
    GSHEET_ID = "12cd88ZmYvH-_LOQ4475XUFZ1NVelUivRa4VrzQdmuZM"

    client = GoogleSheetsClient(GSHEET_CREDENTIALS_FILE, GSHEET_ID)
    client.write(data)
    

if __name__ == "__main__":
    db_params = {
        "host": "localhost",
        "database": "vinted-ai",
        "user": "user",
        "password": "4202",
    }

    # Establish a connection to the PostgreSQL database
    conn = psycopg2.connect(**db_params)
    data = load_data_from_postgres(conn)
    data = assert_types(data)
    export_to_gsheets(data)

