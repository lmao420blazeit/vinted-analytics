import psycopg2
from google.oauth2 import service_account
from googleapiclient.discovery import build
import datetime

# PostgreSQL database connection settings
PG_HOST = 'localhost'
PG_DATABASE = 'vinted-ai'
PG_USER = 'user'
PG_PASSWORD = '4202'

# Google Sheets credentials
GSHEET_CREDENTIALS_FILE = 'data_orchestration\credentials.json'
GSHEET_SPREADSHEET_NAME = 'products_catalog'
GSHEET_WORKSHEET_NAME = 'products_catalog'
GSHEET_ID = "12cd88ZmYvH-_LOQ4475XUFZ1NVelUivRa4VrzQdmuZM"

# Connect to PostgreSQL database
conn = psycopg2.connect(host=PG_HOST, database=PG_DATABASE, user=PG_USER, password=PG_PASSWORD)
cursor = conn.cursor()

# Query data from PostgreSQL table
cursor.execute("SELECT * FROM products_catalog ORDER BY date DESC LIMIT 5000")
data = cursor.fetchall()

# Authenticate with Google Sheets API
#scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
# Authenticate with Google Sheets API
credentials = service_account.Credentials.from_service_account_file(GSHEET_CREDENTIALS_FILE)
service = build('sheets', 'v4', credentials=credentials)

# TypeError: Object of type date is not JSON serializable
# Convert date objects to string representation
data = [[str(cell) if isinstance(cell, datetime.date) else cell for cell in row] for row in data]

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

# Write data to Google Sheets
sheet = service.spreadsheets()
body = {
    'values': data
}
sheet.values().update(
    spreadsheetId= GSHEET_ID,
    range= 'Sheet1!A1',  # Update Sheet1 starting from cell A1
    valueInputOption= 'RAW',
    body= body
).execute()

# Close PostgreSQL connection
cursor.close()
conn.close()

print('Data uploaded to Google Sheets successfully!')