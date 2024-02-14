from google.oauth2 import service_account
from googleapiclient.discovery import build

class GoogleSheetsClient:
    def __init__(self, credentials_file, spreadsheet_id):
        self.credentials = service_account.Credentials.from_service_account_file(credentials_file)
        self.service = build('sheets', 'v4', credentials=self.credentials)
        self.spreadsheet_id = spreadsheet_id

    def write(self, data, sheet = 'Sheet1!A1'):
        """
        Full load.
        Write data to Google Sheets.
        :param data: List of lists representing rows and columns of data.
        :param range_name: Range in A1 notation where the data will be written, e.g., 'Sheet1!A1'.
        :return: None
        """
        sheet = self.service.spreadsheets()
        body = {'values': data}
        sheet.values().update(
            spreadsheetId=self.spreadsheet_id,
            range='Sheet1!A1',
            valueInputOption='RAW',
            body=body
        ).execute()

# Google Sheets credentials
GSHEET_CREDENTIALS_FILE = 'data_orchestration\credentials.json'
GSHEET_SPREADSHEET_NAME = 'products_catalog'
GSHEET_WORKSHEET_NAME = 'products_catalog'
GSHEET_ID = "12cd88ZmYvH-_LOQ4475XUFZ1NVelUivRa4VrzQdmuZM"

# Authenticate with Google Sheets API
#scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
# Authenticate with Google Sheets API
credentials = service_account.Credentials.from_service_account_file(GSHEET_CREDENTIALS_FILE)
service = build('sheets', 'v4', credentials=credentials)

if __name__ == "__main__":
    # Initialize GoogleSheetsClient
    sheets_client = GoogleSheetsClient(GSHEET_CREDENTIALS_FILE, GSHEET_ID)  