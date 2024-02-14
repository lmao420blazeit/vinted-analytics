import json
import requests
import re
import random
from requests.exceptions import HTTPError
from .settings import Urls
import time



HEADERS = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0",  # random.choice(USER_AGENTS),
            "Host": "www.vinted.pt",
            "Referer": "www.vinted.pt"
}

MAX_RETRIES = 3
class Requester:


    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.VINTED_AUTH_URL = f"{Urls.VINTED_BASE_URL}/auth/token_refresh"
        #self.setCookies()



    def get(self, url, params=None, **kwargs):
        """
        Perform a http get request.
        :param url: str
        :param params: dict, optional
        :return: dict
            Json format
        """
        tried = 0
        time.sleep(1)
        while tried < MAX_RETRIES:
            tried += 1

            with self.session.get(url, params=params, **kwargs) as response:

                if response.status_code == 401 and tried < MAX_RETRIES:
                    print(f"Cookies invalid retrying {tried}/{MAX_RETRIES}")
                    self.setCookies()

                elif response.status_code == 200 or tried == MAX_RETRIES:
                    return response


        return HTTPError

    def post(self,url, params=None):
        response = self.session.post(url, params)
        response.raise_for_status()
        return response

    def setCookies(self):
        self.session.cookies.clear_session_cookies()

        try:
            self.post(self.VINTED_AUTH_URL)

        except Exception as e:
            print(
                f"There was an error fetching cookies for vinted\n Error : {e}"
            )

    # def login(self,username,password=None):

    #     # client.headers["X-Csrf-Token"] = csrf_token
    #     # client.headers["Content-Type"] = "*/*"
    #     # client.headers["Host"] = "www.vinted.pt"
    #     print(self.session.headers)
    #     urlCaptcha = "https://www.vinted.pt/api/v2/captchas"
    #     dataCaptcha = {"entity_type":"login", "payload":{"username": username }}

    #     token_endpoint  = "https://www.vinted.pt/oauth/token"
    #     uuid = self.session.post(urlCaptcha, data=json.dumps(dataCaptcha)).json()["uuid"]
    #     log = {"client_id":"web","scope":"user","username":username,"password":password,"uuid":uuid,"grant_type":"password"}
    #     b = self.session.post(token_endpoint, data=json.dumps(log) )
    #     print(b.text)

    # def message(self):
    #     response = self.session.get("https://www.vinted.pt/api/v2/users/33003526/msg_threads?page=1&per_page=20")
    #     print(response.text)


requester = Requester()
