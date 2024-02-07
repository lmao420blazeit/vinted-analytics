from pyVinted.requester import requester
from urllib.parse import urlparse, parse_qsl
from requests.exceptions import HTTPError
from typing import List, Dict
from pyVinted.settings import Urls
import pandas as pd
import urllib
import numpy as np
import random
from time import sleep
import json

class Items:

    def search_all(self, nbrRows, *args, **kwargs):
        nbrpages = round(nbrRows/kwargs["batch_size"])
        df_list = []
        for _page in range(1, nbrpages + 1):
            d = self._search(*args, **kwargs, page = _page)
            if d.empty:
                return
            df_list.append(d)

        return(pd.concat(df_list, axis = 0, ignore_index=True))

    def _search(self, url, batch_size: int = 10, page: int =1, time: int = None) -> pd.DataFrame:
        """
        Retrieves items from a given search url on vinted.

        Args:
            url (str): The url of the research on vinted.
            batch_size (int): Number of items to be returned (default 20).
            page (int): Page number to be returned (default 1).

        """

        params = self.parseUrl(url, batch_size, page, time)
        url = f"{Urls.VINTED_BASE_URL}/{Urls.VINTED_API_URL}/{Urls.VINTED_PRODUCTS_ENDPOINT}?{urllib.parse.urlencode(params)}"
        response = requester.get(url=url)

        try:
            response.raise_for_status()
            items = response.json()
            
            df = pd.DataFrame(items["items"])
            df["catalog_total_items"] = items["pagination"]["total_entries"]
            return (df)

        except HTTPError as err:
            raise err
        
        
    def search_colors(self, max_retries = 3) -> pd.DataFrame:
        
        df_list = []
        #params = self.parseUrl(url)
        #url = f"{Urls.VINTED_BASE_URL}/{Urls.VINTED_API_URL}/{Urls.VINTED_PRODUCTS_ENDPOINT}?{urllib.parse.urlencode(params)}"
        for i in range(1, 50):
            url = f"https://www.vinted.pt/api/v2/catalog/items?color_ids[]={i}"
            response = requester.get(url=url)
            retries = 1
            backoff_factor= random.randint(7, 11)

            while retries <= max_retries:
                sleep(backoff_factor**retries)
                try:
                    response.raise_for_status()
                    items = response.json()
                    
                    df = pd.DataFrame({"catalog_total_items": [items["pagination"]["total_entries"]],
                                    "color_id": [i]})
                    df_list.append(df)

                except HTTPError as err:
                    raise err
                
            if retries == max_retries:
                raise RuntimeError(f"Failed to make the HTTP request after {max_retries} retries.") 
            
            return (pd.concat(df_list, axis=0, ignore_index= True))

    def search_item(self, user_id, time: int = None, max_retries=3) -> pd.DataFrame:
        """
        Retrieves items from a given search url on vinted.

        Args:
            url (str): The url of the research on vinted.
            batch_size (int): Number of items to be returned (default 20).
            page (int): Page number to be returned (default 1).

        """
        #endbyte = random.choice(["%00", "%0d%0a", "%0d", "%0a", "%09", "%0C"])
        user_agent = random.choice(["Mozilla/5.0 (Linux; Android 11; SM-G991B Build/RP1A.200720.012; wv) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.120 Mobile Safari/537.36",
                                    "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1 Mobile/15E148 Safari/604.1",
                                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
                                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"])
        url = f"{Urls.VINTED_BASE_URL}/{Urls.VINTED_API_URL}/users/{user_id}/items"
        retries = 1
        backoff_factor= random.randint(7, 11)

        while retries <= max_retries:
            try:
                sleep(backoff_factor**retries)
                # rotating user agents
                response = requester.get(url=url, 
                                         headers = {"User-Agent": user_agent})
                print(response.headers) 
                if response.status_code == 403:
                    return
                response.raise_for_status()
                items = response.json()

                cols = ["id", "brand", "size", "catalog_id", "color1_id", "favourite_count", 
                        "view_count", "created_at_ts", "original_price_numeric", "price_numeric"]
                # add; promoted_until, is_hidden, number of photos, description attributes (material)
                df = pd.DataFrame(items["items"])#.T
                
                print(df[cols])
                return df[cols]
                #return pd.DataFrame(columns = cols,
                #                    data = [[item_id] + [np.NaN for x in range(len(cols)-1)]])
            
            except HTTPError as err:
                print(err)
                retries += 1

        raise RuntimeError(f"Failed to make the HTTP request after {max_retries} retries.")
        
    def search_brands(self, max_retries = 3) -> pd.DataFrame:
        
        df_list = []
        #params = self.parseUrl(url)
        #url = f"{Urls.VINTED_BASE_URL}/{Urls.VINTED_API_URL}/{Urls.VINTED_PRODUCTS_ENDPOINT}?{urllib.parse.urlencode(params)}"
        for i in range(1, 50):
            url = f"https://www.vinted.pt/api/v2/catalog/items?brand_ids[]={str(i)}"
            response = requester.get(url=url)
            retries = 1
            backoff_factor= random.randint(7, 12)

            while retries <= max_retries:
                sleep(backoff_factor**retries)
                try:
                    response.raise_for_status()
                    items = response.json()
                    
                    df = pd.DataFrame(items["dominant_brand"])
                    df_list.append(df)
                    print(df)

                except HTTPError as err:
                    print(err)
                    raise err
                
                except:
                    return pd.concat(df_list, axis = 0, ignore_index=True)
                
            if retries == max_retries:
                raise RuntimeError(f"Failed to make the HTTP request after {max_retries} retries.") 
            
            return (pd.concat(df_list, axis=0, ignore_index= True))

    def parseUrl(self, url, batch_size=20, page=1, time=None) -> Dict:
        """
        Parse Vinted search url to get parameters the for api call.

        Args:
            url (str): The url of the research on vinted.
            batch_size (int): Number of items to be returned (default 20).
            page (int): Page number to be returned (default 1).

        """
        querys = parse_qsl(urlparse(url).query)

        params = {
            "search_text": "+".join(
                map(str, [tpl[1] for tpl in querys if tpl[0] == "search_text"])
            ),
            "catalog_ids": ",".join(
                map(str, [tpl[1] for tpl in querys if tpl[0] == "catalog_ids[]"])
            ),
            "color_ids": ",".join(
                map(str, [tpl[1] for tpl in querys if tpl[0] == "color_id[]"])
            ),
            "brand_ids": ",".join(
                map(str, [tpl[1] for tpl in querys if tpl[0] == "brand_ids[]"])
            ),
            "size_ids": ",".join(
                map(str, [tpl[1] for tpl in querys if tpl[0] == "size_id[]"])
            ),
            "material_ids": ",".join(
                map(str, [tpl[1] for tpl in querys if tpl[0] == "material_id[]"])
            ),
            "status_ids": ",".join(
                map(str, [tpl[1] for tpl in querys if tpl[0] == "status[]"])
            ),
            "country_ids": ",".join(
                map(str, [tpl[1] for tpl in querys if tpl[0] == "country_id[]"])
            ),
            "city_ids": ",".join(
                map(str, [tpl[1] for tpl in querys if tpl[0] == "city_id[]"])
            ),
            "is_for_swap": ",".join(
                map(str, [1 for tpl in querys if tpl[0] == "disposal[]"])
            ),
            "currency": ",".join(
                map(str, [tpl[1] for tpl in querys if tpl[0] == "currency"])
            ),
            "price_to": ",".join(
                map(str, [tpl[1] for tpl in querys if tpl[0] == "price_to"])
            ),
            "price_from": ",".join(
                map(str, [tpl[1] for tpl in querys if tpl[0] == "price_from"])
            ),
            "page": page,
            "per_page": batch_size,
            "order": ",".join(
                map(str, [tpl[1] for tpl in querys if tpl[0] == "order"])
            ),
            "time": time
        }
        filtered = {k: v for k, v in params.items() if v not in ["", None]}

        return filtered
