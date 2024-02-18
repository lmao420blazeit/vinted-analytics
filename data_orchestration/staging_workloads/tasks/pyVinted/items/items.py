from ..requester import requester
from urllib.parse import urlparse, parse_qsl
from requests.exceptions import HTTPError
from typing import List, Dict
from ..settings import Urls
import pandas as pd
import urllib
import random
from time import sleep
from lxml import html
import json
from PIL import Image
import io

class Items:

    def get_catalog_ids(self):
        # using HTML parser
        res = requester.get("https://www.vinted.pt/catalog?search_text=", 
                    headers ={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36",
                                'referer':'https://www.google.com/'})
        tree = html.fromstring(res.content)

        # Extract the content using the XPath selector
        next_data_content = tree.xpath('//*[@id="__NEXT_DATA__"]/text()')

        

        data = json.loads(next_data_content[0])["props"]["pageProps"]["_layout"]["catalogTree"]
        df_list = []
        for catalog in data:
            for i in range(len(data)):
                for f in range(len(data[i]["catalogs"])):
                    df_list.append(pd.DataFrame({"catalog_id": [data[i]["catalogs"][f]["id"]], 
                                                "code": [data[i]["catalogs"][f]["code"]],
                                                "title": [data[i]["catalogs"][f]["title"]],
                                                "item_count": [data[i]["catalogs"][f]["item_count"]],
                                                "unisex_catalog_id": [data[i]["catalogs"][f]["unisex_catalog_id"]],
                                                "parent_id": [data[i]["id"]],
                                                "parent_title": [data[i]["title"]]
                                                })
                                                )
                    for l in range(len(data[i]["catalogs"][f]["catalogs"])):
                        df_list.append(pd.DataFrame({"catalog_id": [data[i]["catalogs"][f]["catalogs"][l]["id"]], 
                                                    "code": [data[i]["catalogs"][f]["catalogs"][l]["code"]],
                                                    "title": [data[i]["catalogs"][f]["catalogs"][l]["title"]],
                                                    "item_count": [data[i]["catalogs"][f]["catalogs"][l]["item_count"]],
                                                    "unisex_catalog_id": [data[i]["catalogs"][f]["catalogs"][l]["unisex_catalog_id"]],
                                                    "parent_id": [data[i]["catalogs"][f]["id"]],
                                                    "parent_title": [data[i]["catalogs"][f]["code"]]
                                                    })
                                                    )
                

        return(pd.concat(df_list, ignore_index= True))

    def search_all(self, nbrRows, *args, **kwargs):
        nbrpages = round(nbrRows/kwargs["batch_size"])
        df_list = []
        for _page in range(1, nbrpages + 1):
            d = self.search_catalog(*args, **kwargs, page = _page)
            if d.empty:
                return
            df_list.append(d)

        return(pd.concat(df_list, axis = 0, ignore_index=True))

    def search_catalog(self, url, batch_size: int = 10, page: int =1) -> pd.DataFrame:
        """
        Retrieves items from a given search url on vinted.

        Args:
            url (str): The url of the research on vinted.
            batch_size (int): Number of items to be returned (default 20).
            page (int): Page number to be returned (default 1).

        """

        params = self.parseUrl(url, batch_size, page)
        url = f"{Urls.VINTED_BASE_URL}/{Urls.VINTED_API_URL}/{Urls.VINTED_PRODUCTS_ENDPOINT}?{urllib.parse.urlencode(params)}"
        response = requester.get(url=url)

        try:
            response.raise_for_status()
            items = response.json()
            
            df = pd.DataFrame(items["items"])
            df["catalog_total_items"] = items["pagination"]["total_entries"]
            #print(items["items"][0]["photo"]["thumbnails"][0]["url"])
            #print(requester.get(items["items"][0]["photo"]["thumbnails"][0]["url"], stream = True).raw.data)
            #images = [Image.open(io.BytesIO(requester.get(items["items"][i]["photo"]["thumbnails"][0]["url"]), stream = True).raw.data).convert("L") for i in range(len(items["items"]))]
            #print(images)

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

    def search_item(self, user_id, time: int = None) -> pd.DataFrame:
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

        # rotating user agents
        response = requester.get(url=url, 
                                    headers = {"User-Agent": user_agent})
        
        if response.status_code == 403:
            return
        
        response.raise_for_status()
        res = response.json()

        # add; promoted_until, is_hidden, number of photos, description attributes (material)
        items = pd.DataFrame(res["items"])#.T
        users = pd.DataFrame([res["items"][i]["user"] for i in range(len(res["items"]))])
        
        
        return [items, users]

        
    def search_brands(self, brand_id) -> pd.DataFrame:
        df_list = []
        #params = self.parseUrl(url)
        #url = f"{Urls.VINTED_BASE_URL}/{Urls.VINTED_API_URL}/{Urls.VINTED_PRODUCTS_ENDPOINT}?{urllib.parse.urlencode(params)}"
        url = f"https://www.vinted.pt/api/v2/catalog/items?brand_ids[]={str(brand_id)}"
        response = requester.get(url=url)
        response.raise_for_status()
        items = response.json()
        if "dominant_brand" not in items:
            return
        
        df = pd.DataFrame([items["dominant_brand"]])
        return(df)

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
