import numpy as np

def monte_carlo_simulation(expected_book_cost_mean, expected_book_cost_std,
                           expected_selling_price_mean, expected_selling_price_std,
                           shipping_costs_mean, shipping_costs_std, num_simulations):
    # Generate random samples for expected book cost, expected selling price, and shipping costs
    expected_book_cost_samples = np.random.normal(expected_book_cost_mean, expected_book_cost_std, num_simulations)
    expected_selling_price_samples = np.random.normal(expected_selling_price_mean, expected_selling_price_std, num_simulations)
    shipping_costs_samples = np.random.normal(shipping_costs_mean, shipping_costs_std, num_simulations)
    
    # Calculate expected gross margin for each simulation
    expected_gross_margin_samples = expected_selling_price_samples - (expected_book_cost_samples + shipping_costs_samples)
    
    return expected_gross_margin_samples

# Example usage
expected_book_cost_mean = 4.0
expected_book_cost_std = 1.5
expected_selling_price_mean = 8
expected_selling_price_std = 2.0
shipping_costs_mean = 2.0
shipping_costs_std = 0.5
num_simulations = 10000

expected_gross_margin_samples = monte_carlo_simulation(expected_book_cost_mean, expected_book_cost_std,
                                                       expected_selling_price_mean, expected_selling_price_std,
                                                       shipping_costs_mean, shipping_costs_std, num_simulations)

# Print mean and standard deviation of expected gross margin
print("Mean expected gross margin:", np.mean(expected_gross_margin_samples)/(expected_book_cost_mean+shipping_costs_mean))
print("Standard deviation of expected gross margin:", np.std(expected_gross_margin_samples)/(expected_book_cost_mean+shipping_costs_mean))

import requests
from lxml import html
import json

req = requests.Session()
res = req.get("https://www.vinted.pt/catalog?search_text=", 
              headers ={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36",
                        'referer':'https://www.google.com/'})
tree = html.fromstring(res.content)

# Extract the content using the XPath selector
next_data_content = tree.xpath('//*[@id="__NEXT_DATA__"]/text()')

import pandas as pd

data = json.loads(next_data_content[0])["props"]["pageProps"]["_layout"]["catalogTree"]
df_list = []
for catalog in data:
    for i in range(len(data)):
        for f in range(len(data[i]["catalogs"])):
            df_list.append(pd.DataFrame({"id": [data[i]["catalogs"][f]["id"]], 
                                        "code": [data[i]["catalogs"][f]["code"]],
                                        "title": [data[i]["catalogs"][f]["title"]],
                                        "item_count": [data[i]["catalogs"][f]["item_count"]],
                                        "unisex_catalog_id": [data[i]["catalogs"][f]["unisex_catalog_id"]],
                                        "parent_id": [data[i]["id"]],
                                        "parent_title": [data[i]["title"]]
                                        })
                                        )
            for l in range(len(data[i]["catalogs"][f]["catalogs"])):
                df_list.append(pd.DataFrame({"id": [data[i]["catalogs"][f]["catalogs"][l]["id"]], 
                                            "code": [data[i]["catalogs"][f]["catalogs"][l]["code"]],
                                            "title": [data[i]["catalogs"][f]["catalogs"][l]["title"]],
                                            "item_count": [data[i]["catalogs"][f]["item_count"]],
                                            "unisex_catalog_id": [data[i]["catalogs"][f]["catalogs"][l]["unisex_catalog_id"]],
                                            "parent_id": [data[i]["catalogs"][f]["id"]],
                                            "parent_title": [data[i]["catalogs"][f]["code"]]
                                            })
                                            )
        

print(pd.concat(df_list, ignore_index= True))

