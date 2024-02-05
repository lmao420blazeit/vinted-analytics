import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import numpy as np

import statsmodels.api as sm
from statsmodels.formula.api import ols
import umap
import umap.plot

# a case study of sparse high dimensional data
uri = 'postgresql://user:4202@localhost:5432/vinted-ai'
engine = create_engine(uri)

#sql_query = "SELECT price, status, brand_title, catalog_id, size_title FROM public.products_catalog LIMIT 10000"
sql_query = "SELECT price, brand_title, catalog_id FROM public.products_catalog LIMIT 10000"
data = pd.read_sql(sql_query, engine)

# select top 50 brands and reduce dimensionality
brands = data.groupby(["brand_title"])["price"].count().sort_values(ascending = False).head(50).reset_index()["brand_title"] #["brand_title", "catalog_id"]
data = data[data["brand_title"].isin(brands)]

data["catalog_id"] = data["catalog_id"].astype("object")
categorical = data.select_dtypes(include='object')
categorical = pd.get_dummies(categorical, dtype=float)

data = pd.concat([categorical, data["price"]],
                ignore_index= True,
                axis = 1)



from sklearn.decomposition import PCA
pca = PCA(n_components=3)
pca.fit(data)
components = pca.fit_transform(data)

total_var = pca.explained_variance_ratio_.sum() * 100
print(len(data.index))

import plotly.express as px

fig = px.scatter_3d(
    components, 
    x=0, 
    y=1, 
    z=2,
    labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'},
    title=f'Total Explained Variance: {total_var:.2f}%',
)
fig.show()