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
brands = data.groupby(["brand_title"])["price"].count().sort_values(ascending = False).head(50).reset_index()["brand_title"]
data = data[data["brand_title"].isin(brands)]

data["catalog_id"] = data["catalog_id"].astype("object")
categorical = data.select_dtypes(include='object')
categorical = pd.get_dummies(categorical, dtype=float)

#Embedding numerical & categorical
fit1 = umap.UMAP(metric='l2').fit(data["price"].values.reshape(-1, 1))
umap.plot.points(fit1, 
                 width=1000, 
                 height=1000, 
                 labels = data.catalog_id)
#plt.show()
fit2 = umap.UMAP(metric='dice').fit(categorical)



#Augmenting the numerical embedding with categorical
intersection = umap.umap_.general_simplicial_set_intersection(fit1.graph_, fit2.graph_)
intersection = umap.umap_.reset_local_connectivity(intersection)
embedding, _ = umap.umap_.simplicial_set_embedding(data = fit1._raw_data, 
                                                graph = intersection, 
                                                n_components = fit1.n_components, 
                                                initial_alpha=fit1._initial_alpha, 
                                                a = fit1._a, 
                                                b = fit1._b, 
                                                gamma = fit1.repulsion_strength, 
                                                negative_sample_rate= fit1.negative_sample_rate, 
                                                n_epochs = 200, 
                                                init = 'random', 
                                                random_state = np.random, 
                                                metric = fit1.metric, 
                                                metric_kwds= fit1._metric_kwds,
                                                densmap= fit1.densmap,
                                                densmap_kwds=fit1._densmap_kwds, 
                                                output_dens=False)

plt.figure(figsize=(20, 10))
plt.scatter(*embedding.T, s=2, cmap='Spectral', alpha=1.0)
plt.show()