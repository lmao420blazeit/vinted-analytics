import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt

uri = 'postgresql://user:4202@localhost:5432/vinted-ai'
engine = create_engine(uri)

# https://stats.stackexchange.com/questions/76948/what-is-the-minimum-number-of-data-points-required-for-kernel-density-estimation
# minimum sample size for KDE
# DIMENSIONALITY = 2 -> 19
# DIMENSIONALITY = 3 -> 67

sql_query = """
SELECT
  product_id,
  catalog_id,
  brand_title,
  price
FROM
  products_catalog
WHERE brand_title IN (SELECT 
    brand_title
    FROM products_catalog
    GROUP BY brand_title, catalog_id
    HAVING COUNT (*) > 50
)"""
sql_query = """
with sample as (SELECT * FROM products_catalog LIMIT 20000)
, 
subquery as (SELECT 
    brand_title, catalog_id, count(*)
FROM sample
GROUP BY brand_title, catalog_id
HAVING COUNT (product_id) > 50
ORDER BY brand_title)

SELECT product_id, catalog_id, brand_title, price
FROM sample
WHERE (brand_title, catalog_id) IN (SELECT brand_title, catalog_id FROM subquery)
"""

data = pd.read_sql(sql_query, engine)
print(data)


# https://www.kaggle.com/code/yuqizheng/intro-to-kernel-density-estimation-kde
# intro to KDE

from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.neighbors import KernelDensity
import numpy as np
import seaborn as sns

# Get unique catalog_ids
unique_catalog_ids = data["catalog_id"].unique()

# Determine the number of subplots based on the number of unique catalog_ids
num_subplots = len(unique_catalog_ids)

X = data["price"].values

bandwidths = np.linspace(1, 3, 3)
grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                    {'bandwidth': bandwidths},
                    cv=LeaveOneOut()
                    )

fig, axes = plt.subplots(num_subplots, 1, 
                         figsize=(20, num_subplots*20), 
                         constrained_layout=True)

for idx, catalog_id in enumerate(unique_catalog_ids):

    catalog_data = data[data["catalog_id"] == catalog_id]
    percentile_85 = data['price'].quantile(0.85)
    catalog_data = catalog_data[catalog_data["price"] <= percentile_85]

    X_plot = np.linspace(0, 80, num=500)[:, np.newaxis]

    """
    with sns.axes_style('white'):
        sns.jointplot("price", "status_encoded", data = data, kind='kde')
    """
    for brand_title in catalog_data["brand_title"].unique():
        brand_title_data = catalog_data[catalog_data["brand_title"] == brand_title]
        print(len(brand_title_data))
        X = brand_title_data["price"][:50].values

        grid.fit(X[:, None])
        model = grid.best_estimator_
        log_dens = grid.best_estimator_.score_samples(X_plot)

        axes[idx].plot(X_plot[:, 0], 
                np.exp(log_dens), 
                label=f'KDE {brand_title}')
    
        axes[idx].hist(X, 
                bins=30, 
                density=True, 
                alpha=0.5, 
                label=f'Brand: {brand_title}')  # Plot histogram for comparison
        
    #print(grid.best_params_)
    axes[idx].set_xlabel('Value')
    axes[idx].set_ylabel('Density')
    axes[idx].set_title(f'Catalog {catalog_id}')
    axes[idx].legend()

    cdf = np.cumsum(np.exp(log_dens))

    # Compute quartiles probabilities
    quartiles_probabilities = np.percentile(cdf, [25, 50, 75])
    quartiles2 = np.percentile(X, [25, 50, 75])
    #quartiles_x_values = np.interp(quartiles_probabilities, cdf, np.linspace(0, 80, num=500))

    # Convert log-density estimates back to densities
    #quartiles_densities = np.exp(quartiles_probabilities)
    print(brand_title, catalog_id, quartiles_probabilities, quartiles2)

#plt.tight_layout()
plt.show()

