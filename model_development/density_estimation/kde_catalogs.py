import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis


uri = 'postgresql://user:4202@localhost:5432/vinted-ai'
engine = create_engine(uri)

# https://stats.stackexchange.com/questions/76948/what-is-the-minimum-number-of-data-points-required-for-kernel-density-estimation
# minimum sample size for KDE
# DIMENSIONALITY = 2 -> 19
# DIMENSIONALITY = 3 -> 67

def load_data():
    
    # this doesnt work well
    sql_query = """
    WITH Counts AS (
        SELECT
            catalog_id as c,
            COUNT(*) AS sample_count
        FROM products_catalog
        WHERE date >= CURRENT_DATE - INTERVAL '7 days'
        GROUP BY 
            catalog_id
    )
    SELECT subquery.*
    FROM (
        SELECT 
            catalog_id,
            price,
            product_id,
            ROW_NUMBER() OVER (PARTITION BY t.catalog_id) AS row_num
        FROM products_catalog t
        JOIN Counts c 
            ON t.catalog_id = c.c
        WHERE c.sample_count >= 30 
    ) AS subquery
    WHERE row_num <= 30 ;
    """

    data = pd.read_sql(sql_query, engine)
    #print(data.groupby("catalog_id")["product_id"].count())
    return data

def fit_kde(X):
    bandwidths = np.linspace(1, 3, 3)
    grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                        {'bandwidth': bandwidths},
                        cv=LeaveOneOut()
                        )
    grid.fit(X[:, None])
    return grid.best_estimator_

def plot_kde(X_plot, log_dens, X, catalog_id):
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.plot(X_plot[:, 0], 
            np.exp(log_dens), 
            label=f'KDE {catalog_id}')
    ax.hist(X, 
            bins=30, 
            density=True, 
            alpha=0.5, 
            label=f'Real')  
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.set_title(f'Catalog {catalog_id}')
    ax.legend()


# this is such a bad practice
def remove_list(cell):
    return cell[0]

def compute_statistics(samples):
    quartiles_probabilities = np.percentile(samples, [25, 50, 75])
    skewness = skew(samples)
    kurt = kurtosis(samples)
    return np.append(quartiles_probabilities, [skewness, kurt])

# https://www.kaggle.com/code/yuqizheng/intro-to-kernel-density-estimation-kde
# intro to KDE

from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.neighbors import KernelDensity
import numpy as np
import seaborn as sns

def main():
    data = load_data()
    unique_catalog_ids = data["catalog_id"].unique()
    sample_list = []
    data_list = []

    for catalog_id in unique_catalog_ids:

        catalog_data = data[data["catalog_id"] == catalog_id]
        percentile_85 = data['price'].quantile(0.85)
        catalog_data = catalog_data[catalog_data["price"] <= percentile_85]

        X_plot = np.linspace(0, 80, num=500)[:, np.newaxis]
        X = catalog_data["price"][:200].values

        # fit and compile model
        model = fit_kde(X)
        log_dens = model.score_samples(X_plot)
        samples = model.sample(n_samples=100)

        #plot_kde(X_plot, log_dens, X, catalog_id)

        sample_list.append(samples.tolist())
        data_list.append(compute_statistics(samples))

    df = pd.DataFrame(sample_list, index=unique_catalog_ids)
    df = df.T
    df = df.map(remove_list)  # Removing lists
    dt_list = pd.DataFrame(data_list, columns = ["Q1", "Q2", "Q3", "Skew", "Kurt"], index = unique_catalog_ids)
    print(dt_list)


main()

