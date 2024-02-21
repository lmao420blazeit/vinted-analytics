import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.neighbors import KernelDensity
import numpy as np
import seaborn as sns

# https://www.kaggle.com/code/yuqizheng/intro-to-kernel-density-estimation-kde
# intro to KDE

uri = 'postgresql://user:4202@localhost:5432/vinted-ai'
engine = create_engine(uri)

# https://stats.stackexchange.com/questions/76948/what-is-the-minimum-number-of-data-points-required-for-kernel-density-estimation
# minimum sample size for KDE
# DIMENSIONALITY = 2 -> 19
# DIMENSIONALITY = 3 -> 67

def load_data():
    """
    Load data from the 'products_catalog' table in the PostgreSQL database for the last 7 days.
    Returns:
        DataFrame: Data loaded from the table.
    """
    sql_query = """
    WITH Counts AS (
        SELECT
            status as b,
            catalog_id as c,
            COUNT(*) AS sample_count
        FROM products_catalog
        WHERE date >= CURRENT_DATE - INTERVAL '7 days'
        GROUP BY 
            status, 
            catalog_id
    )
    SELECT subquery.*
    FROM (
        SELECT 
            status,
            catalog_id,
            price,
            product_id,
            ROW_NUMBER() OVER (PARTITION BY t.status, t.catalog_id) AS row_num
        FROM products_catalog t
        JOIN Counts c 
            ON t.status = c.b 
            AND t.catalog_id = c.c
        WHERE c.sample_count >= 20 
    ) AS subquery
    WHERE row_num <= 20 ;
    """

    data = pd.read_sql(sql_query, engine)
    #print(data.groupby("catalog_id")["product_id"].count())
    return data

def fit_kde(X):
    """
    Fit a Kernel Density Estimation (KDE) model to the given data.
    Args:
        X (array-like): Input data.
    Returns:
        estimator: Fitted KDE model.
    """
    bandwidths = np.linspace(1, 3, 5)
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
    """
    Remove the outer list from each cell in a DataFrame.
    Args:
        cell (list): Cell value containing a list.
    Returns:
        object: Value extracted from the list.
    """
    return cell[0]

def compute_statistics(samples):
    """
    Compute statistics (quartiles, skewness, and kurtosis) for the given samples.
    Args:
        samples (array-like): Samples for which statistics are to be computed.
    Returns:
        array-like: Computed statistics.
    """
    quartiles_probabilities = np.percentile(samples, [25, 50, 75])
    skewness = skew(samples)
    kurt = kurtosis(samples)
    return np.append(quartiles_probabilities, [skewness, kurt])

def kde_brands():
    """
    Perform Kernel Density Estimation (KDE) for each brand in the loaded data.
    Returns:
        DataFrame: Results of KDE for each brand.
    """
    data = load_data()
    unique_catalog_ids = data["catalog_id"].unique()
    sample_list = []
    data_list = []
    catalog_list = []
    brand_list = []
    sample_size = []
    bandwidth_list = []

    for catalog_id in unique_catalog_ids:

        catalog_data = data[data["catalog_id"] == catalog_id]
        unique_status = catalog_data["status"].unique()

        for status in unique_status:

            brand_data = catalog_data[catalog_data["status"] == status]
            percentile_85 = brand_data['price'].quantile(0.85)
            brand_data = brand_data[brand_data["price"] <= percentile_85]
            

            X_plot = np.linspace(0, 80, num=500)[:, np.newaxis]
            X = brand_data["price"].values

            # fit and compile model
            model = fit_kde(X)
            log_dens = model.score_samples(X_plot)
            samples = model.sample(n_samples=100)
            

            #plot_kde(X_plot, log_dens, X, catalog_id)

            sample_list.append(samples.tolist())
            data_list.append(compute_statistics(samples))
            catalog_list.append(catalog_id)
            brand_list.append(status)
            sample_size.append(len(X))
            bandwidth_list.append(model.bandwidth_)
            
    df = pd.DataFrame(sample_list, index=[brand_list, catalog_list])
    df = df.T
    df = df.map(remove_list)  # Removing lists
    dt_list = pd.DataFrame(data_list, 
                           columns = ["Q1", "Q2", "Q3", "Skew", "Kurt"])
    dt_list["status"] = brand_list
    dt_list["catalog_id"] = catalog_list
    dt_list["sample_size"] = sample_size
    dt_list["bandwidth"] = bandwidth_list
    medians = dt_list.groupby('catalog_id')['Q2'].median()

    dt_list = pd.merge(dt_list, 
                       medians.rename('median_Q2'), 
                       left_on='catalog_id', 
                       right_on='catalog_id',
                       right_index=True)
    
    dt_list['status_premium'] = (dt_list['Q2'] - dt_list['median_Q2'])/(dt_list['median_Q2'] + dt_list['Q2'])
    dt_list = dt_list.drop("median_Q2", axis = 1)

    return(dt_list)

if __name__ == "__main__":
    data = kde_brands()
    pivot_table = pd.pivot_table(data, values='status_premium', index='catalog_id', columns='status', aggfunc='sum', fill_value=0)
    pivot_table.index = pivot_table.index.astype(str)
    # reorder
    pivot_table = pivot_table[["Satisfatório", "Bom", "Muito bom", "Novo sem etiquetas", "Novo com etiquetas"]]

    import plotly.express as px

    fig = px.imshow(pivot_table, 
                    text_auto=True)
    fig.show()
    pivot_table = pd.pivot_table(data, values='Q2', index='catalog_id', columns='status', aggfunc='sum', fill_value=0)
    pivot_table.index = pivot_table.index.astype(str)
    pivot_table = pivot_table[["Satisfatório", "Bom", "Muito bom", "Novo sem etiquetas", "Novo com etiquetas"]]
    fig = px.imshow(pivot_table, 
                    text_auto=True)
    fig.show()

