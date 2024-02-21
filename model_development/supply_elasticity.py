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
            brand_title as b,
            catalog_id as c,
            COUNT(*) AS sample_count
        FROM products_catalog
        WHERE date >= CURRENT_DATE - INTERVAL '15 days'
        GROUP BY 
            brand_title, 
            catalog_id
    )
    SELECT subquery.*
    FROM (
        SELECT 
            brand_title,
            catalog_id,
            price,
            product_id,
            ROW_NUMBER() OVER (PARTITION BY t.brand_title, t.catalog_id) AS row_num
        FROM products_catalog t
        JOIN Counts c 
            ON t.brand_title = c.b 
            AND t.catalog_id = c.c
        WHERE c.sample_count >= 50 
    ) AS subquery
    WHERE row_num <= 50 ;
    """

    data = pd.read_sql(sql_query, engine)
    #print(data.groupby("catalog_id")["product_id"].count())
    return data

data = load_data()
bins = np.linspace(0, 50, 50)
data = data.pivot_table(index='catalog_id', columns=pd.cut(data['price'], bins), aggfunc='size')
print(data.T)