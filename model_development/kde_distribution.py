import pandas as pd
from sqlalchemy import create_engine

uri = 'postgresql://user:4202@localhost:5432/vinted-ai'
engine = create_engine(uri)

sql_query = "SELECT price FROM public.products_catalog WHERE date BETWEEN '2024-02-03' AND '2024-02-09' ORDER BY date DESC LIMIT 3000"
data = pd.read_sql(sql_query, engine)