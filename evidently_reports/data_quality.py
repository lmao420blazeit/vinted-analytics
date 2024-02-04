from evidently.report import Report
from evidently.metric_preset import DataQualityPreset
import pandas as pd
from sqlalchemy import create_engine

uri = 'postgresql://user:4202@localhost:5432/vinted-ai'
engine = create_engine(uri)

sql_query = "SELECT * FROM public.products_catalog ORDER BY date DESC LIMIT 5000"
data = pd.read_sql(sql_query, engine)
data = data[["price", "brand_title", "size_title", "status"]]

boston_data_drift_report = Report(metrics=[DataQualityPreset()])
boston_data_drift_report.run(reference_data=data, 
                             current_data=data[:200])
boston_data_drift_report.save_html("evidently_reports/report.html")