import pandas as pd
from sqlalchemy import create_engine
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

uri = 'postgresql://user:4202@localhost:5432/vinted-ai'
engine = create_engine(uri)

sql_query = "SELECT * FROM public.products_catalog WHERE catalog_id = 76 LIMIT 5000"
data = pd.read_sql(sql_query, engine)

from sklearn import svm

clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
clf.fit(data["price"].values.reshape(-1, 1))
y_pred_train = clf.predict(data["price"].values.reshape(-1, 1))
data["svm"] = y_pred_train
data = data[data["svm"] == 1]

#data = data[data["catalog_id"] == 221]
# detected outliers
# pick outliers
# for each catalog_id
# percentile tag outliers (0 if below Q3, 1 if above) (has to be binary)
# this is an instance of an imbalance dataset
data["target"] = data["price"].apply(lambda x: 1 if x > data["price"].quantile(0.85) else 0)
# build a classification algorithm to predict it

ordinal_encoder = OrdinalEncoder(categories=[["Satisfatório", "Bom", "Muito bom", "Novo sem etiquetas", "Novo com etiquetas"]])
status = pd.DataFrame(ordinal_encoder
                        .fit_transform(data[['status']])
                        ).add_prefix("status_")
                        
status_decoded = ordinal_encoder.get_feature_names_out(input_features= ["status"])

size_onehot_encoder = OneHotEncoder(sparse_output=False)
size_title = pd.DataFrame(size_onehot_encoder
                            .fit_transform(data[['size_title']])
                            ).add_prefix("size_title_")

#print(pd.DataFrame(size_onehot_encoder.inverse_transform(size_title), columns=['Category']))
size_decoded = size_onehot_encoder.get_feature_names_out(input_features= ["size_title"])

brand_onehot_encoder = OneHotEncoder(sparse_output=False)
brand_title = pd.DataFrame(brand_onehot_encoder
                            .fit_transform(data[['brand_title']])
                            ).add_prefix("brand_title_")

brand_decoded = brand_onehot_encoder.get_feature_names_out(input_features= ["brand_title"])

#print(pd.DataFrame(brand_onehot_encoder.inverse_transform(brand_title), columns=['Category']))

labels = pd.concat([size_title, status, brand_title], axis=1, ignore_index=False) #.fillna(0)
cols = size_decoded.tolist() + status_decoded.tolist() + brand_decoded.tolist()

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(penalty='l2', solver='newton-cholesky')
clf.fit(labels, data["target"])
y_pred = clf.predict(labels)

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, mean_squared_error, f1_score

f1 = f1_score(data["target"], 
              y_pred)

acc = accuracy_score(data["target"], 
              y_pred)
# true positive rate
# TP/Predicted positives
rec = recall_score(data["target"], 
              y_pred)

prec = precision_score(data["target"], 
              y_pred)

cm = confusion_matrix(data["target"], 
                      y_pred)

feature_importance = pd.DataFrame({'Feature': cols, 
                                   'Coefficient': clf.coef_[0]})

feature_importance = feature_importance.reindex(feature_importance['Coefficient'].abs().sort_values(ascending=False).index)
print(feature_importance[:20])
print(f1, cm, acc, prec)

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(labels, data["price"])
y_pred = model.predict(labels)

mse = mean_squared_error(data["price"], y_pred)
data["y_pred"] = y_pred
print(data[data["target"] == 1])
print(f'Mean Squared Error: {mse}')
