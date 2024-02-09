import pandas as pd
from sqlalchemy import create_engine
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
import joblib

from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.metrics import make_scorer, RocCurveDisplay

from skopt import BayesSearchCV
from skopt.space import Real

from imblearn.metrics import specificity_score

uri = 'postgresql://user:4202@localhost:5432/vinted-ai'
engine = create_engine(uri)

sql_query = "SELECT * FROM public.products_catalog LIMIT 3000"
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
# build a classification algorithm to predict it


# data transformation prior to training
ordinal_encoder = OrdinalEncoder(categories=[["Satisfatório", "Bom", "Muito bom", "Novo sem etiquetas", "Novo com etiquetas"]])
status = pd.DataFrame(ordinal_encoder
                        .fit_transform(data[['status']])
                        ).add_prefix("status_")
                        
status_decoded = ordinal_encoder.get_feature_names_out(input_features= ["status"])

catalog_onehot_encoder = OneHotEncoder(sparse_output=False)
catalog_id = pd.DataFrame(catalog_onehot_encoder
                            .fit_transform(data[['catalog_id']])
                            ).add_prefix("catalog_id_")

#print(pd.DataFrame(size_onehot_encoder.inverse_transform(size_title), columns=['Category']))
catalog_id_decoded = catalog_onehot_encoder.get_feature_names_out(input_features= ["catalog_id"])

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

labels = pd.concat([size_title, status, brand_title, catalog_id], axis=1, ignore_index=False) #.fillna(0)
cols = size_decoded.tolist() + status_decoded.tolist() + brand_decoded.tolist()

model_params = pd.DataFrame([labels.columns, cols]).T
model_params.columns = ["Labels", "Decoded"]
model_params.to_csv("model_development/model_artifacts/params_xgboost.csv")


X_train, X_test, y_train, y_test = train_test_split(labels, data["price"], random_state=42)
"""
import pickle

# Load the XGBoost model from the pickle file
with open("model_development/model_artifacts/xgboost_regression.pkl", "rb") as f:
    model = pickle.load(f)

# Assuming you have new data for prediction stored in a pandas DataFrame called `new_data`
# Make sure `new_data` has the same features as the data used to train the model

# Use the trained model to make predictions on the new data
predictions = model.predict(X_test)
print(pd.concat([predictions, y_test], axis = 1))
"""
# Print or use the predictions as needed

import xgboost as xgb

regressor=xgb.XGBRegressor()

from skopt.space import Real, Categorical, Integer

params={'min_child_weight': Integer(0, 50, "uniform"),
        'max_depth': Integer(1, 10, 'uniform'),
        'subsample': Real(0.5, 1.0, 'log-uniform'),
        'reg_lambda': Real(1e-5, 100, 'log-uniform'),
        'reg_alpha': Real(1e-5, 100, 'log-uniform')
        }

# Specify scoring metrics
# main scoring param has to be defined as "score" since scikit is fetching "mean_test_score"
# local_results = all_results["mean_test_score"][-len(params):]
scoring = {
        "score" : "neg_mean_absolute_percentage_error",
        "mae": "neg_mean_absolute_error",
        "mse" : "neg_mean_squared_error",
        "r2" : "r2",
}

# Perform hyperparameter tuning with BayesSearchCV over 10 folds with AUC as refit metric.
gs_lr = BayesSearchCV(regressor, 
                      params, 
                      cv=10, 
                      scoring=scoring, 
                      refit="score", 
                      random_state=42, 
                      n_iter=10, 
                      return_train_score=True)

# We need to fit the BayesSearchCV object to the train data in order to make predictions with the best model later
gs_lr.fit(X_train, y_train)

# Run nested cross-validation over 10 folds
lr_scores = cross_validate(gs_lr, X_test, y_test, cv=5, n_jobs= 1, verbose=1,
                        return_train_score=True, scoring=scoring)

# Make cross-validated predictions 
lr_preds = cross_val_predict(regressor, X_test, y_test, cv=5, n_jobs=-1)

import numpy as np 
import matplotlib.pyplot as plt

for score in lr_scores:
        print(f"{score:<17}: {np.mean(lr_scores[score]):.2f}")

best_lr_model = gs_lr.best_estimator_
print(best_lr_model)

from xgboost import plot_importance
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
plt.rcParams.update({'font.size': 16})

fig, ax = plt.subplots(figsize=(12,6))
plot_importance(best_lr_model, max_num_features=8, ax=ax)
plt.show()

#coefficients_df = pd.DataFrame({'Feature': cols, 'Coefficient': best_lr_model.coef_[0]})
#coefficients_df = coefficients_df.sort_values(by='Coefficient', ascending=False).head(10)
#print(coefficients_df)

joblib.dump(best_lr_model, 
            'model_development/model_artifacts/xgboost_regression.pkl')

predictions = best_lr_model.predict(X_test)
print(pd.concat([pd.Series(predictions), y_test], axis = 1, ignore_index = True))
"""
f, axs = plt.subplots(1, 1, figsize=(4, 10))
disp = ConfusionMatrixDisplay.from_predictions(y_test, lr_preds,
                                        display_labels=['Not expensive', 'Expensive'],
                                        normalize=None,
                                        ax=axs)

print(classification_report(y_test, lr_preds, target_names=['Not expensive', 'Expensive']))

f, axs = plt.subplots(1, 1, figsize=(5, 4))
roc = RocCurveDisplay.from_estimator(gs_lr.best_estimator_, X_test, y_test, 
                               name="Logistic Regression",
                               ax=axs)



import shap

explainer = shap.Explainer(best_lr_model, 
                                 X_train, 
                                 feature_names=cols)

shap_values = explainer(X_test)
print(shap_values)

shap.plots.beeswarm(shap_values)
shap.summary_plot(shap_values, 
                  X_test)
plt.show()
"""