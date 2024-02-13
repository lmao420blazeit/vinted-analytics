import pandas as pd
from sqlalchemy import create_engine
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
import joblib

from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split, cross_val_predict

from skopt import BayesSearchCV
from skopt.space import Real

uri = 'postgresql://user:4202@localhost:5432/vinted-ai'
engine = create_engine(uri)

sql_query = "SELECT * FROM public.products_catalog WHERE date BETWEEN '2024-02-03' AND '2024-02-09' ORDER BY date DESC LIMIT 3000"
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

labels = pd.concat([size_title, status, brand_title, catalog_id], 
                   axis=1, 
                   ignore_index=True) #.fillna(0)

cols = size_decoded.tolist() + status_decoded.tolist() + brand_decoded.tolist() + catalog_id_decoded.tolist()
model_params = pd.DataFrame([labels.columns, cols]).T
model_params.columns = ["Labels", "Decoded"]
model_params.to_csv("model_development/model_artifacts/params_lightgbm.csv")

X_train, X_test, y_train, y_test = train_test_split(labels, data["price"], random_state=42)

import lightgbm as lgb

regressor = lgb.LGBMRegressor(boosting_type='dart',
                        objective='regression',
                        metric='rmse',
                        n_jobs=1, 
                        verbose=-1,
                        random_state=0)

from skopt.space import Real, Integer

params = {
    'learning_rate': Real(0.01, 1.0, 'log-uniform'),     # Boosting learning rate
    'n_estimators': Integer(30, 2000),                   # Number of boosted trees to fit
    'num_leaves': Integer(2, 128),                       # Maximum tree leaves for base learners
    'max_depth': Integer(-1, 64),                       # Maximum tree depth for base learners, <=0 means no limit
    'subsample': Real(0.1, 1.0, 'uniform'),             # Subsample ratio of the training instance
    'subsample_freq': Integer(1, 10),                    # Frequency of subsample, <=0 means no enable
    'colsample_bytree': Real(0.1, 1.0, 'uniform'),      # Subsample ratio of columns when constructing each tree
    'reg_lambda': Real(0.1, 100.0, 'log-uniform'),      # L2 regularization
    'reg_alpha': Real(0.1, 100.0, 'log-uniform'),       # L1 regularization
   }

# Specify scoring metrics
# main scoring param has to be defined as "score" since scikit is fetching "mean_test_score"
# local_results = all_results["mean_test_score"][-len(params):]
# the scoring is negative because scikit opt transforms this into minimization problem
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

for score in lr_scores:
        print(f"{score:<17}: {np.mean(lr_scores[score]):.2f}")

best_lr_model = gs_lr.best_estimator_

joblib.dump(best_lr_model, 
            'model_development/model_artifacts/lightgbm_regression.pkl')