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
data["target"] = data["price"].apply(lambda x: 1 if x > data["price"].quantile(0.85) else 0)
# build a classification algorithm to predict it


# data balancing
def stratified_sampling(group):
    return group.sample(min(len(group), target_sample_size), replace=False)

# Calculate the target sample size (assuming equal sizes for both classes)
target_sample_size = data['target'].value_counts().min()

# Apply the stratified sampling using groupby and apply
data = data.groupby('target', group_keys=False).apply(stratified_sampling)


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
model_params.to_csv("model_development/model_artifacts/params.csv")


X_train, X_test, y_train, y_test = train_test_split(labels, data["target"], random_state=42)

# complexity nf(n)**n_features
lr = LogisticRegression(max_iter=1000, 
                        solver = 'saga')

# parameter space
parameters = {'C': Real(1e-3, 1e+2, prior='log-uniform')}

# Specify scoring metrics
# main scoring param has to be defined as "score" since scikit is fetching "mean_test_score"
scoring = {
        "score" : "roc_auc",
        "specificity": make_scorer(specificity_score, average="weighted"),
        "recall" : "recall",
        "accuracy" : "accuracy",
}

# Perform hyperparameter tuning with BayesSearchCV over 10 folds with AUC as refit metric.
gs_lr = BayesSearchCV(lr, parameters, cv=10, scoring=scoring, 
                      refit="score", random_state=42, n_iter=10, return_train_score=True)

# We need to fit the BayesSearchCV object to the train data in order to make predictions with the best model later
gs_lr.fit(X_train, y_train)

# Run nested cross-validation over 10 folds
lr_scores = cross_validate(gs_lr, X_test, y_test, cv=10, n_jobs= 1, verbose=1,
                        return_train_score=True, scoring=scoring)

# Make cross-validated predictions 
lr_preds = cross_val_predict(lr, X_test, y_test, cv=10, n_jobs=-1)

import numpy as np 
import matplotlib.pyplot as plt

for score in lr_scores:
        print(f"{score:<17}: {np.mean(lr_scores[score]):.2f}")

best_lr_model = gs_lr.best_estimator_
print(best_lr_model)
print(len(best_lr_model.coef_[0]))
print(len(cols))

#coefficients_df = pd.DataFrame({'Feature': cols, 'Coefficient': best_lr_model.coef_[0]})
#coefficients_df = coefficients_df.sort_values(by='Coefficient', ascending=False).head(10)
#print(coefficients_df)
"""
joblib.dump(best_lr_model, 
            'model_development/model_artifacts/logisticregression.pkl')

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

"""

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
