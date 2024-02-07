import mlflow
import pandas as pd
from sqlalchemy import create_engine
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, RocCurveDisplay
from utils.visualizations import *
from sklearn.model_selection import train_test_split
from mlflow.models import infer_signature
from skopt import BayesSearchCV
from skopt.space import Real
from imblearn.metrics import specificity_score

class LogisticRegressor:
    """
    A class to train an XGBoost classifier, log metrics, and deploy the model using MLflow.
    
    Parameters:
        - uri (str): The URI for the database engine.
        - experiment_name (str): The name of the MLflow experiment.
    """

    def __init__(self, uri, experiment_name):
        self.engine = create_engine(uri)
        self.experiment_name = experiment_name
        self.load_data()
        self.preprocess_data()
        self.data_transform()
        self.load_model()      
        mlflow.sklearn.autolog() 

    def load_data(self):
        """
        Load data from the 'public.products_catalog' table and select relevant columns.
        """
        sql_query = f"SELECT * FROM public.products_catalog LIMIT 10000"
        self.data = pd.read_sql(sql_query, self.engine)
        self.data = self.data[["price", "brand_title", "size_title", "status", "catalog_id"]]
        # create target variable
        self.data["target"] = self.data["price"].apply(lambda x: 1 if x > self.data["price"].quantile(0.85) else 0)

    def preprocess_data(self):
        from sklearn import svm

        clf = svm.OneClassSVM(nu=0.1, 
                              kernel="rbf", 
                              gamma=0.1)
        clf.fit(self.data["price"].values.reshape(-1, 1))
        y_pred_train = clf.predict(self.data["price"].values.reshape(-1, 1))
        self.data["svm"] = y_pred_train
        self.data = self.data[self.data["svm"] == 1]

    def __remove_imbalance(self):
        def stratified_sampling(group):
            return group.sample(min(len(group), target_sample_size), replace=False)

        # Calculate the target sample size (assuming equal sizes for both classes)
        target_sample_size = self.data['target'].value_counts().min()

        # Apply the stratified sampling using groupby and apply
        self.data = self.data.groupby('target', group_keys=False).apply(stratified_sampling)        

    def data_transform(self):
        """
        Preprocess data by encoding categorical variables and creating target and label dataframes.
        """
        self.__remove_imbalance()
        ordinal_encoder = OrdinalEncoder(categories=[["Satisfatório", "Bom", "Muito bom", "Novo sem etiquetas", "Novo com etiquetas"]])
        status = pd.DataFrame(ordinal_encoder
                                .fit_transform(self.data[['status']])
                                ).add_prefix("status_")
                                
        status_decoded = ordinal_encoder.get_feature_names_out(input_features= ["status"])

        catalog_onehot_encoder = OneHotEncoder(sparse_output=False)
        catalog_id = pd.DataFrame(catalog_onehot_encoder
                                    .fit_transform(self.data[['catalog_id']])
                                    ).add_prefix("catalog_id_")

        catalog_id_decoded = catalog_onehot_encoder.get_feature_names_out(input_features= ["catalog_id"])

        size_onehot_encoder = OneHotEncoder(sparse_output=False)
        size_title = pd.DataFrame(size_onehot_encoder
                                    .fit_transform(self.data[['size_title']])
                                    ).add_prefix("size_title_")

        size_decoded = size_onehot_encoder.get_feature_names_out(input_features= ["size_title"])

        brand_onehot_encoder = OneHotEncoder(sparse_output=False)
        brand_title = pd.DataFrame(brand_onehot_encoder
                                    .fit_transform(self.data[['brand_title']])
                                    ).add_prefix("brand_title_")

        brand_decoded = brand_onehot_encoder.get_feature_names_out(input_features= ["brand_title"])

        self.labels = pd.concat([size_title, status, brand_title, catalog_id], axis=1, ignore_index=False) #.fillna(0)
        self.cols = size_decoded.tolist() + status_decoded.tolist() + brand_decoded.tolist()

    def load_model(self):
        """
        Train an XGBoost classifier on the preprocessed data.
        """

        model_params = pd.DataFrame([self.labels.columns, self.cols]).T
        model_params.columns = ["Labels", "Decoded"]
        model_params.to_csv("model_development/model_artifacts/params.csv")

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.labels, self.data["target"], random_state=42)

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
        gs_lr = BayesSearchCV(lr, 
                              parameters, 
                              cv=10, 
                              scoring=scoring, 
                                refit="score", 
                                random_state=42, 
                                n_iter=10, 
                                return_train_score=True)

        gs_lr.fit(self.X_train, self.y_train)

        # Run nested cross-validation over 10 folds
        self.lr_scores = cross_validate(gs_lr, 
                                        self.X_test, 
                                        self.y_test, 
                                        cv=10, 
                                        n_jobs= 1, 
                                        verbose=1,
                                        return_train_score=True, 
                                        scoring=scoring)

        # Make cross-validated predictions 
        self.lr_preds = cross_val_predict(lr, self.X_test, self.y_test, cv=10, n_jobs=-1)
        self.model = gs_lr.best_estimator_

    def deploy_model(self, run_name="LogisticRegressor"):
        """
        Deploy the trained model and log metrics using MLflow.
        
        Parameters:
            - run_name (str): The name of the MLflow run.
        """
        with mlflow.start_run(run_name=run_name):
            mlflow.set_experiment = self.experiment_name
            #mlflow.log_metrics(self.lr_scores)
            #mlflow.log_params(self.model.coef_)
            #mlflow.log_dict(self.model.coefficients_df.to_dict("records"), "coefficients.json")
            print(self.lr_scores)
            print(self.model.coef_)
            infer_signature(self.X_test,
                            self.y_test)
            args = dict(model = self.model, 
                        testX = self.X_test, 
                        testY = self.y_test, 
                        name = self.__class__.__name__, 
                        log_to_mlflow= True)
            plot_ROC_AUC_curve(**args)
            plot_confusion_matrix(trueY= self.y_test, 
                                  predictions = self.lr_preds, 
                                  log_to_mlflow= True)
            plot_cv_scores(self.lr_scores, 
                           log_to_mlflow= True)
    """            
    def get_features(self, num_features = 10):
        status_decoded = self.ordinal_encoder.get_feature_names_out(input_features= ["status"])
        size_decoded = self.size_onehot_encoder.get_feature_names_out(input_features= ["size_title"])
        brand_decoded = self.brand_onehot_encoder.get_feature_names_out(input_features= ["brand_title"])
        cols = size_decoded.tolist() + status_decoded.tolist() + brand_decoded.tolist()

        feature_importance = pd.DataFrame({'Feature': cols, 
                                        'Coefficient': self.model.coef_[0]})

        feature_importance = feature_importance.reindex(feature_importance['Coefficient'].abs().sort_values(ascending=False).index)
        return (feature_importance[:num_features])
    """

if __name__ == "__main__":
    # example usage
    uri = 'postgresql://user:4202@localhost:5432/vinted-ai'
    experiment_name = "LogisticRegressor-expensiveness-classification"
    mlflow_classifier = LogisticRegressor(uri, experiment_name)
    mlflow_classifier.deploy_model()
