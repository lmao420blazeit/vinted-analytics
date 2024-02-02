import mlflow
import mlflow.xgboost
import pandas as pd
from sqlalchemy import create_engine
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from mlflow.models import infer_signature
import numpy as np

class MLflowXGBoostClassifier:
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
        self.train_model()

    def __compute_metrics(self):
        """
        Compute accuracy and confusion matrix on the test set.
        """
        y_pred = self.model.predict(self.X_test)
        confs = confusion_matrix(self.y_test, y_pred)
        acc = accuracy_score(self.y_test, y_pred)
        return [acc, confs]

    def load_data(self):
        """
        Load data from the 'public.products_catalog' table and select relevant columns.
        """
        sql_query = "SELECT * FROM public.products_catalog"
        df = pd.read_sql(sql_query, self.engine)
        self.data = df[["price", "brand_title", "size_title", "status", "catalog_id"]]

    def preprocess_data(self):
        """
        Preprocess data by encoding categorical variables and creating target and label dataframes.
        """
        ordinal_encoder = OrdinalEncoder(categories=[["Satisfatório", "Bom", "Muito bom", "Novo sem etiquetas", "Novo com etiquetas"]])
        status = pd.DataFrame(ordinal_encoder
                              .fit_transform(self.data[['status']])
                              ).add_prefix("status_")

        onehot_encoder = OneHotEncoder(sparse_output=False)
        size_title = pd.DataFrame(onehot_encoder
                                  .fit_transform(self.data[['size_title']])
                                  ).add_prefix("size_title_")

        self.labels = pd.concat([size_title, status, self.data["price"]], axis=1, ignore_index=False)

        self.ordinal_encoder = OrdinalEncoder(categories="auto")
        self.target = pd.DataFrame(self.ordinal_encoder
                                   .fit_transform(self.data["catalog_id"]
                                                  .values.reshape(-1, 1)))

    def train_model(self):
        """
        Train an XGBoost classifier on the preprocessed data.
        """
        xgb_model = xgb.XGBClassifier(objective="multi:softprob", 
                                      random_state=42)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.labels, self.target, random_state=42)
        xgb_model.fit(self.X_train, self.y_train)
        self.model = xgb_model

    def deploy_model(self, run_name="XGBoost Model"):
        """
        Deploy the trained model and log metrics using MLflow.
        
        Parameters:
            - run_name (str): The name of the MLflow run.
        """
        import plotly.express as px

        mlflow.set_experiment(self.experiment_name)
        acc, confusion = self.__compute_metrics()

        with mlflow.start_run(run_name=run_name):
            mlflow.log_metric("accuracy", 
                              acc)
            mlflow.log_metric("precision_score", 
                              precision_score)
            mlflow.log_metric("f1_score", 
                              f1_score)
            mlflow.log_metric("recall_score", 
                              recall_score)
            
            signature = infer_signature(self.X_train, 
                                        self.model.predict(self.X_train)
                                        )
            
            mlflow.log_params({"objective": "multi:softprob", 
                               "random_state": 42})
            fig = px.imshow(confusion, text_auto=True)
            mlflow.log_figure(fig, "confusion_matrix.html")
            mlflow.xgboost.log_model(self.model, 
                                     "xgboost-model",
                                     signature= signature,
                                     input_example= self.X_train)
            

if __name__ == "__main__":
    # example usage
    uri = 'postgresql://user:4202@localhost:5432/vinted-ai'
    experiment_name = "XGBoost-catalog_classifier"
    mlflow_classifier = MLflowXGBoostClassifier(uri, experiment_name)
    mlflow_classifier.deploy_model()
