import mlflow
import pandas as pd
from sqlalchemy import create_engine
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from mlflow.models import infer_signature

class LogisticRegressor:
    """
    A class to train an XGBoost classifier, log metrics, and deploy the model using MLflow.
    
    Parameters:
        - uri (str): The URI for the database engine.
        - experiment_name (str): The name of the MLflow experiment.
    """

    def __init__(self, uri, experiment_name, catalog_id):
        self.engine = create_engine(uri)
        self.experiment_name = experiment_name
        self.catalog_id = catalog_id
        self.load_data()
        self.preprocess_data()
        self.data_transform()
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
        sql_query = f"SELECT * FROM public.products_catalog WHERE catalog_id = {int(self.catalog_id)} LIMIT 5000"
        self.data = pd.read_sql(sql_query, self.engine)
        self.data = self.data[["price", "brand_title", "size_title", "status"]]

    def preprocess_data(self):
        from sklearn import svm

        clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
        clf.fit(self.data["price"].values.reshape(-1, 1))
        y_pred_train = clf.predict(self.data["price"].values.reshape(-1, 1))
        self.data["svm"] = y_pred_train
        self.data = self.data[self.data["svm"] == 1]        

    def data_transform(self):
        """
        Preprocess data by encoding categorical variables and creating target and label dataframes.
        """
        self.ordinal_encoder = OrdinalEncoder(categories=[["Satisfatório", "Bom", "Muito bom", "Novo sem etiquetas", "Novo com etiquetas"]])
        status = pd.DataFrame(self.ordinal_encoder
                                .fit_transform(self.data[['status']])
                                ).add_prefix("status_")

        size_onehot_encoder = OneHotEncoder(sparse_output=False)
        size_title = pd.DataFrame(size_onehot_encoder
                                    .fit_transform(self.data[['size_title']])
                                    ).add_prefix("size_title_")

        self.brand_onehot_encoder = OneHotEncoder(sparse_output=False)
        brand_title = pd.DataFrame(self.brand_onehot_encoder
                                    .fit_transform(self.data[['brand_title']])
                                    ).add_prefix("brand_title_")

        self.labels = pd.concat([size_title, status, brand_title], axis=1, ignore_index=False) #.fillna(0)

        self.data["target"] = self.data["price"].apply(lambda x: 1 if x > self.data["price"].quantile(0.85) else 0)

    def train_model(self):
        """
        Train an XGBoost classifier on the preprocessed data.
        """
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(penalty='l2', 
                                 solver='newton-cholesky')
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.labels, self.data["target"], random_state=42)
        clf.fit(self.X_train, self.y_train)
        self.model = clf

    def deploy_model(self, run_name="LogisticRegressor"):
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
            
            mlflow.log_params({"penalty": "l2", 
                               "solver": "newton-cholesky"})
            fig = px.imshow(confusion, text_auto=True)
            mlflow.log_figure(fig, "confusion_matrix.html")
            mlflow.sklearn.log_model(self.model, 
                                     "logisticregressor",
                                     signature= signature,
                                     input_example= self.X_train)
            
    def get_features(self, num_features = 10):
        status_decoded = self.ordinal_encoder.get_feature_names_out(input_features= ["status"])
        size_decoded = self.size_onehot_encoder.get_feature_names_out(input_features= ["size_title"])
        brand_decoded = self.brand_onehot_encoder.get_feature_names_out(input_features= ["brand_title"])
        cols = size_decoded.tolist() + status_decoded.tolist() + brand_decoded.tolist()

        feature_importance = pd.DataFrame({'Feature': cols, 
                                        'Coefficient': self.model.coef_[0]})

        feature_importance = feature_importance.reindex(feature_importance['Coefficient'].abs().sort_values(ascending=False).index)
        return (feature_importance[:num_features])

if __name__ == "__main__":
    # example usage
    uri = 'postgresql://user:4202@localhost:5432/vinted-ai'
    experiment_name = "LogisticRegressor-feature-ranking"
    mlflow_classifier = LogisticRegressor(uri, experiment_name, 221)
    mlflow_classifier.deploy_model()
