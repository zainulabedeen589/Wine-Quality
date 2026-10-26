import os
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn

from pathlib import Path
from urllib.parse import urlparse

from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score)

from src.datascience.constants import *
from src.datascience.utils.common import (read_yaml, create_directories, save_json)
from src.datascience.entity.config_entity import ModelEvaluationConfig

class ModelEvaluation:

    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual, pred):

        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)

        return rmse, mae, r2

    def log_into_mlflow(self):

        # Load test dataset
        test_data = pd.read_csv(
            self.config.test_data_path
        )

        # Load trained model
        model = joblib.load(
            self.config.model_path
        )

        # Separate features and target
        test_x = test_data.drop(
            [self.config.target_column],
            axis=1
        )

        test_y = test_data[self.config.target_column]

        # Configure MLflow
        mlflow.set_tracking_uri(
            self.config.mlflow_uri
        )

        mlflow.set_registry_uri(
            self.config.mlflow_uri
        )

        tracking_url_type_store = urlparse(
            mlflow.get_tracking_uri()
        ).scheme

        # Evaluate and log results
        with mlflow.start_run():

            predictions = model.predict(test_x)

            rmse, mae, r2 = self.eval_metrics(
                test_y,
                predictions
            )

            # Print evaluation metrics
            print(f"RMSE: {rmse}")
            print(f"MAE: {mae}")
            print(f"R2 Score: {r2}")

            # Save metrics locally
            scores = {
                "rmse": float(rmse),
                "mae": float(mae),
                "r2": float(r2)
            }

            metrics_path = Path(
                self.config.metrics_file_name
            )

            create_directories([metrics_path.parent])

            save_json(
                path=metrics_path,
                data=scores
            )

            # Log parameters
            mlflow.log_params(
                dict(self.config.all_params)
            )

            # Log metrics
            mlflow.log_metric("rmse", float(rmse))
            mlflow.log_metric("mae", float(mae))
            mlflow.log_metric("r2", float(r2))

            # Log model
            if tracking_url_type_store != "file":

                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    registered_model_name="ElasticnetModel"
                )

            else:

                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model"
                )

            print("Model evaluation completed successfully.")