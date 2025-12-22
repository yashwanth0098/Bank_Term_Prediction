# import os
# import sys
# from typing import Dict

# import numpy as np
# from sklearn.metrics import recall_score, precision_score, classification_report
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier

# from source_main.exception.exception import BankException
# from source_main.logging.logging import logging
# from source_main.entity.config import ModelTrainerConfig
# from source_main.entity.artifact import (
#     ModelTrainerArtifact,
#     DataTransformationArtifact,
# )
# from source_main.utlis.main_utlis.utlis import (
#     load_object,
#     save_object,
#     evaluate_model,
#     evaluate_thresholds,
# )


# class ModelTrainer:
#     """
#     Model Trainer Component
#     -----------------------
#     Responsibilities:
#     - Load transformed data
#     - Train multiple models
#     - Select best model based on recall
#     - Tune probability threshold
#     - Validate business constraints
#     - Save final model
#     """

#     def __init__(
#         self,
#         model_trainer_config: ModelTrainerConfig,
#         data_transformation_artifact: DataTransformationArtifact,
#     ):
#         self.model_trainer_config = model_trainer_config
#         self.data_transformation_artifact = data_transformation_artifact


#     def initiate_model_trainer(self) -> ModelTrainerArtifact:
#         try:
#             logging.info("========== MODEL TRAINER STARTED ==========")

#             X_train = np.load(
#                 self.data_transformation_artifact.X_train_file_path,allow_pickle=True
#             )
#             y_train = np.load(
#                 self.data_transformation_artifact.y_train_file_path,allow_pickle=True
#             ).ravel()
#             X_test = np.load(
#                 self.data_transformation_artifact.X_test_file_path,allow_pickle=True
#             )
#             y_test = np.load(
#                 self.data_transformation_artifact.y_test_file_path,allow_pickle=True
#             ).ravel()

#             artifact = self.train_model(
#                 X_train=X_train,
#                 y_train=y_train,
#                 X_test=X_test,
#                 y_test=y_test,
#             )

#             logging.info("========== MODEL TRAINER COMPLETED ==========")
#             return artifact

#         except Exception as e:
#             logging.error("Model trainer failed")
#             raise BankException(e, sys)


#     def train_model(
#         self,
#         X_train,
#         y_train,
#         X_test,
#         y_test,
#     ) -> ModelTrainerArtifact:

#         try:
#             logging.info("Loading preprocessor object")
#             preprocessor = load_object(
#                 self.data_transformation_artifact.preprocessor_object_file_path
#             )

#             logging.info("Initializing models and hyperparameters")
#             models = {
#                 "xgboost": XGBClassifier(
#                     random_state=42,
#                     eval_metric="logloss",
#                 ),
#                 "random_forest": RandomForestClassifier(random_state=42),
#                 "decision_tree": DecisionTreeClassifier(random_state=42),
#             }

#             params = {
#                 "xgboost": {
#                     "n_estimators": [100, 300],
#                     "learning_rate": [0.05, 0.1],
#                     "max_depth": [3, 5],
#                 },
#                 "random_forest": {
#                     "n_estimators": [100, 300],
#                     "max_depth": [5, 10, None],
#                     "min_samples_split": [2, 5],
#                 },
#                 "decision_tree": {
#                     "max_depth": [5, 10, None],
#                     "min_samples_split": [2, 5],
#                 },
#             }


#             logging.info("Evaluating models using recall metric")
#             model_report: Dict = evaluate_model(
#                 X_train=X_train,
#                 y_train=y_train,
#                 X_test=X_test,
#                 y_test=y_test,
#                 preprocessor=preprocessor,
#                 models=models,
#                 params=params,
#             )

#             best_model_name = max(
#                 model_report,
#                 key=lambda name: model_report[name]["score"],
#             )

#             model_pipeline = model_report[best_model_name]["best_estimator"]
#             logging.info(f"Best model selected: {best_model_name}")


#             y_train_proba = model_pipeline.predict_proba(X_train)[:, 1]
#             y_test_proba = model_pipeline.predict_proba(X_test)[:, 1]


#             logging.info("Performing threshold tuning")
#             threshold_results = evaluate_thresholds(
#                 y_true=y_test,
#                 y_pred_proba=y_test_proba,
#                 min_threshold=self.model_trainer_config.min_probability_threshold,
#                 max_threshold=self.model_trainer_config.max_probability_threshold,
#                 step=self.model_trainer_config.threshold_step,
#             )

#             valid_thresholds = [
#                 t for t in threshold_results
#                 if t["precision"] >= self.model_trainer_config.precision_score
#                 and t["false_positive_rate"]
#                 <= self.model_trainer_config.false_positive_threshold
#             ]

#             if not valid_thresholds:
#                 raise ValueError("No threshold satisfies business constraints")

#             best_threshold = max(
#                 valid_thresholds,
#                 key=lambda t: t["recall"],
#             )["threshold"]

#             logging.info(f"Best threshold selected: {best_threshold}")


#             y_test_pred = (y_test_proba >= best_threshold).astype(int)

#             test_recall = recall_score(y_test, y_test_pred)
#             test_precision = precision_score(
#                 y_test, y_test_pred, zero_division=0
#             )

#             is_model_accepted = (
#                 test_recall >= self.model_trainer_config.expected_recall_score
#             )

#             # -------------------------------------------------
#             # Save Model
#             # -------------------------------------------------
#             logging.info("Saving trained model")
#             save_object(
#                 self.model_trainer_config.trained_model_file_path,
#                 obj=model_pipeline,
#             )

#             return ModelTrainerArtifact(
#                 is_model_accepted=is_model_accepted,
#                 trained_model_file_path=self.model_trainer_config.trained_model_file_path,
#                 train_recall=recall_score(
#                     y_train,
#                     (y_train_proba >= best_threshold).astype(int),
#                 ),
#                 test_recall=test_recall,
#                 train_precision=precision_score(
#                     y_train,
#                     (y_train_proba >= best_threshold).astype(int),
#                     zero_division=0,
#                 ),
#                 test_precision=test_precision,
#                 base_recall=self.model_trainer_config.base_recall_score,
#                 expected_recall=self.model_trainer_config.expected_recall_score,
#                 best_probability_threshold=best_threshold,
#                 classification_report=classification_report(
#                     y_test,
#                     y_test_pred,
#                     output_dict=True,
#                 ),
#             )

#         except Exception as e:
#             logging.error("Model training failed")
#             raise BankException(e, sys)


## Above code is working without mlflow and dagshub



import sys
import os
from typing import Dict

import numpy as np
import joblib
import mlflow
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.metrics import recall_score, precision_score, classification_report
from xgboost import XGBClassifier

from source_main.exception.exception import BankException
from source_main.logging.logging import logging
from source_main.entity.config import ModelTrainerConfig
from source_main.entity.artifact import (
    ModelTrainerArtifact,
    DataTransformationArtifact,
)
from source_main.utlis.main_utlis.utlis import (
    load_object,
    evaluate_model,
    evaluate_thresholds,
)
from source_main.components.mlflow__init__ import init_mlflow


class ModelTrainer:
    """
    Model Trainer Component
    -----------------------
    - Trains models
    - Selects best model using Recall
    - Tunes probability threshold
    - Saves FULL pipeline (preprocessor + model)
    - Logs pipeline to MLflow (DagsHub)
    """

    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact,
    ):
        self.model_trainer_config = model_trainer_config
        self.data_transformation_artifact = data_transformation_artifact

    # =====================================================
    # ORCHESTRATOR
    # =====================================================
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("========== MODEL TRAINER STARTED ==========")

            init_mlflow()
            logging.info(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
            logging.info(f"MLflow Registry URI: {mlflow.get_registry_uri()}")

            X_train = np.load(
                self.data_transformation_artifact.X_train_file_path,
                allow_pickle=True,
            )
            y_train = np.load(
                self.data_transformation_artifact.y_train_file_path,
                allow_pickle=True,
            ).ravel()

            X_test = np.load(
                self.data_transformation_artifact.X_test_file_path,
                allow_pickle=True,
            )
            y_test = np.load(
                self.data_transformation_artifact.y_test_file_path,
                allow_pickle=True,
            ).ravel()

            return self.train_model(X_train, y_train, X_test, y_test)

        except Exception as e:
            raise BankException(e, sys)

    # =====================================================
    # CORE TRAINING LOGIC
    # =====================================================
    def train_model(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
    ) -> ModelTrainerArtifact:

        try:
            logging.info("Loading preprocessor")
            preprocessor = load_object(
                self.data_transformation_artifact.preprocessor_object_file_path
            )

            # ---------------- Class imbalance ----------------
            neg = np.sum(y_train == 0)
            pos = np.sum(y_train == 1)
            scale_pos_weight = neg / pos

            models = {
                "xgboost": XGBClassifier(
                    random_state=42,
                    eval_metric="logloss",
                    scale_pos_weight=scale_pos_weight,
                )
            }

            params = {
                "xgboost": {
                    "n_estimators": [100, 400],
                    "learning_rate": [0.05, 0.1],
                    "max_depth": [3, 5],
                }
            }

            logging.info("Evaluating candidate models")
            model_report: Dict = evaluate_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                preprocessor=None,  # IMPORTANT: data already transformed
                models=models,
                params=params,
            )

            best_model_name = max(
                model_report,
                key=lambda name: model_report[name]["score"],
            )

            trained_model = model_report[best_model_name]["best_estimator"]

            logging.info(f"Best model selected: {best_model_name}")

            # =====================================================
            # BUILD FINAL PIPELINE  (🔥 MOST IMPORTANT FIX)
            # =====================================================
            final_pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", trained_model),
                ]
            )

            os.makedirs(
                os.path.dirname(
                    self.model_trainer_config.trained_model_file_path
                ),
                exist_ok=True,
            )

            joblib.dump(
                final_pipeline,
                self.model_trainer_config.trained_model_file_path,
            )

            joblib.dump(final_pipeline, "final_model/model.pkl")

            # =====================================================
            # MLflow Tracking (PIPELINE)
            # =====================================================
            with mlflow.start_run(run_name=best_model_name, nested=True):

                mlflow.log_param("model_name", best_model_name)

                model_params = {
                    k: v
                    for k, v in trained_model.get_params().items()
                    if isinstance(v, (int, float, str, bool))
                }
                mlflow.log_params(model_params)

                y_train_proba = trained_model.predict_proba(X_train)[:, 1]
                y_test_proba = trained_model.predict_proba(X_test)[:, 1]

                threshold_results = evaluate_thresholds(
                    y_true=y_test,
                    y_pred_proba=y_test_proba,
                    min_threshold=self.model_trainer_config.min_probability_threshold,
                    max_threshold=self.model_trainer_config.max_probability_threshold,
                    step=self.model_trainer_config.threshold_step,
                )

                valid_thresholds = [
                    t
                    for t in threshold_results
                    if t["precision"]
                    >= self.model_trainer_config.precision_score
                    and t["false_positive_rate"]
                    <= self.model_trainer_config.false_positive_threshold
                ]

                if not valid_thresholds:
                    raise ValueError("No threshold satisfies business constraints")

                best_threshold = max(
                    valid_thresholds,
                    key=lambda t: t["recall"],
                )["threshold"]

                y_train_pred = (y_train_proba >= best_threshold).astype(int)
                y_test_pred = (y_test_proba >= best_threshold).astype(int)

                train_recall = recall_score(y_train, y_train_pred)
                test_recall = recall_score(y_test, y_test_pred)

                train_precision = precision_score(
                    y_train, y_train_pred, zero_division=0
                )
                test_precision = precision_score(
                    y_test, y_test_pred, zero_division=0
                )

                mlflow.log_metric("train_recall", train_recall)
                mlflow.log_metric("test_recall", test_recall)
                mlflow.log_metric("train_precision", train_precision)
                mlflow.log_metric("test_precision", test_precision)
                mlflow.log_metric("best_threshold", best_threshold)

                report = classification_report(
                    y_test, y_test_pred, output_dict=True
                )
                mlflow.log_dict(report, "classification_report.json")

                mlflow.sklearn.log_model(
                    sk_model=final_pipeline,
                    artifact_path="model",
                    registered_model_name="Bank_Term_Deposit_Prediction_Azure_Aws",
                )

                is_model_accepted = (
                    test_recall
                    >= self.model_trainer_config.expected_recall_score
                    and test_precision
                    >= self.model_trainer_config.precision_score
                )

                return ModelTrainerArtifact(
                    is_model_accepted=is_model_accepted,
                    trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                    train_recall=train_recall,
                    test_recall=test_recall,
                    train_precision=train_precision,
                    test_precision=test_precision,
                    base_recall=self.model_trainer_config.base_recall_score,
                    expected_recall=self.model_trainer_config.expected_recall_score,
                    best_probability_threshold=best_threshold,
                    classification_report=report,
                )

        except Exception as e:
            raise BankException(e, sys)

## below is the calibarted model  to to use this 


# import os
# import sys
# from typing import Dict

# import numpy as np
# import mlflow
# import mlflow.sklearn

# from sklearn.metrics import recall_score, precision_score, classification_report
# from sklearn.calibration import CalibratedClassifierCV
# from sklearn.pipeline import Pipeline
# from xgboost import XGBClassifier

# from source_main.exception.exception import BankException
# from source_main.logging.logging import logging
# from source_main.entity.config import ModelTrainerConfig
# from source_main.entity.artifact import (
#     ModelTrainerArtifact,
#     DataTransformationArtifact,
# )
# from source_main.utlis.main_utlis.utlis import (
#     load_object,
#     save_object,
#     evaluate_model,
#     evaluate_thresholds,
# )
# from source_main.components.mlflow__init__ import init_mlflow




# class ModelTrainer:
#     """
#     Model Trainer Component
#     -----------------------
#     - Trains model
#     - Improves precision using F-beta optimization
#     - Calibrates probabilities
#     - Saves FINAL model + transformer
#     - Logs to MLflow (DagsHub)
#     """

#     def __init__(
#         self,
#         model_trainer_config: ModelTrainerConfig,
#         data_transformation_artifact: DataTransformationArtifact,
#     ):
#         self.model_trainer_config = model_trainer_config
#         self.data_transformation_artifact = data_transformation_artifact

#     # =====================================================
#     # ORCHESTRATOR
#     # =====================================================
#     def initiate_model_trainer(self) -> ModelTrainerArtifact:
#         try:
#             logging.info("Initializing MLflow")
#             init_mlflow()

#             logging.info(f"Tracking URI: {mlflow.get_tracking_uri()}")
#             logging.info(f"Registry URI: {mlflow.get_registry_uri()}")

#             logging.info("========== MODEL TRAINER STARTED ==========")

#             X_train = np.load(
#                 self.data_transformation_artifact.X_train_file_path,
#                 allow_pickle=True,
#             )
#             y_train = np.load(
#                 self.data_transformation_artifact.y_train_file_path,
#                 allow_pickle=True,
#             ).ravel()

#             X_test = np.load(
#                 self.data_transformation_artifact.X_test_file_path,
#                 allow_pickle=True,
#             )
#             y_test = np.load(
#                 self.data_transformation_artifact.y_test_file_path,
#                 allow_pickle=True,
#             ).ravel()

#             artifact = self.train_model(
#                 X_train, y_train, X_test, y_test
#             )

#             logging.info("========== MODEL TRAINER COMPLETED ==========")
#             return artifact

#         except Exception as e:
#             raise BankException(e, sys)

#     # =====================================================
#     # CORE TRAINING LOGIC
#     # =====================================================
#     def train_model(
#         self,
#         X_train,
#         y_train,
#         X_test,
#         y_test,
#     ) -> ModelTrainerArtifact:

#         try:
#             logging.info("Loading preprocessor")
#             preprocessor = load_object(
#                 self.data_transformation_artifact.preprocessor_object_file_path
#             )

#             models = {
#                 "xgboost": XGBClassifier(
#                     random_state=42,
#                     eval_metric="logloss",
#                 )
#             }

#             params = {
#                 "xgboost": {
#                     "n_estimators": [100, 300],
#                     "learning_rate": [0.05, 0.1],
#                     "max_depth": [3, 5],
#                 }
#             }

#             logging.info("Evaluating model")
#             model_report: Dict = evaluate_model(
#                 X_train=X_train,
#                 y_train=y_train,
#                 X_test=X_test,
#                 y_test=y_test,
#                 preprocessor=preprocessor,
#                 models=models,
#                 params=params,
#             )

#             model_pipeline = model_report["xgboost"]["best_estimator"]

#             # =====================================================
#             # PROBABILITY CALIBRATION
#             # =====================================================
#             logging.info("Calibrating probabilities")

#             calibrated_model = CalibratedClassifierCV(
#                 estimator=model_pipeline.named_steps["model"],
#                 method="isotonic",
#                 cv=3,
#             )

           

#             calibrated_model.fit(X_train, y_train)

#             y_test_proba = calibrated_model.predict_proba(X_test)[:, 1]
#             y_train_proba = calibrated_model.predict_proba(X_train)[:, 1]

#             # =====================================================
#             # THRESHOLD OPTIMIZATION (F-BETA)
#             # =====================================================
#             threshold_results = evaluate_thresholds(
#                 y_true=y_test,
#                 y_pred_proba=y_test_proba,
#                 min_threshold=self.model_trainer_config.min_probability_threshold,
#                 max_threshold=self.model_trainer_config.max_probability_threshold,
#                 step=self.model_trainer_config.threshold_step,
#             )

#             BETA = 0.5  # precision-focused

#             valid_thresholds = [
#                 t for t in threshold_results
#                 if t["precision"] >= self.model_trainer_config.precision_score
#                 and t["recall"] >= self.model_trainer_config.base_recall_score
#                 and t["false_positive_rate"]
#                 <= self.model_trainer_config.false_positive_threshold
#             ]

#             if not valid_thresholds:
#                 raise ValueError("No threshold satisfies constraints")

#             best_threshold = max(
#                 valid_thresholds,
#                 key=lambda t: (
#                     (1 + BETA**2) * t["precision"] * t["recall"]
#                 ) / ((BETA**2 * t["precision"]) + t["recall"] + 1e-8),
#             )["threshold"]

#             # =====================================================
#             # FINAL METRICS
#             # =====================================================
#             y_train_pred = (y_train_proba >= best_threshold).astype(int)
#             y_test_pred = (y_test_proba >= best_threshold).astype(int)

#             train_recall = recall_score(y_train, y_train_pred)
#             test_recall = recall_score(y_test, y_test_pred)
#             train_precision = precision_score(y_train, y_train_pred)
#             test_precision = precision_score(y_test, y_test_pred)

#             report = classification_report(
#                 y_test, y_test_pred, output_dict=True
#             )

#             # =====================================================
#             # FINAL MODEL OBJECT
#             # =====================================================
#             final_model = {
#                 "model" : calibrated_model,
#                 "threshold" : best_threshold
#             }

#             os.makedirs(
#                 self.model_trainer_config.trained_model_dir,
#                 exist_ok=True,
#             )

#             final_model_path = self.model_trainer_config.trained_model_file_path


#             save_object(file_path=final_model_path,obj= final_model)

#             # =====================================================
#             # MLflow Logging
#             # =====================================================
#             with mlflow.start_run(run_name="xgboost_final", nested=True):

#                 mlflow.log_metric("train_recall", train_recall)
#                 mlflow.log_metric("test_recall", test_recall)
#                 mlflow.log_metric("train_precision", train_precision)
#                 mlflow.log_metric("test_precision", test_precision)
#                 mlflow.log_metric("best_threshold", best_threshold)

#                 mlflow.log_dict(report, "classification_report.json")

#                 mlflow.sklearn.log_model(
#                     sk_model=calibrated_model,
#                     name="final_model",
#                     registered_model_name="Bank_Term_Deposit_Prediction_Azure_Aws",
#                 )

#             is_model_accepted = (
#                 test_recall
#                 >= self.model_trainer_config.expected_recall_score
#             )

#             return ModelTrainerArtifact(
#                 is_model_accepted=is_model_accepted,
#                 trained_model_file_path=final_model_path,
#                 train_recall=train_recall,
#                 test_recall=test_recall,
#                 train_precision=train_precision,
#                 test_precision=test_precision,
#                 base_recall=self.model_trainer_config.base_recall_score,
#                 expected_recall=self.model_trainer_config.expected_recall_score,
#                 best_probability_threshold=best_threshold,
#                 classification_report=report,
#             )

#         except Exception as e:
#             raise BankException(e, sys)
