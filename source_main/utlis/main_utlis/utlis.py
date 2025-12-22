import yaml
from source_main.exception.exception import BankException
from source_main.logging.logging import logging
import os,sys
import pandas as pd
import numpy as np
import  joblib,pickle
from sklearn.metrics import recall_score, precision_score, confusion_matrix
import numpy as np
from typing import List, Dict


from sklearn.model_selection import GridSearchCV


def read_yaml_file(file_path:str)->dict:
    try:
        with open(file_path, 'r') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise BankException(e, sys)
    
def write_yaml_file(file_path:str, content:object)->None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as yaml_file:
            yaml.dump(content, yaml_file)
    except Exception as e:
        raise BankException(e, sys)
    
def load_schema():
    try:
        schema_config = os.path.join("data_schema","schema.yaml")
        with open(schema_config, 'r') as yaml_file:
            schema = yaml.safe_load(yaml_file)
        numeric_columns = schema.get("numeric_columns",[])
        categorical_columns = schema.get("categorical_columns",[])   
        target_column = schema.get("target_column",None)
        
        if not target_column:
            raise KeyError("Target column not found in schema.yaml")
        
        return numeric_columns, categorical_columns, target_column
    
    except FileNotFoundError:
        raise FileNotFoundError("Schema file not found at {schema_config}")

    except yaml.YAMLError as e:
        raise ValueError(f'Error parsing schema.yaml: {e}')
    
    
def load_object(file_path: str)->object:
    try:
        if file_path.endswith(".npy"):
            raise ValueError(
                "Use np.load() for .npy files, not load_object()"
            )

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} does not exist")

        return joblib.load(file_path)

    except Exception as e:
        raise BankException(e, sys)
        
        
def save_object(file_path:str,obj:object)->None:
    try:
        logging.info("Entered save_object method of utils")
        dir_name=os.path.dirname(file_path)
        if dir_name:
            os.makedirs(dir_name,exist_ok=True)
        joblib.dump(obj,file_path)
        logging.info("Exited save_object method of utils")
    except Exception as e:
        raise BankException(e, sys)
    

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score
from sklearn.pipeline import Pipeline

def evaluate_model(
    X_train,
    y_train,
    X_test,
    y_test,
    preprocessor,
    models: dict,
    params: dict,
):
    report = {}

    for model_name, model in models.items():
        param_grid = {
            f"model__{k}": v for k, v in params.get(model_name, {}).items()
        }

        pipeline = Pipeline(
            steps=[
                #("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="recall",
            cv=5,
            n_jobs=-1,
            error_score="raise"
        )

        grid.fit(X_train, y_train)

        y_test_pred = grid.predict(X_test)
        recall = recall_score(y_test, y_test_pred)

        report[model_name] = {
            "score": recall,
            "best_estimator": grid.best_estimator_,
        }

    return report


def evaluate_thresholds(
    y_true,
    y_pred_proba,
    min_threshold: float,
    max_threshold: float,
    step: float,
) -> List[Dict]:
    """
    Computes recall, precision, and FPR for different
    probability thresholds.
    """

    results = []

    for threshold in np.arange(min_threshold, max_threshold + step, step):
        y_pred = (y_pred_proba >= threshold).astype(int)

        recall = recall_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        results.append({
            "threshold": round(float(threshold), 4),
            "recall": recall,
            "precision": precision,
            "false_positive_rate": fpr,
        })

    return results

    
    