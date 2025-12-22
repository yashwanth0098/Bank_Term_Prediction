from source_main.entity.artifact import ClassificationMetricArtifact
from source_main.exception.exception import BankException
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score
import numpy as np
import sys 


def get_classification_metric(y_true, y_pred) -> ClassificationMetricArtifact:
    try:
        # Specify average='binary' for binary classification, adjust if multi-class
        model_f1_score = f1_score(y_true, y_pred, average='binary')
        model_precision_score = precision_score(y_true, y_pred, average='binary')
        model_recall_score = recall_score(y_true, y_pred, average='binary')
        model_accuracy_score = accuracy_score(y_true, y_pred)

        classification_metric_artifact = ClassificationMetricArtifact(
            f1_score=model_f1_score,
            precision_score=model_precision_score,
            recall_score=model_recall_score,
            accuracy_score=model_accuracy_score
        )

        return classification_metric_artifact

    except Exception as e:
        raise Bank_Exception(e, sys)