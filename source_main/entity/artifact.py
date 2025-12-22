from dataclasses import dataclass
from typing import Dict

@dataclass 
class DataIngestionArtifact:
    feature_store_file_path: str

@dataclass
class DataValidationArtifact:
    validation_status: bool
    validated_file_path: str
    drift_report_file_path: str
    
@dataclass
class DataTransformationArtifact:
    X_train_file_path: str
    y_train_file_path: str
    X_test_file_path: str
    y_test_file_path: str
    preprocessor_object_file_path: str
    
    
@dataclass
class ModelTrainerArtifact:
    is_model_accepted: bool
    trained_model_file_path: str
    train_recall: float
    test_recall: float

    train_precision: float
    test_precision: float

    base_recall: float
    expected_recall: float

    best_probability_threshold: float

    classification_report: Dict[str, Dict[str, float]]
    
    
    
@dataclass 
class ClassificationMetricArtifact:
    f1_score: float
    precision_score: float
    recall_score: float
    accuracy_score: float