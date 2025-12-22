from datetime import datetime
import os

from typing import Optional
from source_main.constants import pipelineconstants 

print(pipelineconstants.ARTIFACT_DIR)
#print(pipelineconstants.PIPELINE_NAME)

class TrainingPipelineConfig:
    def __init__(self,timestamp=None):
        if timestamp is None:
            timestamp=datetime.now()
            timestamp= timestamp.strftime("%m_%d_%Y_%H_%M_%S")
            
            self.pipeline_name= pipelineconstants.PIPELINE_NAME
            self.artifact_name= pipelineconstants.ARTIFACT_DIR
            self.model_dir = os.path.join("final_models")
            self.artifact_dir =os.path.join(self.artifact_name, timestamp)
            self.timestamp= timestamp
            
            os.makedirs(self.artifact_dir, exist_ok=True) 
            os.makedirs(self.model_dir, exist_ok=True)
            
class DataIngestionConfig:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        self.data_ingestion_dir= os.path.join(
            training_pipeline_config.artifact_dir,
            pipelineconstants.DATA_INGESTION_DIR_NAME
        )
        self.feature_store_dir= os.path.join(
            self.data_ingestion_dir,
            pipelineconstants.DATA_INGESTION_FEATURE_STORE_DIR
        )
        self.feature_store_file_path= os.path.join(
            self.feature_store_dir,
            pipelineconstants.FILE_NAME
        )
        
        self.database_name= pipelineconstants.DATA_INGESTION_DATABASE_NAME
        self.table_name= pipelineconstants.DATA_INGESTION_TABLE_NAME
        
        #os.makedirs(self.feature_store_dir, exist_ok=True)
        
class DataValidationConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_validation_dir= os.path.join(
            training_pipeline_config.artifact_dir,
            pipelineconstants.DATA_VALIDATION_DIR_NAME
        )
        self.valid_dir=os.path.join(
            self.data_validation_dir,
            pipelineconstants.DATA_VALIDATION_VALID_DIR
        )
        self.invalid_dir=os.path.join(
            self.data_validation_dir,
            pipelineconstants.DATA_VALIDATION_INVALID_DIR
        )
        self.drift_report_file_path=os.path.join(
            self.data_validation_dir,
            pipelineconstants.DATA_VALIDATION_DRIFT_REPORT_DIR,
            pipelineconstants.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME
        )


class DataTransformationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):

        # Root directory
        self.data_transformation_dir = os.path.join(
            training_pipeline_config.artifact_dir,
            pipelineconstants.DATA_TRANSFORMATION_DIR_NAME
        )

        # Transformed root
        self.transformed_dir = os.path.join(
            self.data_transformation_dir,
            pipelineconstants.DATA_TRANSFORMATION_TRANSFORMED_DIR
        )

        # Train directory
        self.train_dir = os.path.join(
            self.transformed_dir,
            pipelineconstants.DATA_TRANSFORMATION_TRAIN_DIR
        )

        # Test directory
        self.test_dir = os.path.join(
            self.transformed_dir,
            pipelineconstants.DATA_TRANSFORMATION_TEST_DIR
        )

        # Train files
        self.X_train_file_path = os.path.join(
            self.train_dir,
            pipelineconstants.DATA_TRANSFORMATION_X_TRAIN_FILE
        )

        self.y_train_file_path = os.path.join(
            self.train_dir,
            pipelineconstants.DATA_TRANSFORMATION_Y_TRAIN_FILE 

        )

        # Test files
        self.X_test_file_path = os.path.join(
            self.test_dir,
            pipelineconstants.DATA_TRANSFORMATION_X_TEST_FILE
        )

        self.y_test_file_path = os.path.join(
            self.test_dir,
            pipelineconstants.DATA_TRANSFORMATION_Y_TEST_FILE
        )

        # Preprocessor object
        self.transformed_object_dir = os.path.join(
            self.data_transformation_dir,
            pipelineconstants.DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR
        )

        self.preprocessor_object_file_path = os.path.join(
            self.transformed_object_dir,
            pipelineconstants.DATA_TRANSFORMATION_PREPROCESSOR_FILE
        )

        # Split parameters
        self.test_size =pipelineconstants.DATA_TRANSFORMATION_TRAIN_TEST_SPLIT_RATIO
        self.random_state =pipelineconstants.DATA_TRANSFORMATION_RANDOM_STATE
        
        
class ModelTrainerConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.model_trainer_dir = os.path.join(
            training_pipeline_config.artifact_dir,
            pipelineconstants.MODEL_TRAINER_DIR_NAME
        )

        self.trained_model_dir = os.path.join(
            self.model_trainer_dir,
            pipelineconstants.MODEL_TRAINER_TRAINED_MODEL_DIR
        )

        self.trained_model_file_path = os.path.join(
            self.trained_model_dir,
            pipelineconstants.MODEL_TRAINER_TRAINED_MODEL_NAME
        )
        self.expected_recall_score=pipelineconstants.MODEL_TRAINER_EXPECTED_RECALL
        self.base_recall_score=pipelineconstants.MODEL_TRAINER_BASE_RECALL_SCORE
        self.precision_score=pipelineconstants.MODEL_TRAINER_MIN_PRECISION_SCORE
        self.false_positive_threshold=pipelineconstants.MODEL_TRAINER_MAX_FALSE_POSITIVE_RATE
        self.min_probability_threshold= pipelineconstants.MODEL_TRAINER_MIN_THRESHOLD
        self.max_probability_threshold= pipelineconstants.MODEL_TRAINER_MAX_THRESHOLD
        self.threshold_step= pipelineconstants.MODEL_TRAINER_THRESHOLD_STEP
        
        
        