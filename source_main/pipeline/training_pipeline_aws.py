import os 
import sys 
from dotenv import load_dotenv
load_dotenv 

from source_main.exception.exception import BankException
from source_main.logging.logging import logging

from source_main.components.data_ingestion import DataIngestion
from source_main.components.data_validation import DataValidation
from source_main.components.data_transformation import DataTransformation
from source_main.components.data_model_trainer import ModelTrainer


from source_main.entity.config import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataTransformationConfig,
    DataValidationConfig,
    ModelTrainerConfig
    
)

from source_main.entity.artifact import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact
    
)


from source_main.constants.pipelineconstants import TRAINING_BUCKET_NAME
from cloud.s3_syncer import S3Sync
from source_main.constants.pipelineconstants import SAVE_MODEL_DIR


class TrainingPipeline:
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()
        self.s3_sync = S3Sync()
    
    def start_data_ingestion(self):
        try:
            self.data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info('Starting data ingestion')
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            return data_ingestion.initiate_data_ingestion()
            logging.info('Data ingestion completed')
        except Exception as e:
            raise BankException(e, sys)
        
        
    def start_data_validation(self,data_ingestion_artifact: DataIngestionArtifact):
        try:
            data_validation_config = DataValidationConfig(training_pipeline_config=self.training_pipeline_config)   
            data_validation = DataValidation(data_validation_config=data_validation_config, data_ingestion_artifact=data_ingestion_artifact)
            logging.info('Starting data validation')
            data_validation_artifact = data_validation.initiate_data_validation()
            return data_validation_artifact
            logging.info('Data validation completed')
        except Exception as e:
            raise BankException(e, sys)
        
    def start_data_transformation(self,data_validation_artifact:DataValidationArtifact):
        try:
            data_transformation_config = DataTransformationConfig(training_pipeline_config=self.training_pipeline_config)
            data_transformation = DataTransformation(data_transformation_config=data_transformation_config, data_validation_artifact=data_validation_artifact)
            logging.info('Starting data transformation')
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            return data_transformation_artifact
            logging.info('Data transformation completed')
        except Exception as e:
            raise BankException(e, sys) 
        
    def start_model_trainer(self,data_transformation_artifact:DataTransformationArtifact):
        try:
            model_trainer_config = ModelTrainerConfig(training_pipeline_config=self.training_pipeline_config)
            model_trainer = ModelTrainer(model_trainer_config=model_trainer_config, data_transformation_artifact=data_transformation_artifact)
            logging.info('Starting model trainer')
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            return model_trainer_artifact
            logging.info('Model trainer completed')
        except Exception as e:
            raise BankException(e, sys)
        
    def sync_artifacts_to_s3(self):
        try:
            aws_bucket_url=f"s3://{TRAINING_BUCKET_NAME}/artifacts/{self.training_pipeline_config.timestamp}"
            self.s3_sync.sync_folder_to_s3(folder=self.training_pipeline_config.artifact_dir, aws_bucket_url=aws_bucket_url)
            logging.info(f"Syncing {self.training_pipeline_config.artifact_dir} to {aws_bucket_url}")
           
        except Exception as e:  
            raise BankException(e, sys)
        
    def sync_saved_model_dir_to_s3(self) -> None:
        try:
            aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/final_model/{self.training_pipeline_config.timestamp}"
            self.s3_sync.sync_folder_to_s3(folder=self.training_pipeline_config.model_dir,aws_bucket_url=aws_bucket_url)
        except Exception as e:
            raise BankException(e, sys)
        
    def run_pipeline(self):
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(data_validation_artifact=data_validation_artifact)
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
            self.sync_artifacts_to_s3()
            self.sync_saved_model_dir_to_s3()
            return model_trainer_artifact
        except Exception as e:
            raise BankException(e, sys)
    

