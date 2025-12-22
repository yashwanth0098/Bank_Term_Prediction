from source_main.components.data_ingestion import DataIngestion
from source_main.components.data_validation import DataValidation
from source_main.components.data_transformation import DataTransformation
from source_main.entity.config import DataIngestionConfig, TrainingPipelineConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig
from source_main.exception.exception import BankException
from source_main.components.data_model_trainer import ModelTrainer
import sys,os
from source_main.logging.logging import logging



if __name__=="__main__":
    try:
        trainingpipelineconfig= TrainingPipelineConfig()
        data_ingestion_config= DataIngestionConfig(trainingpipelineconfig)
        dataingestion= DataIngestion(data_ingestion_config)
        logging.info("Exporting the data from source to feature store")
        dataingestionartifact=dataingestion.initiate_data_ingestion()
        print(dataingestionartifact)
        logging.info("Data Ingestion Completed")
    
        data_validation_config= DataValidationConfig(trainingpipelineconfig)
        data_validation= DataValidation(data_validation_config,data_ingestion_config)
        data_validationartifact=data_validation.initiate_data_validation()
        print(data_validationartifact)
    
        data_validation_config = DataValidationConfig(trainingpipelineconfig)
        data_validation = DataValidation(data_validation_config, data_ingestion_config)
        logging.info("Data Starting data validation")
        data_validationartifact = data_validation.initiate_data_validation()
        print(data_validationartifact)
        logging.info("Data Validation Completed")
    
        data_transformation_config= DataTransformationConfig(trainingpipelineconfig)
        data_transformation= DataTransformation(data_transformation_config,data_validationartifact)
        logging.info("Data Starting data transformation")
        data_transformationartifact=data_transformation.initiate_data_transformation()
        print(data_transformationartifact)
        logging.info("Data Transformation Completed")
    
        modeltrainer_config=ModelTrainerConfig(trainingpipelineconfig)
        modeltrainer=ModelTrainer(modeltrainer_config,data_transformationartifact)
        modeltrainerartifact=modeltrainer.initiate_model_trainer()
        print(modeltrainerartifact)
        logging.info("Model Training Completed")
    
    
    
    
    except Exception as e:
        raise BankException(e, sys)
    
    
           