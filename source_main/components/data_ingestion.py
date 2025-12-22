import os
import sys
import pandas as pd 
import  numpy as np 
import mysql.connector
from dotenv import load_dotenv
from source_main.entity.artifact import DataIngestionArtifact

from source_main.entity.config import DataIngestionConfig   
from source_main.exception.exception import BankException
from source_main.logging.logging import logging
load_dotenv()


MYSQL_HOST= os.getenv("MYSQL_HOST")
MYSQL_USER= os.getenv("MYSQL_USER")
MYSQL_PASSWORD= os.getenv("MYSQL_PASSWORD")
MYSQL_DATABASE= os.getenv("MYSQL_DATABASE")

class DataIngestion:
    def __init__(self, data_ingestion_config:DataIngestionConfig):
        try:
            self.data_ingestion_config= data_ingestion_config
        except Exception as e:
            raise BankException(e, sys) from e
        
        
    def export_table_as_dataframe(self):
        try:
            table_name = self.data_ingestion_config.table_name
            #database_name = self.data_ingestion_config.database_name

            connection = mysql.connector.connect(
                host=MYSQL_HOST,
                user=MYSQL_USER,
                password=MYSQL_PASSWORD,
                database=MYSQL_DATABASE
            )
            cursor = connection.cursor()
            cursor.execute(f"SELECT * FROM {table_name}")

            rows = cursor.fetchall()
            column_names = [col[0] for col in cursor.description]
            df = pd.DataFrame(rows, columns=column_names)

            return df

        except Exception as e:
            raise BankException(e, sys)
        
        
    def export_feature_into_feature_store(self,dataframe:pd.DataFrame):
        try:
            feature_store_file_path= self.data_ingestion_config.feature_store_file_path
            dir_path= os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            return feature_store_file_path
        except Exception as e:
            raise BankException(e, sys) 
        
    def initiate_data_ingestion(self):
        try:
            logging.info('starting data ingestion')
            df= self.export_table_as_dataframe()
            feature_store_file_path = self.export_feature_into_feature_store(df)
            data_ingestion_artifact = DataIngestionArtifact(
            feature_store_file_path=feature_store_file_path )

            logging.info(
            f"Data ingestion completed. "
            f"Feature store path: {feature_store_file_path}")

            return data_ingestion_artifact

        except Exception as e:
            raise BankException(e, sys)