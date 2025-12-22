import os
import sys
import pandas as pd
from dotenv import load_dotenv

from sqlalchemy import create_engine,text
from sqlalchemy.exc import SQLAlchemyError
from source_main.exception.exception import BankException
from source_main.logging.logging import logging
load_dotenv() # load environment variables from .env file

MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_PORT = os.getenv("MYSQL_PORT")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE")
DATABASE_URL = f"mysql+mysqlconnector://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}/{MYSQL_DATABASE}"

def make_engine():
    try:
        engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=1800, future=True)
        return engine
    except SQLAlchemyError as e:
        raise BankException(e, sys)


class bankdata_exporter:
    def __init__(self):
        try:
            self.engine = make_engine()
        except Exception as e:
            raise BankException(e, sys)
        
    def csv_to_dataframe(self, csv_file_path: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(csv_file_path)
            logging.info(f"CSV file {csv_file_path} loaded into DataFrame")
            return df
        except Exception as e:
            raise BankException(e, sys)
        
    def ensure_table(self, table_name: str, dataframe: pd.DataFrame):
        try:
            with self.engine.connect() as connection:
                columns_with_types = ", ".join([f"{col} VARCHAR(255)" for col in dataframe.columns])
                create_table_query = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns_with_types});"
                connection.execute(text(create_table_query))
                logging.info(f"Table {table_name} ensured in database")
        except Exception as e:
            raise BankException(e, sys)

    def insert_data_mysql(self, table_name: str, dataframe: pd.DataFrame):
        try:
            with self.engine.connect() as connection:
                dataframe.to_sql(name=table_name, con=connection, if_exists='append', index=False)
                logging.info(f"Data inserted into table {table_name} successfully")
        except Exception as e:
            raise BankException(e, sys)


if __name__=="__main__":
    
    FILE_PATH="data/bank_data.csv"
    TABLE_NAME="bankdeposit"
    
    try:
        logging.info("Starting data push to MySQL")
        loader=bankdata_exporter()
        df=loader.csv_to_dataframe(FILE_PATH)
        logging.info(f' Read data from {FILE_PATH} successfully')
        loader.ensure_table(TABLE_NAME, df)
        inserted= loader.insert_data_mysql(TABLE_NAME,df)
        logging.info(f' Data inserted into {TABLE_NAME} table successfully')
    except Exception as e:
        raise BankException(e, sys)
    
    logging.info("Data push to MySQL completed")