import os
import sys
import pandas as pd

from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass

from sklearn.model_selection import train_test_split

# initialize the Data Ingestion Configuration
@dataclass
class DataIngestionConfig:
    train_data_path : str = os.path.join("artifacts","train.csv")
    test_data_path : str = os.path.join("artifacts","test.csv")
    raw_data_path : str = os.path.join("artifacts","raw.csv")

# Create Data Ingestion class
class DataIngestion:

    def __init__(self):
        self.ingestion_config = DataIngestionConfig() 

    
    def initiate_data_ingestion(self):
        logging.info("Data Ingestion methods starts")

        try:
            df = pd.read_csv(os.path.join("notebooks","insurance.csv"))
            logging.info("Dataset read as pandas dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False)

            logging.info("Raw data is created")

            train_set , test_set = train_test_split(df,test_size=0.3,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False)
            logging.info("Train data is created")

            test_set.to_csv(self.ingestion_config.test_data_path,index=False)
            logging.info("Test data is created")

            logging.info("Ingestion of data completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info("Exception occurred in data ingestion stage")
            raise CustomException(e,sys)


