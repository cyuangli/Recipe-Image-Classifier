import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
from src.config.configuration import ConfigurationManager

class DataIngestion:
    def __init__(self):
        config_manager = ConfigurationManager()
        self.config = config_manager.get_data_ingestion_config()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component.")
        try:
            df = pd.read_csv(self.config.source_path)
            logging.info("Read the source dataset as dataframe.")

            os.makedirs(os.path.dirname(self.config.raw_data_path), exist_ok=True)

            df.to_csv(self.config.raw_data_path, index=False, header=True)
            
            logging.info("Train-test split initiated.")
            train_set, test_set = train_test_split(df, test_size=self.config.test_size, random_state=self.config.random_state)

            train_set.to_csv(self.config.train_data_path, index=False, header=True)

            test_set.to_csv(self.config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed.")

            return (
                self.config.train_data_path,
                self.config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)