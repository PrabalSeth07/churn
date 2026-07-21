import os
import sys
import pandas as pd

from src.components.data_transformation import DataTransformation

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")

class DataIngestion:

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the Data Ingestion component.")

        try:
            # Load dataset
            df = pd.read_csv("notebook/data/Telco_Customer_Churn.csv")
            logging.info("Dataset loaded successfully into a DataFrame.")

            # Create artifacts directory
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            # Save raw dataset
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("Train-test split initiated.")
            
            # Dataset-specific cleaning
            df["TotalCharges"] = pd.to_numeric(
                df["TotalCharges"],
                errors="coerce"
            )
            df = df.dropna().reset_index(drop=True)

            # Split dataset
            train_set, test_set = train_test_split(df,test_size=0.2,random_state=42,stratify=df["Churn"])

            # Save train dataset
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            # Save test dataset
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Data ingestion completed successfully.")

            return (self.ingestion_config.train_data_path,self.ingestion_config.test_data_path)

        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    obj = DataIngestion()

    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()

    train_arr, test_arr, preprocessor_path = (
        data_transformation.initiate_data_transformation(
            train_data,
            test_data
        )
    )