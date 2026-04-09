import os
import sys
from src.exception import CustomException # Custom error handling ke liye
from src.logger import logging             # Project ki tracking (logs) ke liye
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig



# 1. Configuration Class: Iska kaam hai paths define karna (Artifacts folder ke liye)
@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv") # Train file kahan save hogi
    test_data_path: str=os.path.join('artifacts',"test.csv")   # Test file kahan save hogi
    raw_data_path: str=os.path.join('artifacts',"data.csv")    # Raw data kahan save hoga

# 2. Main Data Ingestion Class
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig() # Config ko initialize kiya

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Data ko read karna (Local folder, Database ya Cloud se)
            df=pd.read_csv('notebook\data\stud.csv') 
            logging.info('Read the dataset as dataframe')

            # 'artifacts' folder banana agar nahi bana toh
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            # Raw data ko artifacts folder mein save karna
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            # Data ko Train aur Test mein split karna
            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            # Split kiye huye data ko CSV files mein save karna
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of the data is completed")

            # Agle component (Transformation) ke liye paths return karna
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys) # Agar error aaye toh CustomException trigger hoga
        
# 3. Execution Block (Main)
if __name__=="__main__":
    # Data Ingestion start karna
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    # Data milne ke baad Transformation call karna
    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    # # Transform hone ke baad Model training call karna
    # modeltrainer=ModelTrainer()
    # print(modeltrainer.initiate_model_trainer(train_arr,test_arr))