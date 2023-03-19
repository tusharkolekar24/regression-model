import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.exception import CustomException
import sys
import os
from src.logger import logging
from src.components.data_transformation import DataTransformation,DataTransformationConfig
from src.components.model_trainer import ModelTrainer,ModelTrainerConfig
@dataclass
class DataIngestionConfig:
      raw_data_path  : str = os.path.join(os.getcwd(),'artifacts','raw_data.csv')
      train_data_path: str = os.path.join(os.getcwd(),'artifacts','train_data.csv')
      test_data_path : str = os.path.join(os.getcwd(),'artifacts','test_data.csv')


class DataIngestion:
      def __init__(self):
            self.dataingestion = DataIngestionConfig()

      def initiate_data_ingestion(self):
          logging.info("Data Ingestion Process Started")

          try:
              df = pd.read_csv(os.path.join(os.getcwd(),'datasets','students.csv'))

              logging.info('Read the dataset as dataframe')

              os.makedirs(os.path.dirname(self.dataingestion.raw_data_path),
                          exist_ok=True)
              
              df.to_csv(self.dataingestion.raw_data_path,
                        index=False,header=True)

              logging.info("Train test split initiated")
              
              train_set,test_set=train_test_split(df,test_size=0.2,
                                                  random_state=42)

              train_set.to_csv(self.dataingestion.train_data_path,
                               index=False,header=True)

              test_set.to_csv(self.dataingestion.test_data_path,
                              index=False,header=True)

              logging.info("Inmgestion of the data iss completed")              

              return(
                    self.dataingestion.train_data_path,
                    self.dataingestion.test_data_path
                )

          except Exception as e:
               raise CustomException(e,sys)
          
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()
    
    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))

    