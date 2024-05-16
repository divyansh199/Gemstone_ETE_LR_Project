import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import OrdinalEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging 

from src.utils import save_object

@dataclass 
class DataTransfromationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_tranformation_config = DataTransfromationConfig()
    
    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation initated')

            #Defining the columns should be odinal_encoded and which should be scaled
            numerical_cols = ['carat','depth','table','x','y','z']
            categorical_cols = ['cut', 'color', 'clarity']

            #defining custom ranking for each odinal variable
            cut_categories = ['Fair','Good','Very Good','Premium','Ideal']
            color_categories= ['D','E','F','G','H','I','J']
            clarity_categories= ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            #numerical pipeline
            num_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy="median")),
                    ('scaler',StandardScaler())
                ]
            )
            
            logging.info('pipeline initiated')

            #categorical pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy="most_frequent")),
                    ('ordinalencoder',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                    ('scaler',StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer([
                ('num_pipeline',num_pipeline,numerical_cols),
                ('cat_pipeline',cat_pipeline,categorical_cols)
            ])

            return preprocessor
            logging.info("pipeline completed")
  
        except Exception as e:
            logging.info("Error in data trasformation")
            raise CustomException(e,sys)
        
    def initiate_data_trasformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading Train and Test data completed")
            logging .info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging .info(f'Test Dataframe Head : \n{test_df.head().to_string()}')

            logging.info('Obtaning preprocessing object')

            target_column_name = 'price'
            drop_columns = [target_column_name,'id']
            
            input_feature_train_df = train_df.drop(columns = drop_columns,axis =1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns = drop_columns,axis =1)
            target_feature_test_df = test_df[target_column_name]

            preprocessing_obj = self.get_data_transformation_object()

            ## Transforming using preprocessor object
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)  

            logging.info("applying preprocessing object on traning and testing dataset")

            train_arr = np.c_[input_feature_train_arr,target_feature_train_df]
            test_arr = np.c_[input_feature_test_arr,target_feature_test_df]

            save_object(
                file_path=self.data_tranformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )
            logging.info('Preprocessor Pickel file saved')

            return (
                train_arr,
                test_arr,
                self.data_tranformation_config.preprocessor_obj_file_path
            ) 
        except Exception as e:
            logging.info("Exception occured in the  initated_datatrasnformation")
            raise CustomException(e,sys) 

