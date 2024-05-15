import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate

from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initate_model_traning(self,train_array,test_array):
        try:
            logging.info("splitting Dependent and Independent variables from train and test")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                        'LinearRegression': LinearRegression(),
                        'Lasso': Lasso(),
                        'Ridge': Ridge(),
                        'Elasticnet': ElasticNet()  
                    }
            
            model_report:dict = evaluate(X_train,y_train,X_test,y_test,models)
            print(model_report)
            print('\n=============================================')
            logging.info(f'Model Report : {model_report}')

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            print(f'best model found, model name: {best_model_name}, r2 score: {best_model_score}')
            print('\n===============================================')
            logging.info(f'best model found, model name: {best_model_name}, r2 score: {best_model_score}')

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

        except Exception as e:
            logging.info('exception occured at Model Traning')
            raise CustomException(e,sys)


