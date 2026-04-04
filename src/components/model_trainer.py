import sys
import os
from src.logger import logging
from dataclasses import dataclass
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from sklearn.ensemble import (AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor)
from src.exception import CustomException
from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    restrained_model_file_path= os.path.join('artifacts',"model.pkl")
    
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("Splitting train and test data")
            x_train,y_train,x_test,y_test = (train_arr[:,:-1],train_arr[:,-1],test_arr[:,:-1],test_arr[:,-1])
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "KNN Regressor": KNeighborsRegressor(),
                "XGBoosting": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }
            model_report:dict = evaluate_models(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test, models=models)
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]
            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best model found {best_model_name}")
            save_object(file_path=self.model_trainer_config.restrained_model_file_path, obj= best_model)
            predicted = best_model.predict(x_test)
            r2 = r2_score(y_test,predicted)
            return r2
        
        except Exception as e:
            raise CustomException(e,sys)



