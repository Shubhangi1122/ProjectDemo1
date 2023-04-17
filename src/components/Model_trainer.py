import os
import sys
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor
from  catboost import CatBoostRegressor
from xgboost import XGBRegressor
from src.utils import evalute_model,save_object
from sklearn.metrics import r2_score

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            model_report:dict=evalute_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models)
            
            ## to get best model score
            best_model_score=max(sorted(model_report.values()))

            ## get best model name
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException('No best model found')
            logging.info("Best model found")


            save_object(file_path=self.model_trainer_config.trained_model_file_path,
                        obj=best_model)
            
            predicted =best_model.predict(X_test)
            r2_score1=r2_score(y_test,predicted)

            return r2_score1
            

        except Exception as e:
            raise CustomException(e,sys)



