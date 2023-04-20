import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
import dill
from sklearn.metrics import r2_score 
from sklearn.model_selection import GridSearchCV
import pickle

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)

def evalute_model(X_train,y_train,X_test,y_test,models,param):
    try:
        report={}

        for i in range(len(list(models))):
            model=list(models.values())[i]
            para=param[list(models.keys())[i]]
            #model.fit(X_train,y_train)

            gs=GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)
            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)
            y_predict_train=model.predict(X_train)
            y_predict_test=model.predict(X_test)

            training_score=r2_score(y_train,y_predict_train)
            testing_score=r2_score(y_test,y_predict_test)

            report[list(models.keys())[i]]=testing_score

        return report
    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return  pickle.load(file_obj)
        
    except Exception as e:
        raise CustomException(e,sys)
               