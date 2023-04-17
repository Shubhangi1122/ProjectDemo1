import os
import sys
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.utils import save_object

@dataclass

class Data_Transformation_Config:
    preprocessor_obj_file_path=os.path.join('artifact','preprocessor.pkl')
class Data_Transformation:
    def __init__(self):
        self.data_transformation_config=Data_Transformation_Config()

    def  Get_Data_Transformer_obj(self):
        try:
            numerical_columns=["reading_score","writing_score"]
            categorical_columns=["gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course"]
            numerical_pipeline=Pipeline(steps=[("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())])
            categorical_pipeline=Pipeline(steps=[("imputer",SimpleImputer(strategy="most_frequent")),("onehotencoder",OneHotEncoder())])
            logging.info("Numerical and categorical columns preprocessing completed")

            preprocessor=ColumnTransformer([("numerical_pipeline",numerical_pipeline,numerical_columns),('categorical_pipeline',categorical_pipeline,categorical_columns)])
            return preprocessor
            #logging.info(print(preprocessor))
        except Exception as e:
            raise CustomException(e,sys)

    def Initiate_Data_Transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info('Reading of training and testing data completed')
            logging.info("Obtaining preprpcessing object")
            preprocessing_obj=self.Get_Data_Transformer_obj()
            target_column_name="math_score"
            numerical_columns=["reading_score","writing_score"]
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            logging.info(f"training input feature: {input_feature_train_df}")
            
            target_feature_train_df=train_df[target_column_name]
            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]
            logging.info("Applying preprocessing object on train and test datasets")
            logging.info(f"testing input feature: {input_feature_test_df}")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            logging.info("Applied preprocessed object on train and test dataset")

            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            
            logging.info(f"Saving preprocessed object")
            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path,obj=preprocessing_obj)

            return (train_arr,test_arr,self.data_transformation_config.preprocessor_obj_file_path)

        except Exception as e:
            raise CustomException(e,sys)



