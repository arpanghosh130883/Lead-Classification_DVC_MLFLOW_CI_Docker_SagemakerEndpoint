import pandas as pd
import argparse
import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import mlflow
# import shutil
# from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories
import sqlite3
import dvc.api

# path = 'dvc_data/raw_data.csv'
# repo = 'C:/Users/Aishwarya.Chandra/SelfDevelopment/leadgen-experimentation'
# # repo = 'C:/Users/AISHWA~1.CHA/AppData/Local/Temp/dvc-storage'
# version = 'B'

# data_url = dvc.api.get_url(path = path,   repo=repo,   rev=version)



STAGE = "STAGE_01_DATA_PREPARATION" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def main(config_path, params_path):

   # mlflow.set_tracking_uri('sqlite:///trial_database.db')
   # mlflow.set_tracking_uri("sqlite:///mlflow_db.db")

    ## read config files
   config = read_yaml(config_path)
   params = read_yaml(params_path)
   
   ## read data from source folder
   
   source_data = config["source_data"]
   input_data = os.path.join(source_data["data_directory"], source_data["data_file"])

   


   ## read input data as a dataframe
   # in_df = pd.read_csv(data_url, sep=",")
   in_df = pd.read_csv(input_data, sep=",")

   # mlflow.log_param('data_url', data_url)
   # mlflow.log_param('data_version', version)
   mlflow.log_param('input_rows', in_df.shape[0])
   mlflow.log_param('input_columns', in_df.shape[1])

   input_features = pd.DataFrame(list(in_df.columns))

   artifacts = config['artifacts']
   output_file = os.path.join(artifacts['artifacts_directory'], artifacts['input_features_dir'])
   create_directories([output_file])
   input_features_path = os.path.join(output_file, artifacts["input_features_file"])
   input_features.to_csv(input_features_path, header=False, index=False)

   mlflow.log_artifact(input_features_path)
    

   # in_df.drop(["ID", "Name",'bing_news_url', 'bing_news_desc', 'bing_news_latest_date'], inplace=True, axis=1)
   df = in_df.drop_duplicates()

   #Renaming the columns as per standard
   df.rename(columns = {'overseas_countries':'OverseasCountries', 'profitorlossamount':'ProfitOrLossAmount','globalultimatefamilytreelinkagecount':'GlobalUltimateFamilyTreeLinkageCount','totalliabilitiesamount':'TotalLiabilitiesAmount','totalcurrentliabilitiesamount':'TotalCurrentLiabilitiesAmount', 'totalcurrentassetsamount':'TotalCurrentAssetsAmount', 'emp_count': 'EmpCount', 'Emp count':'EmpCount','Networth':'NetWorth','networth':'NetWorth', 'totalassetsamount':'TotalAssetsAmount', 'sales_turnover_gbp':'SalesTurnoverGBP', 'Sales_turnover_GBP':'SalesTurnoverGBP', 'lineofbusinessdescription':'LineOfBusinessDescription','primarytownname':'PrimaryTownName'}, inplace =True)
   
   #To know whether the incoming lead has business overseas or not
   df['OverseasCountriesKnown'] = np.where(df['OverseasCountries'] == 'uk', 0,1)

   ## Insert target column 
   df['IsClientConverted'] = np.where(df['acc_status'] == 'Marketing',0,1)

   #Taking required columns
   df_req = df[['OverseasCountriesKnown', 
      'ProfitOrLossAmount', 'GlobalUltimateFamilyTreeLinkageCount',
      'TotalLiabilitiesAmount', 'TotalCurrentLiabilitiesAmount',
      'TotalCurrentAssetsAmount', 'EmpCount', 'NetWorth',
      'TotalAssetsAmount', 'SalesTurnoverGBP', 'LineOfBusinessDescription',
      'PrimaryTownName', 'IsClientConverted']]

    ## Treating the null values for numerical data

    # Get the filling values

   SalesTurnover_GBP = params['Numeric']['SalesTurnoverGBP']
   ProfitOrLoss_Amount = params['Numeric']['ProfitOrLossAmount']
   Emp_Count = params['Numeric']['EmpCount']
   # FamilyTreeHierarchy_Level = params['Numeric']['FamilyTreeHierarchyLevel']
   GlobalUltimateFamilyTreeLinkage_Count = params['Numeric']['GlobalUltimateFamilyTreeLinkageCount']
   TotalAssets_Amount = params['Numeric']['TotalAssetsAmount']
   Net_Worth = params['Numeric']['NetWorth']

 


   # Fill the null values

   df_req['SalesTurnoverGBP'].fillna(SalesTurnover_GBP, inplace=True)
   df_req['ProfitOrLossAmount'].fillna(ProfitOrLoss_Amount, inplace=True)
   df_req['EmpCount'].fillna(Emp_Count, inplace=True)
    # df_req['FamilyTreeHierarchyLevel'].fillna(FamilyTreeHierarchy_Level, inplace=True)
   df_req['GlobalUltimateFamilyTreeLinkageCount'].fillna(GlobalUltimateFamilyTreeLinkage_Count, inplace=True)
   df_req['TotalAssetsAmount'].fillna(TotalAssets_Amount, inplace=True)
   # df_req['TotalLiabilitiesAmount'].fillna(TotalLiabilities_Amount, inplace=True)
   # df_req['TotalCurrentAssetsAmount'].fillna(TotalCurrentAssets_Amount, inplace=True)
   # df_req['TotalCurrentLiabilitiesAmount'].fillna(TotalCurrentLiabilities_Amount, inplace=True)
   df_req['NetWorth'].fillna(Net_Worth, inplace=True)

   
   ## Treating the null values for categorical data 

   df_req['PrimaryTownName'].fillna('LONDON', inplace=True)
   df_req['LineOfBusinessDescription'].fillna('Business services', inplace=True)
   

   ## Converting categorical data into numerical values
   
   df_req['PrimaryTownName'] = df_req['PrimaryTownName'].map(params['PrimaryTownEncoding'])
   df_req['LineOfBusinessDescription'] = df_req['LineOfBusinessDescription'].map(params['BizDescEncoding'])



    ## Save the preprocessed data as a csv file

   


   # artifacts = config['artifacts']
   # output_file = os.path.join(artifacts['artifacts_directory'], artifacts['preprocessed_data_dir'])
   # create_directories([output_file])
   # prepared_data_path = os.path.join(output_file, artifacts["preprocessed_data_file"])
   # df_req.to_csv(prepared_data_path, index=False) 


   #  # Read the processed data from database

   # input_df = pd.read_csv(prepared_data_path, sep=",")

 

    ## Split the data into train and test sets

   test_split_ratio = params["ModelTraining"]["test_split"]
   val_split_ratio = params["ModelTraining"]["val_split"]
   seed = params["ModelTraining"]["seed"]

   mlflow.log_params(params["ModelTraining"])

    # Performing Train-test Split

   train_val, test = train_test_split(df_req, test_size=test_split_ratio, random_state=seed)

   train, val = train_test_split(train_val, test_size=val_split_ratio, random_state=seed)

    
    ## Storing the train, val and test data sets

   artifacts = config['artifacts']
   train_val_test_data_directory = os.path.join(artifacts['artifacts_directory'], artifacts['train_val_test_data_directory'])
   create_directories([train_val_test_data_directory])

   train_data_path = os.path.join(train_val_test_data_directory, artifacts["train_data_file"])
   train.to_csv(train_data_path, index=False)

   val_data_path = os.path.join(train_val_test_data_directory, artifacts["val_data_file"])
   val.to_csv(val_data_path, index=False)

   test_data_path = os.path.join(train_val_test_data_directory, artifacts["test_data_file"])
   test.to_csv(test_data_path, index=False)
    


if __name__ == '__main__':
   args = argparse.ArgumentParser()
   args.add_argument("--config", "-c", default="configs/config.yaml")
   args.add_argument("--params", "-p", default="params.yaml")
   parsed_args = args.parse_args()

   try:
      logging.info("\n********************")
      logging.info(f">>>>> stage {STAGE} started <<<<<")
      main(config_path=parsed_args.config, params_path=parsed_args.params)
      logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
   except Exception as e:
      logging.exception(e)
      raise e
