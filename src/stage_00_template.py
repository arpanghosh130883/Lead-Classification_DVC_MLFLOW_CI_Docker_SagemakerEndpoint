import pandas as pd
import argparse
import os
import numpy as np
import pickle
import logging
from src.utils.common import read_yaml, create_directories
import shap


STAGE = "STAGE_01_DATA_PREPARATION" 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def main(config_path, params_path):

    

    ## read config files
    config = read_yaml(config_path)
    params = read_yaml(params_path)
    
    ## read data from source folder
    source_data = config["source_data"]
    input_data = os.path.join(source_data["data_directory"], source_data["new_data_file"])

    ## read input data as a dataframe
    df = pd.read_csv(input_data, sep=",")
   

    #Renaming the columns as per standard
    df.rename(columns = {'profitorlossamount':'ProfitOrLossAmount','familytreehierarchylevel':'FamilyTreeHierarchyLevel', 'globalultimatefamilytreelinkagecount':'GlobalUltimateFamilyTreeLinkageCount','totalliabilitiesamount':'TotalLiabilitiesAmount','totalcurrentliabilitiesamount':'TotalCurrentLiabilitiesAmount', 'totalcurrentassetsamount':'TotalCurrentAssetsAmount', 'emp_count':'EmpCount','networth':'NetWorth', 'totalassetsamount':'TotalAssetsAmount','sales_turnover_gbp':'SalesTurnoverGBP', 'lineofbusinessdescription':'LineOfBusinessDescription','primarytownname':'PrimaryTownName', 'org':'Org', 'import_true':'ImportTrue', 'export_true':'ExportTrue', 'standalone_org':'StandAloneOrg', 'ownership_type':'OwnershipType', 'familytreememberroletext':'FamilyTreeMemberRoleText', 'minorityownedindicator':'MinorityOwnedIndicator', 'foreignindicator':'ForeignIndicator', 'operatingstatus':'OperatingStatus'}, inplace =True)



    # Create null availability columns
    df['SalesTurnoverAvailable'] = np.where(df['SalesTurnoverGBP'].isnull(),0,1)
    df['ProfitOrLossAmountAvailable'] = np.where(df['ProfitOrLossAmount'].isnull(),0,1)
    df['EmpCountAvailable'] = np.where(df['EmpCount'].isnull(),0,1)
    df['FamilyTreeHierarchyLevelAvailable'] = np.where(df['FamilyTreeHierarchyLevel'].isnull(),0,1)
    df['GlobalUltimateFamilyTreeLinkageCountAvailable'] = np.where(df['GlobalUltimateFamilyTreeLinkageCount'].isnull(),0,1)
    df['TotalAssetsAmountAvailable'] = np.where(df['TotalAssetsAmount'].isnull(),0,1)
    df['TotalLiabilitiesAmountAvailable'] = np.where(df['TotalLiabilitiesAmount'].isnull(),0,1)
    df['TotalCurrentAssetsAmountAvailable'] = np.where(df['TotalCurrentAssetsAmount'].isnull(),0,1)
    df['TotalCurrentLiabilitiesAmountAvailable'] = np.where(df['TotalCurrentLiabilitiesAmount'].isnull(),0,1)
    df['NetworthAvailable'] = np.where(df['NetWorth'].isnull(),0,1)



    # Get the filling values for numeric data    
    SalesTurnover_GBP = params['Numeric']['SalesTurnoverGBP']
    ProfitOrLoss_Amount = params['Numeric']['ProfitOrLossAmount']
    Emp_Count = params['Numeric']['EmpCount']
    FamilyTreeHierarchy_Level = params['Numeric']['FamilyTreeHierarchyLevel']
    GlobalUltimateFamilyTreeLinkage_Count = params['Numeric']['GlobalUltimateFamilyTreeLinkageCount']
    TotalAssets_Amount = params['Numeric']['TotalAssetsAmount']
    Net_Worth = params['Numeric']['NetWorth']



    # Fill the null values for numeric data
    df['SalesTurnoverGBP'].fillna(SalesTurnover_GBP, inplace=True)
    df['ProfitOrLossAmount'].fillna(ProfitOrLoss_Amount, inplace=True)
    df['EmpCount'].fillna(Emp_Count, inplace=True)
    df['FamilyTreeHierarchyLevel'].fillna(FamilyTreeHierarchy_Level, inplace=True)
    df['GlobalUltimateFamilyTreeLinkageCount'].fillna(GlobalUltimateFamilyTreeLinkage_Count, inplace=True)
    df['TotalAssetsAmount'].fillna(TotalAssets_Amount, inplace=True)
    df['NetWorth'].fillna(Net_Worth, inplace=True)
    


    # Fill null values for categorical variables
    df['OwnershipType'] = df['OwnershipType'].fillna('Privately owned')  
    df['FamilyTreeMemberRoleText'] = df['FamilyTreeMemberRoleText'].fillna('Unknown')
    df['overseas_countries'] = df['overseas_countries'].fillna('uk')
    df['LineOfBusinessDescription'] = df['LineOfBusinessDescription'].fillna('Business services')
    df['PrimaryTownName'] = df['PrimaryTownName'].fillna('LONDON')
    df['Org'] = df['Org'].fillna('Currencies Direct')
    df['OperatingStatus'] = df['OperatingStatus'].fillna('Active')



    # Fill null values for binary variables
    df['StandAloneOrg'] = df['StandAloneOrg'].fillna(True)
    df['ImportTrue'] = df['ImportTrue'].fillna(False)
    df['ExportTrue'] = df['ExportTrue'].fillna(False)
    df['MinorityOwnedIndicator'] = df['MinorityOwnedIndicator'].fillna(False)
    df['ForeignIndicator'] = df['ForeignIndicator'].fillna(False)



    # Convert categorical/binary values to numeric
    df['OverseasCountriesKnown'] = np.where(df['overseas_countries'] == 'uk', 0,1)
    df['StandaloneOrgTrue'] = np.where(df.StandAloneOrg == True,1,0)


    ## Converting categorical data into numerical values

    df.Org = df.Org.map(params['OrgMap'])
    df.OwnershipType = df.OwnershipType.map(params['OwnershipType'])
    df.OperatingStatus = df.OperatingStatus.map(params['OperatingStatus'])
    df.FamilyTreeMemberRoleText = df.FamilyTreeMemberRoleText.map(params['FamilyTreeMemberRoleText'])
    df.PrimaryTownName = df.PrimaryTownName.map(params['PrimaryTownEncoding'])
    df.LineOfBusinessDescription = df.LineOfBusinessDescription.map(params['BizDescEncoding'])

    ## After encoding, there will be a null value for any new value in the test data that has not been witnessed in training data,
    # this null value will have to be treated and encoded again. 

    # Fill null values for categorical variables after encoding
    df['OwnershipType'] = df['OwnershipType'].fillna('Privately owned')  
    df['FamilyTreeMemberRoleText'] = df['FamilyTreeMemberRoleText'].fillna('Unknown')
    df['overseas_countries'] = df['overseas_countries'].fillna('uk')
    df['LineOfBusinessDescription'] = df['LineOfBusinessDescription'].fillna('Business services')
    df['PrimaryTownName'] = df['PrimaryTownName'].fillna('LONDON')
    df['Org'] = df['Org'].fillna('Currencies Direct')
    df['OperatingStatus'] = df['OperatingStatus'].fillna('Active')

    ## Encode again
    df.Org = df.Org.map(params['OrgMap'])
    df.OwnershipType = df.OwnershipType.map(params['OwnershipType'])
    df.OperatingStatus = df.OperatingStatus.map(params['OperatingStatus'])
    df.FamilyTreeMemberRoleText = df.FamilyTreeMemberRoleText.map(params['FamilyTreeMemberRoleText'])
    df.PrimaryTownName = df.PrimaryTownName.map(params['PrimaryTownEncoding'])
    df.LineOfBusinessDescription = df.LineOfBusinessDescription.map(params['BizDescEncoding'])



    #Taking required columns
    df_req = df[['Org', 'OwnershipType', 'OperatingStatus', 'SalesTurnoverGBP',
       'ProfitOrLossAmount', 'ImportTrue', 'ExportTrue',
       'LineOfBusinessDescription', 'EmpCount', 'MinorityOwnedIndicator',
       'FamilyTreeHierarchyLevel', 'GlobalUltimateFamilyTreeLinkageCount',
       'FamilyTreeMemberRoleText', 'ForeignIndicator', 'TotalAssetsAmount',
        'PrimaryTownName', 'NetWorth',
       'SalesTurnoverAvailable', 'ProfitOrLossAmountAvailable',
       'EmpCountAvailable', 'FamilyTreeHierarchyLevelAvailable',
       'GlobalUltimateFamilyTreeLinkageCountAvailable',
       'TotalAssetsAmountAvailable', 'NetworthAvailable',
       'OverseasCountriesKnown', 'StandaloneOrgTrue']]

   


    # Save the preprocessed data as a csv file

    artifacts = config['artifacts']
    output_file = os.path.join(artifacts['artifacts_directory'], artifacts['preprocessed_data_dir'])
    create_directories([output_file])
    prepared_data_path = os.path.join(output_file, artifacts["preprocessed_data_file"])
    df_req.to_csv(prepared_data_path, index=False) 



    
    ## Load the scaler pickle file 

    scaler_path = config['scaler']
    scaler = os.path.join(scaler_path['scaler_dir'], scaler_path['scaler_file'])

    load_scaler = pickle.load(open(scaler,'rb'))

    ## Scale the input data

    df_req_scaled = (load_scaler.transform(df_req))

 


    ## Load the trained model

    model_path = config['model']
    model = os.path.join(model_path['model_dir'], model_path['model_file'])

    load_model = pickle.load(open(model,'rb'))


    # Predict the outcome of input lead 

    prediction = load_model.predict(df_req_scaled)
    print('predicted value', prediction)


    ## Check the features contributing to the output of a single instance
    # input_instance = df_req_scaled  

    explainer = shap.TreeExplainer(load_model)
    shap_values = explainer.shap_values(df_req_scaled)

    features = ['Org',
    'OwnershipType',
    'OperatingStatus',
    'SalesTurnoverGBP',
    'ProfitOrLossAmount',
    'ImportTrue',
    'ExportTrue',
    'LineOfBusinessDescription',
    'EmpCount',
    'MinorityOwnedIndicator',
    'FamilyTreeHierarchyLevel',
    'GlobalUltimateFamilyTreeLinkageCount',
    'FamilyTreeMemberRoleText',
    'ForeignIndicator',
    'TotalAssetsAmount',
    'PrimaryTownName',
    'NetWorth',
    'OverseasCountriesKnown',
    'StandaloneOrgTrue']

    shap_sum = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame([features, shap_sum.tolist()]).T
    importance_df.columns = ['column_name', 'importance_score']
    importance_df = importance_df.sort_values('importance_score', ascending=False)
    print(importance_df[importance_df.column_name.notnull()].head())



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

