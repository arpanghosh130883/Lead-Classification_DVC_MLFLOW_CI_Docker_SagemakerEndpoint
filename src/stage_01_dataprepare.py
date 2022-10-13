import pandas as pd
import argparse
import os
import numpy as np
from sklearn.model_selection import train_test_split
import mlflow
import logging
from src.utils.common import read_yaml, create_directories
import dvc.api
from imblearn.over_sampling import SMOTE

"""
path = 'dvc_data/raw_data.csv'
repo = 'C:/Users/Aishwarya.Chandra/SelfDevelopment/mlflow_dvc_trial'
version = 'v1'

# data_url = dvc.api.get_url(path = path,   repo=repo)
data_url = dvc.api.get_url(path = path,   repo=repo,   rev=version)
"""


STAGE = "STAGE_01_DATA_PREPARATION"  ## <<< change stage name

logging.basicConfig(
    filename=os.path.join("logs", "running_logs.log"),
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a",
)


def main(config_path, params_path):

    ## read config files
    config = read_yaml(config_path)
    params = read_yaml(params_path)

    ## read data from source folder

    source_data = config["source_data"]
    input_data = os.path.join(source_data["data_directory"], source_data["data_file"])

    # version = os.path.join(source_data["version"])
    # path = os.path.join(source_data["path"])
    # repo = os.path.join(source_data["repo"])

    # data_url = dvc.api.get_url(path=path, repo=repo, rev=version)

    ## read input data as a dataframe
    df = pd.read_csv(input_data, sep=",")

    # mlflow.log_param("data_url", data_url)
    # mlflow.log_param("data_version", version)

    input_features = pd.DataFrame(list(df.columns))

    artifacts = config["artifacts"]
    output_file = os.path.join(
        artifacts["artifacts_directory"], artifacts["input_features_dir"]
    )
    create_directories([output_file])
    input_features_path = os.path.join(output_file, artifacts["input_features_file"])
    input_features.to_csv(input_features_path, header=False, index=False)

    mlflow.log_artifact(input_features_path)

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Create target column
    df["IsClientConverted"] = np.where(df["acc_status"] == "Marketing", 0, 1)

    # Renaming the columns as per standard
    df.rename(
        columns={
            "org": "Org",
            "Sales_turnover_GBP": "SalesTurnoverGBP",
            "profitorlossamount": "ProfitOrLossAmount",
            "familytreehierarchylevel": "FamilyTreeHierarchyLevel",
            "globalultimatefamilytreelinkagecount": "GlobalUltimateFamilyTreeLinkageCount",
            "totalliabilitiesamount": "TotalLiabilitiesAmount",
            "totalcurrentliabilitiesamount": "TotalCurrentLiabilitiesAmount",
            "totalcurrentassetsamount": "TotalCurrentAssetsAmount",
            "Emp count": "EmpCount",
            "emp_count": "EmpCount",
            "networth": "NetWorth",
            "Networth": "NetWorth",
            "Ownership_type": "OwnershipType",
            "totalassetsamount": "TotalAssetsAmount",
            "sales_turnover_gbp": "SalesTurnoverGBP",
            "lineofbusinessdescription": "LineOfBusinessDescription",
            "primarytownname": "PrimaryTownName",
            "org": "Org",
            "import_true": "ImportTrue",
            "export_true": "ExportTrue",
            "standalone_org": "StandAloneOrg",
            "Standalone_org": "StandAloneOrg",
            "ownership_type": "OwnershipType",
            "familytreememberroletext": "FamilyTreeMemberRoleText",
            "minorityownedindicator": "MinorityOwnedIndicator",
            "foreignindicator": "ForeignIndicator",
            "operatingstatus": "OperatingStatus",
        },
        inplace=True,
    )

    # Create null availability columns
    df["SalesTurnoverAvailable"] = np.where(df["SalesTurnoverGBP"].isnull(), 0, 1)
    df["ProfitOrLossAmountAvailable"] = np.where(
        df["ProfitOrLossAmount"].isnull(), 0, 1
    )
    df["EmpCountAvailable"] = np.where(df["EmpCount"].isnull(), 0, 1)
    df["FamilyTreeHierarchyLevelAvailable"] = np.where(
        df["FamilyTreeHierarchyLevel"].isnull(), 0, 1
    )
    df["GlobalUltimateFamilyTreeLinkageCountAvailable"] = np.where(
        df["GlobalUltimateFamilyTreeLinkageCount"].isnull(), 0, 1
    )
    df["TotalAssetsAmountAvailable"] = np.where(df["TotalAssetsAmount"].isnull(), 0, 1)
    df["TotalLiabilitiesAmountAvailable"] = np.where(
        df["TotalLiabilitiesAmount"].isnull(), 0, 1
    )
    df["TotalCurrentAssetsAmountAvailable"] = np.where(
        df["TotalCurrentAssetsAmount"].isnull(), 0, 1
    )
    df["TotalCurrentLiabilitiesAmountAvailable"] = np.where(
        df["TotalCurrentLiabilitiesAmount"].isnull(), 0, 1
    )
    df["NetworthAvailable"] = np.where(df["NetWorth"].isnull(), 0, 1)

    # Get the filling values for numeric data
    SalesTurnover_GBP = params["Numeric"]["SalesTurnoverGBP"]
    ProfitOrLoss_Amount = params["Numeric"]["ProfitOrLossAmount"]
    Emp_Count = params["Numeric"]["EmpCount"]
    FamilyTreeHierarchy_Level = params["Numeric"]["FamilyTreeHierarchyLevel"]
    GlobalUltimateFamilyTreeLinkage_Count = params["Numeric"][
        "GlobalUltimateFamilyTreeLinkageCount"
    ]
    TotalAssets_Amount = params["Numeric"]["TotalAssetsAmount"]
    Net_Worth = params["Numeric"]["NetWorth"]

    # Fill the null values for numeric data
    df["SalesTurnoverGBP"].fillna(SalesTurnover_GBP, inplace=True)
    df["ProfitOrLossAmount"].fillna(ProfitOrLoss_Amount, inplace=True)
    df["EmpCount"].fillna(Emp_Count, inplace=True)
    df["FamilyTreeHierarchyLevel"].fillna(FamilyTreeHierarchy_Level, inplace=True)
    df["GlobalUltimateFamilyTreeLinkageCount"].fillna(
        GlobalUltimateFamilyTreeLinkage_Count, inplace=True
    )
    df["TotalAssetsAmount"].fillna(TotalAssets_Amount, inplace=True)
    df["NetWorth"].fillna(Net_Worth, inplace=True)
    # print(df.isnull().sum())

    # Fill null values for categorical variables
    df["OwnershipType"] = df["OwnershipType"].fillna("Privately owned")
    df["FamilyTreeMemberRoleText"] = df["FamilyTreeMemberRoleText"].fillna("Unknown")
    df["overseas_countries"] = df["overseas_countries"].fillna("uk")
    df["LineOfBusinessDescription"] = df["LineOfBusinessDescription"].fillna(
        "Business services"
    )
    df["PrimaryTownName"] = df["PrimaryTownName"].fillna("LONDON")
    df["Org"] = df["Org"].fillna("Currencies Direct")
    df["OperatingStatus"] = df["OperatingStatus"].fillna("Active")

    # Fill null values for binary variables
    df["StandAloneOrg"] = df["StandAloneOrg"].fillna(1)
    df["ImportTrue"] = df["ImportTrue"].fillna(0)
    df["ExportTrue"] = df["ExportTrue"].fillna(0)
    df["MinorityOwnedIndicator"] = df["MinorityOwnedIndicator"].fillna(0)
    df["ForeignIndicator"] = df["ForeignIndicator"].fillna(0)

    # Convert categorical/binary values to numeric
    df["OverseasCountriesKnown"] = np.where(df["overseas_countries"] == "uk", 0, 1)
    df["StandaloneOrgTrue"] = np.where(df.StandAloneOrg == True, 1, 0)
    df["ImportTrue"] = np.where(df.ImportTrue == True, 1, 0)
    df["ExportTrue"] = np.where(df.ExportTrue == True, 1, 0)
    df["MinorityOwnedIndicator"] = np.where(df.MinorityOwnedIndicator == True, 1, 0)
    df["ForeignIndicator"] = np.where(df.ForeignIndicator == True, 1, 0)

    ## Converting categorical data into numerical values

    df.Org = df.Org.map(params["OrgMap"])
    df.OwnershipType = df.OwnershipType.map(params["OwnershipType"])
    df.OperatingStatus = df.OperatingStatus.map(params["OperatingStatus"])
    df.FamilyTreeMemberRoleText = df.FamilyTreeMemberRoleText.map(
        params["FamilyTreeMemberRoleText"]
    )
    df.PrimaryTownName = df.PrimaryTownName.map(params["PrimaryTownEncoding"])
    df.LineOfBusinessDescription = df.LineOfBusinessDescription.map(
        params["BizDescEncoding"]
    )

    ## After encoding, there will be a null value for any new value in the test data that has not been witnessed in training data,
    # this null value will have to be treated and encoded again.

    privately_owned = params["OwnershipType"]["Privately owned"]
    unknown = params["FamilyTreeMemberRoleText"]["Unknown"]
    business_services = params["BizDescEncoding"]["Business services"]
    london = params["PrimaryTownEncoding"]["LONDON"]
    cd = params["OrgMap"]["Currencies Direct"]
    active = params["OperatingStatus"]["Active"]

    # # Fill null values for categorical variables after encoding
    df["OwnershipType"] = df["OwnershipType"].fillna(privately_owned)
    df["FamilyTreeMemberRoleText"] = df["FamilyTreeMemberRoleText"].fillna(unknown)
    df["overseas_countries"] = df["overseas_countries"].fillna(0)
    df["LineOfBusinessDescription"] = df["LineOfBusinessDescription"].fillna(
        business_services
    )
    df["PrimaryTownName"] = df["PrimaryTownName"].fillna(london)
    df["Org"] = df["Org"].fillna(cd)
    df["OperatingStatus"] = df["OperatingStatus"].fillna(active)

    # Taking required columns
    df_req = df[
        [
            "Org",
            "OwnershipType",
            "OperatingStatus",
            "SalesTurnoverGBP",
            "ProfitOrLossAmount",
            "ImportTrue",
            "ExportTrue",
            "LineOfBusinessDescription",
            "EmpCount",
            "MinorityOwnedIndicator",
            "FamilyTreeHierarchyLevel",
            "GlobalUltimateFamilyTreeLinkageCount",
            "FamilyTreeMemberRoleText",
            "ForeignIndicator",
            "TotalAssetsAmount",
            "PrimaryTownName",
            "NetWorth",
            "SalesTurnoverAvailable",
            "ProfitOrLossAmountAvailable",
            "EmpCountAvailable",
            "FamilyTreeHierarchyLevelAvailable",
            "GlobalUltimateFamilyTreeLinkageCountAvailable",
            "TotalAssetsAmountAvailable",
            "NetworthAvailable",
            "OverseasCountriesKnown",
            "StandaloneOrgTrue",
            "IsClientConverted",
        ]
    ]

    mlflow.log_param("input_rows", df_req.shape[0])

    ## Save the preprocessed data as a csv file

    artifacts = config["artifacts"]
    output_file = os.path.join(
        artifacts["artifacts_directory"], artifacts["processed_data_dir"]
    )
    create_directories([output_file])
    prepared_data_path = os.path.join(output_file, artifacts["processed_data_file"])
    df_req.to_csv(prepared_data_path, index=False)

    y_df = df_req["IsClientConverted"]
    x_df = df_req.drop(["IsClientConverted"], axis=1)

    mlflow.log_param("input_columns", x_df.shape[1])

    ## Apply smote
    oversample = SMOTE()
    x_balanced, y_balanced = oversample.fit_resample(x_df, y_df)

    ## Split the data into train and test sets

    test_split_ratio = params["ModelTraining"]["test_split"]
    val_split_ratio = params["ModelTraining"]["val_split"]
    seed = params["ModelTraining"]["seed"]

    mlflow.log_params(params["ModelTraining"])

    # Performing Train-test Split

    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x_balanced, y_balanced, test_size=test_split_ratio, random_state=seed
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val, y_train_val, test_size=val_split_ratio, random_state=seed
    )

    ## Storing the train, val and test data sets

    artifacts = config["artifacts"]
    train_val_test_data_directory = os.path.join(
        artifacts["artifacts_directory"], artifacts["train_val_test_data_directory"]
    )
    create_directories([train_val_test_data_directory])

    x_train_data_path = os.path.join(
        train_val_test_data_directory, artifacts["x_train_data_file"]
    )
    x_train.to_csv(x_train_data_path, index=False)

    y_train_data_path = os.path.join(
        train_val_test_data_directory, artifacts["y_train_data_file"]
    )
    y_train.to_csv(y_train_data_path, index=False)

    x_val_data_path = os.path.join(
        train_val_test_data_directory, artifacts["x_val_data_file"]
    )
    x_val.to_csv(x_val_data_path, index=False)

    y_val_data_path = os.path.join(
        train_val_test_data_directory, artifacts["y_val_data_file"]
    )
    y_val.to_csv(y_val_data_path, index=False)

    x_test_data_path = os.path.join(
        train_val_test_data_directory, artifacts["x_test_data_file"]
    )
    x_test.to_csv(x_test_data_path, index=False)

    y_test_data_path = os.path.join(
        train_val_test_data_directory, artifacts["y_test_data_file"]
    )
    y_test.to_csv(y_test_data_path, index=False)


if __name__ == "__main__":
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
