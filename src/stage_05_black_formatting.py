import black
from pathlib import Path
import argparse
import logging
from black import FileMode, format_file_contents, format_file_in_place, diff, WriteBack

STAGE = "STAGE_05_BLACK_FORMATTING"  ## <<< change stage name


def main(config_path, params_path):

    stage_01 = "src/stage_01_dataprepare.py"
    stage_02 = "src/stage_02_modeltraining.py"
    stage_03 = "src/stage_03_modelevaluation.py"
    stage_04a = "src/stage_04a_unittesting.py"
    stage_04b = "src/stage_04b_integrationtesting.py"
    stage_05 = "src/stage_05_black_formatting.py"
    stage_06 = "src/stage_06_vulture_stdcoding.py"

    stages = [stage_01, stage_02, stage_03, stage_04a, stage_04b, stage_05, stage_06]

    files_reformatted = 0

    for stage in stages:
        path = Path(stage)
        format_file_result = format_file_in_place(
            path, fast=False, mode=FileMode(), write_back=WriteBack.YES
        )
        if format_file_result == True:
            files_reformatted += 1
            print(stage, "reformatted")

    print("Files reformatted: ", files_reformatted)


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
