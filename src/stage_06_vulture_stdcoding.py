import vulture
from pathlib import Path
import os
import argparse
import logging
from src.utils.common import read_yaml, create_directories

STAGE = "STAGE_06_VULTURE_STD_CODING"  ## <<< change stage name


def main(config_path, params_path):

    stage_01 = "src/stage_01_dataprepare.py"
    stage_02 = "src/stage_02_modeltraining.py"
    stage_03 = "src/stage_03_modelevaluation.py"
    stage_04a = "src/stage_04a_unittesting.py"
    stage_04b = "src/stage_04b_integrationtesting.py"
    stage_05 = "src/stage_05_black_formatting.py"
    stage_06 = "src/stage_06_vulture_stdcoding.py"

    stages = [stage_01, stage_02, stage_03, stage_04a, stage_04b, stage_05, stage_06]

    v = vulture.Vulture()

    for stage in stages:
        path = os.path.normcase(stage)
        v.scavenge([path])

        for item in v.get_unused_code():
            print(item.filename, item.name, item.typ, item.first_lineno)


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
