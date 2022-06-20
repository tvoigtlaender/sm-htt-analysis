#!/usr/bin/env python

import argparse
import logging
logger = logging.getLogger("sum_training_weights")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

import yaml
import os


def parse_arguments():
    parser = argparse.ArgumentParser(description="Sum training weights of classes in training dataset.")
    parser.add_argument("--input-base-path", required=False, help="Base path of inputs.")
    parser.add_argument("--output-dir", type=str,required=True, help="Output directory of this script")
    return parser.parse_args()

def main(args):
    eras = ["2016", "2017", "2018"]
    configs = []
    # Load training config files for all eras
    for era in eras:
        config_path = "{}/{}/training_config.yaml".format(args.input_base_path, era)
        logger.info("Try to open {}".format(config_path))
        config = yaml.load(open(config_path, 'r'), Loader =yaml.SafeLoader)
        configs.append(config)


    all_era_template = {}
    for key in configs[0]:
        if key == "class_weights":
            # Gather all class weights
            for i_era, era in enumerate(eras):
                all_era_template["class_weights_{}".format(era)] = configs[i_era]["class_weights"]
        else:
            # Use configs of first era
            all_era_template[key] = configs[0][key]

    if not os.path.exists(args.output_dir + "/all_eras"):
        os.mkdir(args.output_dir + "/all_eras")

    output_file = args.output_dir + "/all_eras/training_config.yaml"

    logger.info("Writing new dataset config for all eras to {}".format(output_file))
    # Save merged config files
    yaml.dump(all_era_template, open(output_file, 'w'), default_flow_style=False)

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
