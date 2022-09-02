import argparse
import logging as log
log.basicConfig(
    format="Tensorflow_training - %(levelname)s - %(message)s", level=log.DEBUG
)
from ml_trainings.Config_merger import get_merged_config
import pickle
import yaml
import json
# import copy

from tensorflow.keras.models import load_model

def parse_arguments():
    parser = argparse.ArgumentParser(description="Export model for Htt analyses to .json format")
    parser.add_argument("--training-config", help="Path to training config file")
    parser.add_argument("--training-name", help="Name of training")
    parser.add_argument("--in-out-dir", help="Input and output directory of converter")
    parser.add_argument("--fold", help="Fold of training")
    return parser.parse_args()


def parse_config(file, name):
    with open(file, "r") as stream:
        config = yaml.safe_load(stream)
    training_config = get_merged_config(config, name)
    return training_config

def main(args, training_config):
    
    ids = list(training_config["parts"].keys())
    num_id_inputs = len(ids) if len(ids) > 1 else 0
    processes = training_config["processes"]
    classes = training_config["classes"]
    variables = training_config["variables"]
    var_and_id = variables + (ids if num_id_inputs else [])
    weight_var = "weight"
    fold = args.fold
    log.debug("Used identifiers: {}".format(ids))
    log.debug("Used processes: {}".format(processes))
    log.debug("Used classes: {}".format(classes))
    log.debug("Used variables: {}".format(variables))
    log.debug("Used var + id: {}".format(var_and_id))
    log.debug("Used fold: {}".format(fold))

    # Create template dictionary for variables.json
    variables_template = {
        "class_labels" : classes,
        "inputs" : [{"name" : variable, "offset" : 0.0, "scale" : 1.0} for variable in var_and_id]
    } 
    print(variables_template)
    work_dir = args.in_out_dir
    classifier_template = "{dir}/fold{fold}_keras_model.h5"
    weights_template = "{dir}/fold{fold}_keras_weights.h5"
    preprocessing_template = "{dir}/fold{fold}_keras_preprocessing.pickle"
    architecture_template = "{dir}/fold{fold}_keras_architecture.json"
    variable_exports_template = "{dir}/fold{fold}_keras_variables.json"

    # Load keras model and preprocessing
    model = load_model(classifier_template.format(dir=work_dir, fold=fold))
    with open(preprocessing_template.format(dir=work_dir, fold=fold), "rb") as f:
        scaler = pickle.load(f, encoding="bytes")
    # export weights in .h5 format & model in .json format
    model.save_weights(weights_template.format(dir=work_dir, fold=fold))
    with open(architecture_template.format(dir=work_dir, fold=fold), "w") as f:
        f.write(model.to_json())
    # export scale & offsets vor variables
    tmp_variables = variables_template
    for variable,offset,scale in zip(tmp_variables["inputs"],scaler.mean_,scaler.scale_): # NOTE: offsets & scales are in the same order as in config_training["variables"]
        variable["offset"] = -offset
        variable["scale"] = 1.0/scale
    with open(variable_exports_template.format(dir=work_dir, fold=fold), "w") as f:
        f.write(json.dumps(tmp_variables))

if __name__ == "__main__":
    args = parse_arguments()
    training_config = parse_config(args.training_config, args.training_name)
    main(args, training_config)
