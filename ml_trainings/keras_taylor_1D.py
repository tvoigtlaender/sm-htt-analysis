#!/usr/bin/env python

import logging as log
log.basicConfig(
    format="Tensorflow_training - %(levelname)s - %(message)s", level=log.INFO
)
import argparse
import os
import yaml
import pickle
import uproot
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from ml_trainings.Config_merger import get_merged_config
import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['font.size'] = 16
import matplotlib.pyplot as plt


def parse_arguments():
    parser = argparse.ArgumentParser(description="Calculate 1D taylor coefficients.")
    parser.add_argument("--config-file", help="Path to training config file")
    parser.add_argument("--training-name", help="Name of training")
    parser.add_argument("--model-dir", help="Dir of trained ML model")
    parser.add_argument("--data-dir", help="Dir of process datasets")
    parser.add_argument("--output-dir", help="Output directory of training")
    parser.add_argument(
        "--num-events", help="Number of events in one chunk", 
        default="10 MB",
    )
    parser.add_argument(
        "--no-abs",
        action="store_true",
        default=False,
        help="Do not use abs for metric.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        default=False,
        help="Normalize rows.",
    )
    return parser.parse_args()

def parse_config(file, name):
    with open(file, "r") as stream:
        config = yaml.safe_load(stream)
    training_config = get_merged_config(config, name)
    return training_config

def create_plots(vars_, classes, fold, grad_mat, name):
    log.info("Write 1D taylor plot for {} fold {}.".format(name, fold))
    plt.figure(0, figsize=(len(vars_), len(classes)))
    axis = plt.gca()
    for i in range(grad_mat.shape[0]):
        for j in range(grad_mat.shape[1]):
            axis.text(j + 0.5,
                      i + 0.5,
                      '{:.2f}'.format(grad_mat[i, j]),
                      ha='center',
                      va='center')
    q = plt.pcolormesh(grad_mat, cmap='Wistia')
    #cbar = plt.colorbar(q)
    #cbar.set_label("mean(abs(Taylor coefficients))", rotation=270, labelpad=20)
    plt.xticks(np.array(range(len(vars_))) + 0.5,
               vars_,
               rotation='vertical')
    plt.yticks(np.array(range(len(classes))) + 0.5,
               classes,
               rotation='horizontal')
    plt.xlim(0, len(vars_))
    plt.ylim(0, len(classes))
    output_path = os.path.join(
        args.output_dir, 
        "fold{}_keras_taylor_1D_{}".format(fold, name)
    )
    log.info("Save plot to {}.".format(output_path))
    plt.savefig(output_path + ".png", bbox_inches='tight')
    plt.savefig(output_path + ".pdf", bbox_inches='tight')
    plt.close()



# Function to compute gradients of model answers in optimized graph mode
@tf.function(experimental_relax_shapes=True)
def get_gradients(model, samples, output_ind):
    # Define function to get single gradient
    def get_single_gradient(single_sample):
        # (Expand shape of sample for gradient function)
        single_sample = tf.expand_dims(single_sample, axis=0)
        with tf.GradientTape() as tape:
            tape.watch(single_sample)
            # Get response from model with (only choosen output class)
            response = model(single_sample, training=False)[:, output_ind]
        # Get gradient of choosen output class wrt. input sample
        grad = tape.gradient(response, single_sample)
        return grad

    # Apply function to every sample to get all gradients
    grads = tf.vectorized_map(get_single_gradient, samples)
    return grads

def main(rgs, training_config):
    ids = list(training_config["parts"].keys())
    num_id_inputs = len(ids) if len(ids) > 1 else 0
    processes = training_config["processes"]
    classes = training_config["classes"]
    variables = training_config["variables"]
    weight_var = "weight"
    folds = [0, 1]
    inverse_folds = [1, 0]
    log.debug("Used identifiers: {}".format(ids))
    log.debug("Used processes: {}".format(processes))
    log.debug("Used classes: {}".format(classes))
    log.debug("Used variables: {}".format(variables))
    log.info("Open {} events at once.".format(args.num_events))
    for fold in folds:
        # Load scaler
        preprocessing_path = os.path.join(
            args.model_dir,
            "fold{fold}_keras_preprocessing.pickle".format(fold=fold)
        )
        log.info("Load preprocessing {}.".format(preprocessing_path))
        with open(preprocessing_path, "rb") as stream:
            scaler = pickle.load(stream , encoding="bytes")
        # Load trained model
        model_path = os.path.join(
            args.model_dir,
            "fold{fold}_keras_model.h5".format(fold=fold),
        )
        log.info("Load keras model {}.".format(model_path))
        model = load_model(model_path)
        all_gradient_weights = {}
        all_gradients = {}
        for i_id, id_ in enumerate(ids):
            gradients_intermediate = {}
            gradient_weights_intermediate = {}
            for class_ in classes:
                gradients_intermediate[class_] = np.zeros(len(variables) + num_id_inputs)
                gradient_weights_intermediate[class_] = 0
            for process in processes:
                # Load datashard for this process
                mapped_class = training_config["mapping"][process]
                # Get index of mapped training class
                i_class = classes.index(mapped_class)
                file_path = os.path.join(
                    args.data_dir,
                    "{id}_{pr}_{t_c}_datashard_fold{fold}.root".format(
                        id=id_,
                        pr=process,
                        t_c=mapped_class,
                        fold=inverse_folds[fold],
                    ),
                )
                log.debug("Reading {}".format(file_path))
                # Get input data from files
                file_class = uproot.open(file_path).keys()[0].split(";")[0]
                if mapped_class != file_class:
                    log.error(
                        "Class mapped by the config file and present in the "
                        "datashard do not match for {}: {} and {}".format(
                            file_path, mapped_class, file_class
                        )
                    )
                    raise Exception("Consistency error in Tensorflow training.")
                N_entries = uproot.open(file_path)[mapped_class].num_entries
                log.info("Process {} with class {} of fold {}:".format(
                    process, mapped_class, fold
                ))
                log.info("Contains {} events.".format(N_entries))
                for val_wei in uproot.iterate(
                    file_path,
                    expressions=variables + [weight_var],
                    library="np",
                    step_size=args.num_events,
                ):
                    # Get weights
                    input_weights = val_wei[weight_var]
                    log.info("Read chunk with {} events.".format(len(input_weights)))
                    # Apply preprocessing to input data
                    input_data = scaler.transform(
                        np.transpose(
                            [val_wei[var] for var in variables]
                        )
                    )
                    # Add one-hot-encoding for the training identifiers if there is more than one
                    # (All 1 if only one identifier is used)
                    if len(ids) > 1:
                        input_data = np.insert(input_data, len(ids) * [len(variables)], 0, axis=1)
                        input_data[:, len(variables) + i_id] = 1
                    # Create one-hot-encoded labels for the training classes
                    input_labels = np.array(len(input_data) * [len(classes) * [0]])
                    input_labels[:, i_class] = 1
                    # Transform numpy array with samples to tensorflow tensor
                    sample_tensor = tf.convert_to_tensor(input_data)
                    # Get array of gradients of model wrt. samples
                    gradients = tf.squeeze(
                        get_gradients(model, sample_tensor, i_class), axis=1
                    )
                    # Fix dimensions if only one sample remains
                    if len(val_wei) == 1:
                        input_weights = np.array(input_weights)
                    # Concatenate new gradients/ abs of gradients to previous results
                    if args.no_abs:
                        gradients = np.concatenate(
                            ([gradients_intermediate[mapped_class]], gradients), axis=0)
                    else:
                        gradients = np.concatenate(
                            ([gradients_intermediate[mapped_class]], np.abs(gradients)), axis=0)
                    # Concatenate new weights to previous weights
                    gradients_weights = np.concatenate(
                        ([gradient_weights_intermediate[mapped_class]], input_weights), axis=0)
                    # Get new itermediate averages and weights
                    gradients_intermediate[mapped_class] = np.average(gradients,
                                                        weights=gradients_weights,
                                                        axis=0)
                    gradient_weights_intermediate[mapped_class] = np.sum(gradients_weights)
            all_gradient_weights[id_] = gradient_weights_intermediate
            all_gradients[id_] = gradients_intermediate
            matrix = np.vstack([gradients_intermediate[class_] for class_ in classes])
            # Normalize rows
            if not args.no_normalize:
                for i_class, class_ in enumerate(classes):
                    matrix[i_class, :] = matrix[i_class, :] / np.sum(
                        matrix[i_class, :])
            # Plot results
            if num_id_inputs:
                vars_ = variables + ids
            else:
                vars_ = variables
            create_plots(vars_, classes, fold, matrix, id_)
        if num_id_inputs:
            # 1D tylor across the different identifiers
            # Weighted average over all identifiers
            gradients = {}
            for class_ in classes:
                grad = [all_gradients[id_][class_] for id_ in ids]
                wght = [all_gradient_weights[id_][class_] for id_ in ids]
                gradients[class_] = np.average(
                    grad,
                    weights=wght,
                    axis=0
                )
            matrix = np.vstack([gradients[class_] for class_ in classes])
            # Normalize rows
            if not args.no_normalize:
                for i_class, class_ in enumerate(classes):
                    matrix[i_class, :] = matrix[i_class, :] / np.sum(
                        matrix[i_class, :])
            # Plot results
            create_plots(vars_, classes, fold, matrix, args.training_name)


if __name__ == "__main__":
    args = parse_arguments()
    training_config = parse_config(args.config_file, args.training_name)
    main(args, training_config)
