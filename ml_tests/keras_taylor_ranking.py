#!/usr/bin/env python

import logging as log

log.basicConfig(
    format="Tensorflow_training - %(levelname)s - %(message)s", level=log.INFO
)
import argparse
import os
import yaml
import pickle
import time
import uproot
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from ml_util.config_merger import get_merged_config
import matplotlib as mpl

mpl.use("Agg")
mpl.rcParams["font.size"] = 20
import matplotlib.pyplot as plt


def parse_arguments():
    parser = argparse.ArgumentParser(description="Calculate taylor ranking.")
    parser.add_argument("--config-file", help="Path to training config file")
    parser.add_argument("--training-name", help="Name of training")
    parser.add_argument("--model-dir", help="Dir of trained ML model")
    parser.add_argument("--data-dir", help="Dir of process datasets")
    parser.add_argument("--output-dir", help="Output directory of training")
    parser.add_argument(
        "--num-events",
        help="Number of events in one chunk",
        default="1 MB",
    )
    parser.add_argument(
        "--no-abs",
        action="store_true",
        default=False,
        help="Do not use abs for metric.",
    )
    return parser.parse_args()


def parse_config(file, name):
    with open(file, "r") as stream:
        config = yaml.safe_load(stream)
    training_config = get_merged_config(config, name)
    return training_config


def create_plots(mean_deriv, class_deriv_weights, classes, deriv_ops_names, fold, name):
    log.info("Write 1D taylor plot for {} fold {}.".format(name, fold))
    deriv_all = np.vstack([mean_deriv[class_] for class_ in classes])
    weights_all = np.hstack([[class_deriv_weights[class_]] for class_ in classes])
    if args.no_abs:
        mean_deriv_all = np.average((deriv_all), weights=weights_all, axis=0)
    else:
        mean_deriv_all = np.average(np.abs(deriv_all), weights=weights_all, axis=0)
    mean_deriv["all"] = mean_deriv_all
    # Get ranking
    ranking = {}
    labels = {}
    for class_ in classes + ["all"]:
        labels_tmp = []
        ranking_tmp = []
        for names, value in zip(deriv_ops_names, mean_deriv[class_]):
            labels_tmp.append(", ".join(names))
            if len(names) == 2:
                if names[0] == names[1]:
                    ranking_tmp.append(0.5 * value)
                else:
                    ranking_tmp.append(value)
            else:
                ranking_tmp.append(value)

        yx = list(zip(ranking_tmp, labels_tmp))
        yx = sorted(yx, reverse=True)
        labels_tmp = [x for y, x in yx]
        ranking_tmp = [y for y, x in yx]

        ranking[class_] = ranking_tmp
        labels[class_] = labels_tmp

    ranking_singles = {}
    labels_singles = {}
    for class_ in classes + ["all"]:
        labels_tmp = []
        ranking_tmp = []
        for names, value in zip(deriv_ops_names, mean_deriv[class_]):
            if len(names) > 1:
                continue
            labels_tmp.append(", ".join(names))
            ranking_tmp.append(value)

        yx = list(zip(ranking_tmp, labels_tmp))
        yx = sorted(yx, reverse=True)
        labels_tmp = [x for y, x in yx]
        ranking_tmp = [y for y, x in yx]

        ranking_singles[class_] = ranking_tmp
        labels_singles[class_] = labels_tmp

    # Write table
    for class_ in classes + ["all"]:
        output_path = os.path.join(
            args.output_dir,
            "fold{}_keras_taylor_ranking_{}_{}.txt".format(fold, name, class_),
        )
        log.info("Save table to {}.".format(output_path))
        f = open(output_path, "w")
        for rank, (label, score) in enumerate(zip(labels[class_], ranking[class_])):
            f.write("{0:<4} : {1:<60} : {2:g}\n".format(rank, label, score))

    # # Write table
    # for class_ in classes + ["all"]:
    #     plot_name_era = "_{}".format(args.era) if args.era else ""
    #     output_path = os.path.join(
    #         config_train["output_path"],
    #         "fold{}_keras_taylor_1D_{}{}.txt".format(args.fold, class_,
    #                                                  plot_name_era))
    #     log.info("Save table to {}.".format(output_path))
    #     f = open(output_path, "w")
    #     for rank, (label, score) in enumerate(
    #             zip(labels_singles[class_], ranking_singles[class_])):
    #         f.write("{0:<4} : {1:<60} : {2:g}\n".format(rank, label, score))

    # Store results for combined metric in file
    output_yaml = []
    for names, score in zip(labels["all"], ranking["all"]):
        output_yaml.append({"variables": names.split(", "), "score": float(score)})
    output_path = os.path.join(
        args.output_dir, "fold{}_keras_taylor_ranking_{}.yaml".format(fold, name)
    )
    yaml.dump(output_yaml, open(output_path, "w"), default_flow_style=False)
    log.info("Save results to {}.".format(output_path))

    # Plotting
    for class_ in classes + ["all"]:
        plt.figure(figsize=(7, 4))
        ranks_1d = []
        ranks_2d = []
        scores_1d = []
        scores_2d = []
        for i, (label, score) in enumerate(zip(labels[class_], ranking[class_])):
            if ", " in label:
                scores_2d.append(score)
                ranks_2d.append(i)
            else:
                scores_1d.append(score)
                ranks_1d.append(i)
        plt.clf()

        plt.plot(
            ranks_2d,
            scores_2d,
            "+",
            mew=10,
            ms=3,
            label="Second-order features",
            alpha=1.0,
        )
        plt.plot(
            ranks_1d,
            scores_1d,
            "+",
            mew=10,
            ms=3,
            label="First-order features",
            alpha=1.0,
        )
        plt.xlabel("Rank")
        plt.ylabel("$\\langle t_{i} \\rangle$")
        plt.legend()
        output_path = os.path.join(
            args.output_dir,
            "fold{}_keras_taylor_ranking_{}_{}.png".format(fold, name, class_),
        )
        log.info("Save plot to {}.".format(output_path))
        plt.savefig(output_path, bbox_inches="tight")
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


# Function to compute hessian of model answers in optimized graph mode
@tf.function(experimental_relax_shapes=True)
def get_hessians(model, samples, output_ind):
    # Define function to get single hessian
    def get_single_hessian(single_sample):
        single_sample = tf.expand_dims(single_sample, axis=0)

        # Function to compute gradient of a vector in regard to inputs (on outer tape)
        def tot_gradient(vector):
            return tape.gradient(vector, single_sample)

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(single_sample)
            with tf.GradientTape(persistent=True) as tape_of_tape:
                tape_of_tape.watch(single_sample)
                # Get response from model with (only choosen output class)
                response = model(single_sample, training=False)[:, output_ind]
            # Get gradient of choosen output class wrt. input sample
            grads = tape_of_tape.gradient(response, single_sample)
            # Compute hessian of model from gradients of model wrt. input sample
            hessian = tf.map_fn(tot_gradient, grads[0])
        return hessian

    # Apply function to every sample to get all hessians
    hessians = tf.vectorized_map(get_single_hessian, samples)
    return hessians


# Function to get upper triangle of matrix for every matrix in array
def triu_map(matrix_array, size):
    # get upper triangle indices for given matrix size
    triu = np.triu_indices(size)

    # Function to apply indices to single element of matrix array and get single output
    def single_triu(single_mat):
        return single_mat[triu]

    # Apply function to every element
    return np.array(list(map(single_triu, matrix_array)))


def main(args, training_config):
    # Set up CPU resources
    if os.getenv("OMP_NUM_THREADS"):
        n_CPU_cores = int(os.environ["OMP_NUM_THREADS"])
    else:
        log.info("'OMP_NUM_THREADS' is not set. Defaulting to 12.")
        n_CPU_cores = 12
    # Set up GPUs if available
    physical_GPU_devices = tf.config.list_physical_devices("GPU")
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    if physical_GPU_devices:
        log.info("Default GPU Devices: {}".format(physical_GPU_devices))
        log.info("Using {} GPUs.".format(len(physical_GPU_devices)))
        for device in physical_GPU_devices:
            try:
                tf.config.experimental.set_memory_growth(device, True)
            except:
                # Invalid device or cannot modify virtual devices once initialized.
                log.error(
                    "Device memory growth of {} could not be changed.".format(device)
                )
        tf.config.threading.set_intra_op_parallelism_threads(max(n_CPU_cores - 1, 1))
        tf.config.threading.set_inter_op_parallelism_threads(1)
    else:
        log.info("No GPU found. Using only CPU.")
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(max(n_CPU_cores - 1, 1))
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
    length_variables = len(variables) + num_id_inputs
    length_deriv_class = (
        length_variables**2 + length_variables
    ) / 2 + length_variables
    log.debug("Set up derivative names.")
    deriv_ops_names = []
    if num_id_inputs:
        vars_ = variables + ids
    else:
        vars_ = variables
    for var_ in vars_:
        deriv_ops_names.append([var_])
    for i, i_var in enumerate(vars_):
        for j, j_var in enumerate(vars_):
            if j < i:
                continue
            deriv_ops_names.append([i_var, j_var])
    for fold in folds:
        # Load scaler
        preprocessing_path = os.path.join(
            args.model_dir,
            "fold{fold}_keras_preprocessing.pickle".format(fold=fold),
        )
        log.info("Load preprocessing {}.".format(preprocessing_path))
        with open(preprocessing_path, "rb") as stream:
            scaler = pickle.load(stream, encoding="bytes")
        # Load trained model
        model_path = os.path.join(
            args.model_dir,
            "fold{fold}_keras_model.h5".format(fold=fold),
        )
        log.info("Load keras model {}.".format(model_path))
        model = load_model(model_path)
        all_deriv_values = {}
        all_deriv_weights = {}
        for i_id, id_ in enumerate(ids):

            # Get names for first-order and second-order derivatives
            deriv_values_intermediate = {}
            deriv_weights_intermediate = {}
            for class_ in classes:
                len_inputs = len(variables) + num_id_inputs
                deriv_values_intermediate[class_] = np.zeros(
                    int((len_inputs * (len_inputs + 3)) / 2.0)
                )
                deriv_weights_intermediate[class_] = 0
            for process in processes:
                # Load datashard for this process
                mapped_class = training_config["mapping"][process]
                log.info(process)
                log.info(mapped_class)
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
                with uproot.open(file_path) as upfile:
                    file_class = upfile.keys()[0].split(";")[0]
                    if mapped_class != file_class:
                        log.error(
                            "Class mapped by the config file and present in the "
                            "datashard do not match for {}: {} and {}".format(
                                file_path, mapped_class, file_class
                            )
                        )
                        raise Exception("Consistency error in Tensorflow training.")
                    uptree = upfile[mapped_class]
                    N_entries = uptree.num_entries
                    log.info(
                        "Process {} with class {} of fold {}:".format(
                            process, mapped_class, fold
                        )
                    )
                    log.info("Contains {} events.".format(N_entries))
                    for val_wei in uptree.iterate(
                        expressions=variables + [weight_var],
                        library="np",
                        step_size=args.num_events,
                    ):
                        # Get weights
                        input_weights = val_wei[weight_var]
                        log.info(
                            "Read chunk with {} events.".format(len(input_weights))
                        )
                        # Apply preprocessing to input data
                        input_data = scaler.transform(
                            np.transpose([val_wei[var] for var in variables])
                        )
                        # Add one-hot-encoding for the training identifiers if there is more than one
                        # (All 1 if only one identifier is used)
                        if len(ids) > 1:
                            input_data = np.insert(
                                input_data, len(ids) * [len(variables)], 0, axis=1
                            )
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
                        # Get array of hessians of model wrt. samples
                        hessians = tf.squeeze(
                            get_hessians(model, sample_tensor, i_class), axis=2
                        )
                        # Fix dimensions if only one sample remains
                        if len(val_wei) == 1:
                            input_weights = np.array(input_weights)
                        # Get array of upper triangles of hessians of model wrt. samples
                        upper_hessian_half = triu_map(
                            hessians.numpy(), length_variables
                        )
                        # Append gradient values to hessian values
                        deriv_values = np.concatenate(
                            (gradients, upper_hessian_half), axis=1
                        )
                        ## Calculate taylor coefficients ##
                        # Add coefficients / abs of coefficients to previous results
                        if args.no_abs:
                            deriv_values = np.concatenate(
                                (
                                    [deriv_values_intermediate[mapped_class]],
                                    deriv_values,
                                ),
                                axis=0,
                            )
                        else:
                            deriv_values = np.concatenate(
                                (
                                    [deriv_values_intermediate[mapped_class]],
                                    np.abs(deriv_values),
                                ),
                                axis=0,
                            )
                        # Add weights coefficients to previous weights
                        deriv_weights = np.concatenate(
                            ([deriv_weights_intermediate[mapped_class]], input_weights),
                            axis=0,
                        )
                        # Calculate intermeiate results for coefficients and weights
                        deriv_values_intermediate[mapped_class] = np.average(
                            deriv_values, weights=deriv_weights, axis=0
                        )
                        deriv_weights_intermediate[mapped_class] = np.sum(deriv_weights)
            # Collect derivatives and weights
            all_deriv_values[id_] = deriv_values_intermediate
            all_deriv_weights[id_] = deriv_weights_intermediate
            # Create plots for each identifier
            create_plots(
                deriv_values_intermediate,
                deriv_weights_intermediate,
                classes,
                deriv_ops_names,
                fold,
                id_,
            )
        if num_id_inputs:
            # Tylor ranking across the different identifiers
            # Weighted average over all identifiers
            gradients = {}
            weights = {}
            for class_ in classes:
                dval = [all_deriv_values[id_][class_] for id_ in ids]
                dwght = [all_deriv_weights[id_][class_] for id_ in ids]
                gradients[class_] = np.average(dval, weights=dwght, axis=0)
                weights[class_] = np.sum(dwght)
            # Create combined plots
            create_plots(
                gradients, weights, classes, deriv_ops_names, fold, args.training_name
            )


if __name__ == "__main__":
    args = parse_arguments()
    training_config = parse_config(args.config_file, args.training_name)
    runtime_start = time.time()
    main(args, training_config)
    runtime_end = time.time()
    log.info("Elapsed runtime: {}".format(runtime_end - runtime_start))
