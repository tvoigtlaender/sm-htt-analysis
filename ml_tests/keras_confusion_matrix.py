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
from ml_util.config_merger import get_merged_config
import matplotlib as mpl

mpl.use("Agg")
mpl.rcParams["font.size"] = 16
import matplotlib.pyplot as plt


def parse_arguments():
    parser = argparse.ArgumentParser(description="Produce confusion matrice")
    parser.add_argument("--config-file", help="Path to training config file")
    parser.add_argument("--training-name", help="Name of training")
    parser.add_argument("--model-dir", help="Dir of trained ML model")
    parser.add_argument("--data-dir", help="Dir of process datasets")
    parser.add_argument("--output-dir", help="Output directory of training")
    parser.add_argument(
        "--num-events",
        help="Number of events in one chunk",
        default="100 MB",
    )
    return parser.parse_args()


def parse_config(file, name):
    with open(file, "r") as stream:
        config = yaml.safe_load(stream)
    training_config = get_merged_config(config, name)
    return training_config


def get_efficiency_representations(m):
    ma = np.zeros(m.shape)
    mb = np.zeros(m.shape)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ma[i, j] = m[i, j] / m[i, i]
            mb[i, j] = m[i, j] / np.sum(m[i, :])
    return ma, mb


def get_purity_representations(m):
    ma = np.zeros(m.shape)
    mb = np.zeros(m.shape)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ma[i, j] = m[i, j] / m[j, j]
            mb[i, j] = m[i, j] / np.sum(m[:, j])
    return ma, mb


def plot_confusion(confusion, classes, filename, label, markup="{:.2f}"):
    log.debug("Write plot to %s.", filename)
    plt.figure(figsize=(2.5 * confusion.shape[0], 2.0 * confusion.shape[1]))
    axis = plt.gca()
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            axis.text(
                i + 0.5,
                j + 0.5,
                markup.format(confusion[i, -1 - j]),
                ha="center",
                va="center",
            )
    q = plt.pcolormesh(np.transpose(confusion)[::-1], cmap="Wistia")
    cbar = plt.colorbar(q)
    cbar.set_label(label, rotation=270, labelpad=50)
    plt.xticks(np.array(range(len(classes))) + 0.5, classes, rotation="vertical")
    plt.yticks(
        np.array(range(len(classes))) + 0.5, classes[::-1], rotation="horizontal"
    )
    plt.xlim(0, len(classes))
    plt.ylim(0, len(classes))
    plt.ylabel("Predicted class")
    plt.xlabel("True class")
    plt.savefig(filename + ".png", bbox_inches="tight")
    plt.savefig(filename + ".pdf", bbox_inches="tight")
    plt.close()

    d = {}
    for i1, c1 in enumerate(classes):
        d[c1] = {}
        for i2, c2 in enumerate(classes):
            d[c1][c2] = float(confusion[i1, i2])
    f = open(filename + ".yaml", "w")
    yaml.dump(d, f)


def print_matrix(p, title):
    stdout.write(title + "\n")
    for i in range(p.shape[0]):
        stdout.write("    ")
        for j in range(p.shape[1]):
            stdout.write("{:.4f} & ".format(p[i, j]))
        stdout.write("\b\b\\\\\n")


def create_plots(classes, conf1, conf2, fold, name):
    conf1 = np.array(conf1)
    conf2 = np.array(conf2)
    # Plot confusion matrix
    log.info("Write confusion matrices for {} fold {}.".format(name, fold))
    path_template = os.path.join(args.output_dir, "fold{}_keras_confusion_{}_{}")

    plot_confusion(
        conf1, classes, path_template.format(fold, name, "standard"), "Arbitrary unit"
    )
    plot_confusion(
        conf2,
        classes,
        path_template.format(fold, name, "standard_cw"),
        "Arbitrary unit",
    )

    confusion_eff1, confusion_eff2 = get_efficiency_representations(conf1)
    confusion_eff3, confusion_eff4 = get_efficiency_representations(conf2)
    plot_confusion(
        confusion_eff1,
        classes,
        path_template.format(fold, name, "efficiency1"),
        "Efficiency",
    )
    plot_confusion(
        confusion_eff2,
        classes,
        path_template.format(fold, name, "efficiency2"),
        "Efficiency",
    )
    plot_confusion(
        confusion_eff3,
        classes,
        path_template.format(fold, name, "efficiency_cw1"),
        "Efficiency",
    )
    plot_confusion(
        confusion_eff4,
        classes,
        path_template.format(fold, name, "efficiency_cw2"),
        "Efficiency",
    )

    confusion_pur1, confusion_pur2 = get_purity_representations(conf1)
    confusion_pur3, confusion_pur4 = get_purity_representations(conf2)
    plot_confusion(
        confusion_pur1, classes, path_template.format(fold, name, "purity1"), "Purity"
    )
    plot_confusion(
        confusion_pur2, classes, path_template.format(fold, name, "purity2"), "Purity"
    )
    plot_confusion(
        confusion_pur3,
        classes,
        path_template.format(fold, name, "purity_cw1"),
        "Purity",
    )
    plot_confusion(
        confusion_pur4,
        classes,
        path_template.format(fold, name, "purity_cw2"),
        "Purity",
    )


# Function to compute model answers in optimized graph mode
@tf.function(experimental_relax_shapes=True)
def get_values(model, samples):
    responses = model(samples, training=False)
    return responses


def main(args, training_config):
    # log.info(args)
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
    all_sum_weights = {}
    for id_ in ids:
        all_sum_weights[id_] = 0
    all_confusion = {}
    all_confusion2 = {}
    for fold in folds:
        # Load scaler
        preprocessing_path = os.path.join(
            args.model_dir, "fold{fold}_keras_preprocessing.pickle".format(fold=fold)
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

        all_sum_weights[fold] = {}
        all_confusion[fold] = {}
        all_confusion2[fold] = {}
        for i_id, id_ in enumerate(ids):
            confusion = np.zeros((len(classes), len(classes)), dtype=float)
            sum_weights = np.zeros(len(classes), dtype=float)
            for process in processes:
                # Load datashard for this process
                mapped_class = training_config["mapping"][process]
                # Get index of mapped training class
                i_class = classes.index(mapped_class)
                file_path = "{d_p}/{id}_{pr}_{t_c}_datashard_fold{fold}.root".format(
                    d_p=args.data_dir,
                    id=id_,
                    pr=process,
                    t_c=mapped_class,
                    fold=inverse_folds[fold],
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
                        log.info("Read chunk with {} events.".format(len(input_weights)))
                        # Add sum of weights to the mapped class
                        sum_weights[i_class] += np.sum(input_weights)
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
                        # Get interference from trained model
                        event_responses = get_values(model, sample_tensor)
                        # Determine class for each sample
                        max_indexes = np.argmax(event_responses, axis=1)
                        # Sum over weights of samples for each response
                        for i, indexes in enumerate(classes):
                            mask = max_indexes == i
                            confusion[i_class, i] += np.sum(input_weights[mask])
                # Collect weight sums and confusion matrices
                all_sum_weights[id_] += sum_weights
                all_confusion[fold][id_] = confusion
    all_class_weights = {}
    for id_ in ids:
        # Get weighted confusion matrices for each part of the dataset separately
        class_weights = [
            np.sum(all_sum_weights[id_]) / weight for weight in all_sum_weights[id_]
        ]
        all_class_weights[id_] = class_weights
        for fold in folds:
            confusion = all_confusion[fold][id_]
            # Get class confusion matrice weighted by the class weights
            confusion2 = []
            for i in range(len(classes)):
                confusion2.append([val_ * class_weights[i] for val_ in confusion[i]])
            # confusion2 = np.array(confusion2)
            all_confusion2[fold][id_] = confusion2
            # Debug output to ensure that plotting is correct
            for i_class, class_ in enumerate(classes):
                log.debug("True class: {}".format(class_))
                for j_class, class2 in enumerate(classes):
                    log.debug(
                        "Predicted {}: {}".format(class2, confusion[i_class, j_class])
                    )
            create_plots(classes, confusion, confusion2, fold, id_)
    if num_id_inputs:
        # Sum of weights across the different identifiers
        sum_weights = sum([all_sum_weights[id_] for id_ in ids])
        class_weights = [np.sum(sum_weights) / weight for weight in sum_weights]
        # Confusion matrices across the different identifiers
        for fold in folds:
            confusion = sum([all_confusion[fold][id_] for id_ in ids])
            confusion2 = []
            for i in range(len(classes)):
                confusion2.append([val_ * class_weights[i] for val_ in confusion[i]])
            # Debug output to ensure that plotting is correct
            for i_class, class_ in enumerate(classes):
                log.debug("True class: {}".format(class_))
                for j_class, class2 in enumerate(classes):
                    log.debug(
                        "Predicted {}: {}".format(class2, confusion[i_class, j_class])
                    )
            # Create combined plots
            create_plots(classes, confusion, confusion2, fold, args.training_name)


if __name__ == "__main__":
    args = parse_arguments()
    training_config = parse_config(args.config_file, args.training_name)
    main(args, training_config)
