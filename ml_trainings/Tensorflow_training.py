import argparse
import logging as log
log.basicConfig(
    format="Tensorflow_training - %(levelname)s - %(message)s", level=log.INFO
)

import os
import time
import yaml
import pickle

import uproot
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing, model_selection

import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams["font.size"] = 16
import matplotlib.pyplot as plt

from ml_trainings.Config_merger import get_merged_config
import Tensorflow_models

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train ML network")
    parser.add_argument("--config-file", help="Path to training config file")
    parser.add_argument("--training-name", help="Name of training")
    parser.add_argument("--data-dir", help="Dir of process datasets")
    parser.add_argument("--fold", help="Fold of training (0 or 1)")
    parser.add_argument("--output-dir", help="Output directory of training")
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="use of mixed precision when training on GPU",
    )
    return parser.parse_args()

def parse_config(file, name):
    with open(file, "r") as stream:
        config = yaml.safe_load(stream)
    training_config = get_merged_config(config, name)
    return training_config

def main(args, training_config):
    # Start setup timer
    time_setup_start = time.time()
    log.info("Python inputs: {}".format(args))

    # Get config of training
    model_config = training_config["model"]
    log.debug("Using the configs: {}".format(model_config))

    ## Set up of TensorFlow and Numpy
    np.random.seed(int(model_config["seed"]))
    tf.random.set_seed(int(model_config["seed"]))
    log.debug("Using Tensorflow {} from {}".format(tf.__version__, tf.__file__))
    # Set up CPU resources
    if os.getenv("OMP_NUM_THREADS"):
        n_CPU_cores = int(os.environ["OMP_NUM_THREADS"])
    else:
        log.info("'OMP_NUM_THREADS' is not set. Defaulting to 12.")
        n_CPU_cores = 12
    physical_CPU_devices = tf.config.list_physical_devices("CPU")
    log.info("Default CPU Devices: {}".format(physical_CPU_devices))
    log.info("Using {} CPU cores.".format(n_CPU_cores))
    # Set up GPU resources
    physical_GPU_devices = tf.config.list_physical_devices("GPU")
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
        if len(physical_GPU_devices) <= 1:
            distribution_strategy = tf.distribute.get_strategy()
        else:
            # Strategy to use for multi-gpu training
            distribution_strategy = tf.distribute.MultiWorkerMirroredStrategy()
        if args.mixed_precision:
            # Use of mixed precision with GPU
            from tensorflow.keras import mixed_precision

            policy = mixed_precision.Policy("mixed_float16")
            mixed_precision.set_global_policy(policy)
            log.info("Using mixed precision:")
            log.info("Compute dtype: {}".format(policy.compute_dtype))
            log.info("Variable dtype: {}".format(policy.variable_dtype))

    else:
        log.info("No GPU found. Using only CPU.")
        distribution_strategy = tf.distribute.get_strategy()

    ids = list(training_config["parts"].keys())
    num_id_inputs = len(ids) if len(ids) > 1 else 0
    processes = training_config["processes"]
    classes = training_config["classes"]
    variables = training_config["variables"]
    weight_var = "weight"
    log.debug("Used identifiers: {}".format(ids))
    log.debug("Used processes: {}".format(processes))
    log.debug("Used classes: {}".format(classes))
    log.debug("Used variables: {}".format(variables))
    allowed_types = ["float", "int32_t", "model"]

    full_data = {}
    for id_ in ids:
        full_data[id_] = {}
        for class_ in classes:
            full_data[id_][class_] = {}
            full_data[id_][class_]["weights"] = []
            full_data[id_][class_]["inputs"] = {}
            for var in variables:
                full_data[id_][class_]["inputs"][var] = []
        for process in processes:
            # Load datashard for this process
            mapped_class = training_config["mapping"][process]
            file_path = "{d_p}/{id}_{pr}_{t_c}_datashard_fold{fold}.root".format(
                d_p=args.data_dir,
                id=id_,
                pr=process,
                t_c=mapped_class,
                fold=args.fold,
            )
            # if not os.path.exists(file_path):
            #     log.error("File {} does not exist.".format(file_path))
            #     raise Exception("File not found.")
            log.debug("Reading {}".format(file_path))
            upfile = uproot.open(file_path)
            if mapped_class != upfile.keys()[0].split(";")[0]:
                log.error(
                    "Class mapped by the config file and present in the "
                    "datashard do not match for {}: {} and {}".format(
                        file_path, mapped_class, upfile.keys()[0].split(";")[0]
                    )
                )
                raise Exception("Consistency error in Tensorflow training.")
            uptree = upfile[mapped_class]
            # Check if types of variables can be used
            typenames = uptree.typenames()
            for variable in variables + [weight_var]:
                if not variable in typenames:
                    log.error(
                        "Training variable {} specified in training config not found in datashard.".format(
                            variable
                        )
                    )
                    raise Exception("Consistency error in Tensorflow training.")
                if typenames[variable] not in allowed_types:
                    log.error(
                        "Training variable {} has type {} which is not allowed.".format(
                            variable, typenames[variable]
                        )
                    )
                    raise Exception("Consistency error in Tensorflow training.")

            # Read input variables from datashards
            file_inputs = uptree.arrays(variables, library="np")
            for var in variables:
                full_data[id_][mapped_class]["inputs"][var].append(file_inputs[var])
            # Read event weights from datashards
            file_weights = uptree[weight_var].array(library="np")
            full_data[id_][mapped_class]["weights"].append(file_weights)
        for class_ in classes:
            # Check if any training class remains empty after loading the data
            if not full_data[id_][class_]["inputs"]:
                log.error("Class {} of id {} has no samples.".format(class_, id_))
                log.error(
                    "The current mappings are: {}".format(training_config["mapping"])
                )
                raise Exception("Invalid process mapping.")
            # Combine data of different processes in the same class
            for var in variables:
                full_data[id_][class_]["inputs"][var] = np.concatenate(
                    full_data[id_][class_]["inputs"][var]
                )
            full_data[id_][class_]["weights"] = np.concatenate(
                full_data[id_][class_]["weights"]
            )

    # Transform input data into sklearn readable format
    input_prescale = np.transpose(
        np.concatenate(
            [
                [full_data[id_][class_]["inputs"][var] for var in variables]
                for class_ in classes
                for id_ in ids
            ],
            axis=1,
        )
    )

    log.info("Use preprocessing method {}.".format(model_config["preprocessing"]))
    if "standard_scaler" in model_config["preprocessing"]:
        scaler = preprocessing.StandardScaler().fit(input_prescale)
        for var, mean, std in zip(variables, scaler.mean_, scaler.scale_):
            log.debug(
                "Preprocessing (variable, mean, std): {}, {}, {}".format(var, mean, std)
            )
    elif "identity" in model_config["preprocessing"]:
        scaler = preprocessing.StandardScaler().fit(input_prescale)
        for i in range(len(scaler.mean_)):
            scaler.mean_[i] = 0.0
            scaler.scale_[i] = 1.0
        for var, mean, std in zip(variables, scaler.mean_, scaler.scale_):
            log.debug(
                "Preprocessing (variable, mean, std): {}, {}, {}".format(var, mean, std)
            )
    elif "robust_scaler" in model_config["preprocessing"]:
        scaler = preprocessing.RobustScaler().fit(input_prescale)
        for var, mean, std in zip(variables, scaler.center_, scaler.scale_):
            log.debug(
                "Preprocessing (variable, mean, std): {}, {}, {}".format(var, mean, std)
            )
    elif "min_max_scaler" in model_config["preprocessing"]:
        scaler = preprocessing.MinMaxScaler(feature_range=(-1.0, 1.0)).fit(
            input_prescale
        )
        for var, min_, max_ in zip(variables, scaler.data_min_, scaler.data_max_):
            log.debug(
                "Preprocessing (variable, min, max): {}, {}, {}".format(var, min_, max_)
            )
    elif "quantile_transformer" in model_config["preprocessing"]:
        scaler = preprocessing.QuantileTransformer(
            output_distribution="normal", random_state=int(model_config["seed"])
        ).fit(input_prescale)
    else:
        log.fatal(
            "Preprocessing {} is not implemented.".format(model_config["preprocessing"])
        )
        raise Exception("Invalid config.")
    path_preprocessing = "{}/fold{}_keras_preprocessing.pickle".format(
        args.output_dir, args.fold
    )
    log.info("Write preprocessing object to {}.".format(path_preprocessing))
    # Save preprocessing object
    with open(path_preprocessing, "wb") as stream:
        pickle.dump(scaler, stream)

    for i_id, id_ in enumerate(ids):
        for i_class, class_ in enumerate(classes):
            # Apply preprocessing to input data
            input_data = scaler.transform(
                np.transpose(
                    [full_data[id_][class_]["inputs"][var] for var in variables]
                )
            )
            # Add one-hot-encoding for the training identifiers if there is more than one
            # (All 1 if only one identifier is used)
            if len(ids) > 1:
                input_data = np.insert(input_data, len(ids) * [len(variables)], 0, axis=1)
                input_data[:, len(variables) + i_id] = 1

            input_weights = full_data[id_][class_]["weights"]

            # Create one-hot-encoded labels for the training classes
            input_labels = np.array(len(input_data) * [len(classes) * [0]])
            input_labels[:, i_class] = 1

            # Split input, weights and labels into training and test datasets
            (
                train_data,
                test_data,
                train_weights,
                test_weights,
                train_labels,
                test_labels,
            ) = model_selection.train_test_split(
                input_data,
                input_weights,
                input_labels,
                test_size=1.0 - model_config["train_test_split"],
                random_state=int(model_config["seed"]),
            )
            full_data[id_][class_]["inputs"] = {
                "train": train_data,
                "test": test_data,
            }
            full_data[id_][class_]["weights"] = {
                "train": train_weights,
                "test": test_weights,
            }
            full_data[id_][class_]["labels"] = {
                "train": train_labels,
                "test": test_labels,
            }

    # Count number of samples for each identifier/class
    tot_train_samples = []
    tot_test_samples = []
    for id_ in ids:
        log.debug("Training samples in {}".format(id_))
        num_train_samples = [
            len(full_data[id_][class_]["inputs"]["train"]) for class_ in classes
        ]
        num_test_samples = [
            len(full_data[id_][class_]["inputs"]["test"]) for class_ in classes
        ]
        sample_string = [
            "{}: {}".format(class_, num_train_samples[i_class])
            for i_class, class_ in enumerate(classes)
        ]
        log.debug("Training samples per class: {}".format(", ".join(sample_string)))
        tot_train_samples.append(num_train_samples)
        tot_test_samples.append(num_test_samples)
    log.info(
        "Total number of events in training data: {}".format(np.sum(tot_train_samples))
    )

    # Add callbacks
    model_path = "{}/fold{}_keras_model.h5".format(args.output_dir, args.fold)
    callbacks = []
    if "early_stopping" in model_config:
        from tensorflow.keras.callbacks import EarlyStopping

        log.info("Stop early after {} tries.".format(model_config["early_stopping"]))
        callbacks.append(EarlyStopping(patience=model_config["early_stopping"]))
    if "tensorboard" in model_config:
        from tensorflow.keras.callbacks import TensorBoard

        tensorboard_dir = "{}/tensorboard_logs_fold{}".format(
            args.output_dir, args.fold
        )
        log.info("Saving tensorboard logs in {}".format(tensorboard_dir))
        callbacks.append(
            TensorBoard(
                log_dir=tensorboard_dir,
                profile_batch="2, {}".format(model_config["steps_per_epoch"]),
            )
        )

    if "save_best_only" in model_config:
        from tensorflow.keras.callbacks import ModelCheckpoint

        log.info("Write best model to {}.".format(model_path))
        callbacks.append(ModelCheckpoint(model_path, save_best_only=True, verbose=1))

    if "reduce_lr_on_plateau" in model_config:
        from tensorflow.keras.callbacks import ReduceLROnPlateau

        log.info(
            "Reduce learning-rate after {} tries.".format(
                model_config["reduce_lr_on_plateau"]
            )
        )
        callbacks.append(
            ReduceLROnPlateau(patience=model_config["reduce_lr_on_plateau"], verbose=1)
        )

    if not hasattr(Tensorflow_models, model_config["name"]):
        log.fatal("Model {} is not implemented.".format(model_config["name"]))
        raise Exception("Invalid config.")

    log.info("Train keras model {}.".format(model_config["name"]))
    with distribution_strategy.scope():
        model_impl = getattr(Tensorflow_models, model_config["name"])
        model = model_impl(len(variables) + num_id_inputs, len(classes))
        # model.summary()
    log.info("Running on balanced batches.")

    # Generator python for balanced batches
    from tensorflow.keras.utils import Sequence

    class balancedBatchGenerator(Sequence):
        def __init__(self):
            self.nperClassId = int(model_config["eventsPerClassAndBatch"])
            self.classes = classes
            self.ids = ids
            # Number of samples in the classes for each identifier
            self.class_lenghts = {
                id_: {
                    class_: len(full_data[id_][class_]["inputs"]["train"])
                    for class_ in self.classes
                }
                for id_ in self.ids
            }

        def __len__(self):
            # Max num of batches needed during training
            return model_config["steps_per_epoch"] * model_config["epochs"]

        # Function to get dict of random indices for each class.
        # The list of each class contains the same number of indices
        #   regardless of how many samples the class contains.
        # Also returns a list of indices to shuffle the generated databatch.
        def get_index_list(self):
            rand_ints = {}
            for id_ in self.ids:
                rand_ints[id_] = {}
                for class_ in self.classes:
                    rand_ints[id_][class_] = np.random.randint(
                        self.class_lenghts[id_][class_],
                        size=self.nperClassId,
                    )
            shuffle_list = np.arange(
                len(self.ids) * len(self.classes) * self.nperClassId
            )
            np.random.shuffle(shuffle_list)
            return rand_ints, shuffle_list

        def __getitem__(self, idx):
            # Collect training data, weights and labels based on balanced batch approach
            index_list, shuffle_list = self.get_index_list()
            input_collect = np.concatenate(
                [
                    full_data[id_][class_]["inputs"]["train"][index_list[id_][class_]]
                    for class_ in self.classes
                    for id_ in self.ids
                ]
            )[shuffle_list]
            label_collect = np.concatenate(
                [
                    full_data[id_][class_]["labels"]["train"][index_list[id_][class_]]
                    for class_ in self.classes
                    for id_ in self.ids
                ]
            )[shuffle_list]
            # Weights are normed to the number of samples from each class
            weight_collect = np.concatenate(
                [
                    full_data[id_][class_]["weights"]["train"][index_list[id_][class_]]
                    * (
                        self.nperClassId
                        / np.sum(
                            full_data[id_][class_]["weights"]["train"][
                                index_list[id_][class_]
                            ]
                        )
                    )
                    for class_ in self.classes
                    for id_ in self.ids
                ]
            )[shuffle_list]
            # return list of choosen values
            return input_collect, label_collect, weight_collect

    # Signature consists of (input data, labels, weights)
    gen_output_signature = (
        tf.TensorSpec(
            shape=(None, len(variables) + num_id_inputs),
            dtype=tf.float64,
        ),
        tf.TensorSpec(
            shape=(None, len(classes)),
            dtype=tf.float64,
        ),
        tf.TensorSpec(
            shape=None,
            dtype=tf.float64,
        ),
    )

    # Wrap training data generator as tf.Dataset
    traindata = tf.data.Dataset.from_generator(
        balancedBatchGenerator, output_signature=gen_output_signature
    )

    # Set up validation data as a single batch
    input_test = np.concatenate(
        [full_data[id_][class_]["inputs"]["test"] for class_ in classes for id_ in ids]
    )
    label_test = np.concatenate(
        [full_data[id_][class_]["labels"]["test"] for class_ in classes for id_ in ids]
    )
    num_train_samples = np.sum(tot_test_samples)
    weight_test = np.concatenate(
        [
            full_data[id_][class_]["weights"]["test"]
            * (num_train_samples / np.sum(full_data[id_][class_]["weights"]["test"]))
            for class_ in classes
            for id_ in ids
        ]
    )
    # Wrap validation data as tf.Dataset
    validata = tf.data.Dataset.from_tensor_slices(
        (input_test, label_test, weight_test)
    ).batch(len(input_test))

    # Return set-up time
    time_setup_end = time.time()
    log.info("Elapsed setup time: {}".format(time_setup_end - time_setup_start))

    # Run ML training with timer
    time_training_start = time.time()
    ML_fit_parameters = {
        "x": traindata,
        "steps_per_epoch": model_config["steps_per_epoch"],
        "epochs": model_config["epochs"],
        "callbacks": callbacks,
        "validation_data": validata,
        "verbose": 2,
    }
    history = model.fit(**ML_fit_parameters)
    time_training_end = time.time()
    log.info(
        "Elapsed training time: {}".format(time_training_end - time_training_start)
    )

    # Plot the course of the losses during the training
    path_plot = "{}/fold{}_loss".format(args.output_dir, args.fold)
    epochs = range(1, len(history.history["loss"]) + 1)
    axes = plt.gca()
    add_vertical_space = 1
    axes.set_ylim([0, history.history["val_loss"][0] + add_vertical_space])
    plt.plot(epochs, history.history["loss"], lw=3, label="Training loss")
    plt.plot(epochs, history.history["val_loss"], lw=3, label="Validation loss")
    plt.xlabel("Epoch"), plt.ylabel("Loss")
    plt.legend()
    plt.savefig(path_plot + ".png", bbox_inches="tight")
    plt.savefig(path_plot + ".pdf", bbox_inches="tight")

    # Save the trained model
    if not "save_best_only" in model_config:
        logger.info("Write model to {}.".format(model_path))
        model.save(model_path)


if __name__ == "__main__":
    args = parse_arguments()
    training_config = parse_config(args.config_file, args.training_name)
    main(args, training_config)
