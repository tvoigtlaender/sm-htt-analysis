#!/usr/bin/env python

import logging

logger = logging.getLogger("keras_training")

import argparse
import yaml
import os
import pickle
import random
import sys
import uproot
import time
from datetime import datetime
import matplotlib as mpl

mpl.use('Agg')
mpl.rcParams['font.size'] = 16
import matplotlib.pyplot as plt


def parse_arguments():
    logger.debug("Parse arguments.")
    parser = argparse.ArgumentParser(
        description="Train machine Keras models for Htt analyses")
    parser.add_argument("--data-dir", help="Dir of training config file and datashards")
    parser.add_argument("--config-dir", help="Dir of training config file and datashards")
    parser.add_argument("--fold", type=int, help="Select the fold to be trained")
    parser.add_argument("--era-training", help="Era of training (2016,2017,2018 or all_eras)")
    parser.add_argument("--channel-training", help="Era of training (2016,2017,2018 or all_eras)")
    parser.add_argument(
        "--randomization",
        required=False,
        type=bool,
        default=False,
        help=
        "Randomize signal classes for conditional training in case one era has insufficient signal data."
    )
    parser.add_argument(
        "--sample_frac",
        required=False,
        type=float,
        default=-1,
        help=
        "Remove samples from available data, leaving sample_frac of data."
    )
    parser.add_argument(
        "--addn",
        required=False,
        type=float,
        default=0,
        help=
        "Percentage of samples that are duplicated as additional samples with negative weight (sample -> 2positive + 1negative)."
    )
    parser.add_argument(
        "--balance-batches",
        required=False,
        type=bool,
        default=False,
        help=
        "Use a equal amount of events of each class in a batch and normalize those by dividing each individual event weigth by the sum of event weight of the respective class in that batch "
    )
    parser.add_argument(
        "--max-threads",
        default=12,
        help="Maximum number of threads used for training"
    )
    return parser.parse_args()


def parse_config(filename):
    logger.debug("Parse config.")
    return yaml.load(open(filename, "r"), Loader=yaml.SafeLoader)


def setup_logging(level, output_file=None):
    logger.setLevel(level)
    formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if not output_file == None:
        file_handler = logging.FileHandler(output_file, "w")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

def main(args, config):
    logger.info(args)
    if args.era_training == "all_eras":
        conditional = True
    else:
        conditional = False
    output_dir = "/".join([args.config_dir, args.era_training])
    # Set seed and import packages
    # NOTE: This need to be done before any keras module is imported!
    logger.debug("Import packages and set random seed to %s.",
                 int(config["seed"]))
    import numpy as np
    np.random.seed(int(config["seed"]))
    # np.__config__.show()
    
    #Run Tensorflow as deterministic as possible
    import tensorflow as tf
    tf.random.set_seed(int(config["seed"]))

    # check tf version
    logger.info("Using {} from {}".format(tf.__version__, tf.__file__))
    # check CPU availability

    n_CPU = int(args.max_threads)
    n_threads = n_CPU * 1
    m_threads = int(np.ceil(n_threads/10.))
    # check GPU availability
    physical_GPU_devices = tf.config.list_physical_devices('GPU')
    # Flip distribution of CPU cores if a GPU is available
    physical_CPU_devices = tf.config.list_physical_devices('CPU')
    logger.info("Default CPU Devices: {}".format(physical_CPU_devices))
    logger.info("Using {} CPUs.".format(n_CPU))
    if len(physical_GPU_devices)>0:
        m_threads = n_threads - m_threads
    if n_CPU < 2:
        logger.info("Less than two CPU cores available. Thread parallelism not possible.\n Setting intra and inter op parallelism to 1.")
        n_threads = 2
        m_threads = 1
    else:
        logger.info("Using {} threads for intra op parallelism".format(n_threads-m_threads))
        logger.info("Using {} threads for inter op parallelism".format(m_threads))
    tf.config.threading.set_intra_op_parallelism_threads(n_threads-m_threads)
    tf.config.threading.set_inter_op_parallelism_threads(m_threads)
    
    if len(physical_GPU_devices)>0:
        logger.info("Default GPU Devices: {}".format(physical_GPU_devices))
        logger.info("Using {} GPUs.".format(len(physical_GPU_devices)))
        try:
            for device in physical_GPU_devices:
                tf.config.experimental.set_memory_growth(device, True)
        except:
            # Invalid device or cannot modify virtual devices once initialized.
            logger.debug("Dynamic memory growth could not be set for all GPU devices.")
            pass
        # Enable multi GPU training
        # Use default strategy if only one GPU is available
        if len(physical_GPU_devices)<=1:
            distribution_strategy = tf.distribute.get_strategy()
        else:
            distribution_strategy = tf.distribute.MultiWorkerMirroredStrategy()
        
    else:
        logger.info("No GPU found. Using only CPU.")
        distribution_strategy = tf.distribute.get_strategy()
    mixed_precision_var = True
    if (len(physical_GPU_devices)>0 & mixed_precision_var):
        from tensorflow.keras import mixed_precision
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        logger.info('Compute dtype: %s' % policy.compute_dtype)
        logger.info('Variable dtype: %s' % policy.variable_dtype)

    from sklearn import preprocessing, model_selection
    import keras_models
    from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, TensorBoard

    # Extract list of variables
    variables = config["variables"]
    classes = config["classes"]
    processes = config["processes"]
    e_weights = config["event_weights"]
    logger.debug("Use variables:")
    for v in variables:
        logger.debug("%s", v)

    if args.randomization:
        signal_classes = [
            class_ for class_ in classes
            if class_.startswith(('ggh', 'qqh', 'vbftopo'))
        ]
        randomization_era = "2016"
    else:
        signal_classes = []
        randomization_era = None

    # Load training dataset
    if conditional:
        # args.balance_batches = True
        eras = ['2016', '2017', '2018']
        len_eras = len(eras)
    else:
        eras = ['any']
        len_eras = 0
    x = []  # Training input
    y = []  # Target classes
    w = []  # Weights for training
    z = []  # Era information for batching
    for i_era, era in enumerate(eras):
        # Get the datashards of all processes listed in the training config file 
        if conditional:
            filenames = ["{}/{}_{}/{}_datashard_fold{}.root".format(args.data_dir, era, args.channel_training, process, args.fold) for process in processes]
        else:
            filenames = ["{}/{}_{}/{}_datashard_fold{}.root".format(args.data_dir, args.era_training, args.channel_training, process, args.fold) for process in processes]
            # filename = config["datasets"][args.fold]
        print(filenames)
        logger.debug("Load training dataset from {}.".format(filenames))
        upfiles = [uproot.open(filename) for filename in filenames]
        x_era = []
        y_era = []
        w_era = []
        for i_class, class_ in enumerate(classes):
            logger.debug("Process class %s.", class_)
            uptrees = [upfile[class_] for upfile in upfiles if class_ in upfile.keys()[0]]
            if not uptrees:
                logger.fatal("Tree %s not found in files %s.", class_,
                                filenames)
                raise Exception

            # Get inputs for this class from all shards
            class_num_entries = np.sum([uptree.num_entries for uptree in uptrees])
            x_class = np.zeros(
                (class_num_entries, len(variables) + len_eras))
            x_conv_array = [uptree.arrays(expressions=variables, library='np') for uptree in uptrees]
            for i_var, var in enumerate(variables):
                x_conv_var = []
                for x_conv in x_conv_array:
                    x_conv_var.extend(x_conv[var])
                x_class[:, i_var] = x_conv_var
            if np.any(np.isnan(x_class)):
                logger.fatal(
                    "Nan in class {} for era {} in files {} for any of {}".
                    format(class_, era, filenames, variables))
                raise Exception

            # One hot encode eras if conditional. Additionally randomize signal class eras if desired.
            if conditional:
                if (class_ in signal_classes) and args.randomization:
                    logger.debug("Randomizing class {}".format(class_))
                    random_era = np.zeros((class_num_entries, len_eras))
                    for event in random_era:
                        idx = np.random.randint(3, size=1)
                        event[idx] = 1
                    x_class[:, -3:] = random_era
                else:
                    if era == "2016":
                        x_class[:, -3] = np.ones((class_num_entries))
                    elif era == "2017":
                        x_class[:, -2] = np.ones((class_num_entries))
                    elif era == "2018":
                        x_class[:, -1] = np.ones((class_num_entries))
            x_era.append(x_class)

            # Get weights from all shards
            w_class = np.zeros((class_num_entries, 1))
            w_conv = []
            for uptree in uptrees:
                w_conv.extend(uptree.arrays(expressions=[e_weights], library='np')[e_weights])
            if args.balance_batches:
                w_class[:, 0] = w_conv
            else:
                if conditional:
                    w_class[:,
                            0] = w_conv * config[
                                "class_weights_{}".format(era)][class_]
                else:
                    w_class[:, 0] = w_conv * config["class_weights"][class_]
            if np.any(np.isnan(w_class)):
                logger.fatal(
                    "Nan in weight class {} for era {} in files {}.".format(
                        class_, era, filenames))
                raise Exception
            w_era.append(w_class)

            # Get targets for this class
            y_class = np.zeros((class_num_entries, len(classes)))
            y_class[:, i_class] = np.ones((class_num_entries))
            y_era.append(y_class)

        # Stack inputs, targets and weights to a Keras-readable dataset
        x_era = np.vstack(x_era)  # inputs
        y_era = np.vstack(y_era)  # targets
        w_era = np.vstack(w_era)  # weights
        z_era = np.zeros((y_era.shape[0], len(eras)))  # era information
        z_era[:, i_era] = np.ones((y_era.shape[0]))
        x.append(x_era)
        y.append(y_era)
        w.append(w_era)
        z.append(z_era)

    # Stack inputs, targets and weights to a Keras-readable dataset
    x = np.vstack(x)  # inputs
    y = np.vstack(y)  # targets
    w = np.vstack(w)  # weights
    w = np.squeeze(w)  # needed to get weights into keras
    z = np.vstack(z)  # era information

    if args.sample_frac > 0:
        len_samples = len(x)
        len_samples_new = int(len_samples * args.sample_frac)
        logger.info("Removing {} of available data. Now {} of {}.".format(1-args.sample_frac, len_samples_new, len_samples))
        new_samples_index = sorted(np.random.choice(len_samples,len_samples_new,replace=False))
        x = x[new_samples_index]
        y = y[new_samples_index]
        z = z[new_samples_index]
        w = w[new_samples_index]

    # Copy a percentage of each class with original and inverted weight values
    perc_np_test = args.addn
    # perc_np_test = 0.5
    if perc_np_test>0:
        logger.info("Using a portion of {} of duplicated samples.".format(perc_np_test))
        random.seed(config["seed"])
        logger.info("Before all classes: {}, {}, {}, {}".format(len(x), len(y), len(z), len(w)))
        for class_i, _class in enumerate(classes):
            class_mask = (y[:, class_i] == 1)
            x_tmp = x[class_mask]
            y_tmp = y[class_mask]
            z_tmp = z[class_mask]
            w_tmp = w[class_mask]
            length_class = len(x_tmp)
            if int(length_class*perc_np_test)==0:
                pass
            logger.info("There are {} samples in class {}.".format(length_class, _class))
            rand_samples_i = random.sample(range(0, length_class), int(length_class*perc_np_test))
            logger.info("Duplicating {} random samples of class {} (positive and negative weights).".format(len(rand_samples_i), _class))
            x = np.append(x, x_tmp[rand_samples_i], axis=0)
            y = np.append(y, y_tmp[rand_samples_i], axis=0)
            w = np.append(w, w_tmp[rand_samples_i], axis=0)
            z = np.append(z, z_tmp[rand_samples_i], axis=0)
            x = np.append(x, x_tmp[rand_samples_i], axis=0)
            y = np.append(y, y_tmp[rand_samples_i], axis=0)
            w = np.append(w, np.negative(w_tmp[rand_samples_i]), axis=0)
            z = np.append(z, z_tmp[rand_samples_i], axis=0)
            logger.info("After class {}: {}, {}, {}, {}".format(_class, len(x), len(y), len(z), len(w)))
        logger.info("After all classes: {}, {}, {}, {}".format(len(x), len(y), len(z), len(w)))
        x_testing = x
        y_testing = y
        w_testing = w
        logger.info("Sum of weights for classes befor scaling:")
    if config["model"]["cut_fraction"] > 0:
        logger.info("Using limit of {} on generated batches.".format(config["model"]["cut_fraction"]))
    # Perform input variable transformation and pickle scaler object.
    # Only perform transformation on continuous variables
    x_scaler = x[:, :len(variables)]
    logger.info("Use preprocessing method %s.", config["preprocessing"])
    if "standard_scaler" in config["preprocessing"]:
        scaler = preprocessing.StandardScaler().fit(x_scaler)
        for var, mean, std in zip(variables, scaler.mean_, scaler.scale_):
            logger.debug("Preprocessing (variable, mean, std): %s, %s, %s",
                            var, mean, std)
    elif "identity" in config["preprocessing"]:
        scaler = preprocessing.StandardScaler().fit(x_scaler)
        for i in range(len(scaler.mean_)):
            scaler.mean_[i] = 0.0
            scaler.scale_[i] = 1.0
        for var, mean, std in zip(variables, scaler.mean_, scaler.scale_):
            logger.debug("Preprocessing (variable, mean, std): %s, %s, %s",
                            var, mean, std)
    elif "robust_scaler" in config["preprocessing"]:
        scaler = preprocessing.RobustScaler().fit(x_scaler)
        for var, mean, std in zip(variables, scaler.center_,
                                    scaler.scale_):
            logger.debug("Preprocessing (variable, mean, std): %s, %s, %s",
                            var, mean, std)
    elif "min_max_scaler" in config["preprocessing"]:
        scaler = preprocessing.MinMaxScaler(
            feature_range=(-1.0, 1.0)).fit(x_scaler)
        for var, min_, max_ in zip(variables, scaler.data_min_,
                                    scaler.data_max_):
            logger.debug("Preprocessing (variable, min, max): %s, %s, %s",
                            var, min_, max_)
    elif "quantile_transformer" in config["preprocessing"]:
        scaler = preprocessing.QuantileTransformer(
            output_distribution="normal",
            random_state=int(config["seed"])).fit(x_scaler)
    else:
        logger.fatal("Preprocessing %s is not implemented.",
                        config["preprocessing"])
        raise Exception
    x[:, :len(variables)] = scaler.transform(x_scaler)

    path_preprocessing = os.path.join(
        output_dir,
        "fold{}_keras_preprocessing.pickle".format(args.fold))
    logger.info("Write preprocessing object to %s.", path_preprocessing)
    pickle.dump(scaler, open(path_preprocessing, 'wb'))
    
    # Split data in training and testing
    x_train, x_test, y_train, y_test, w_train, w_test, z_train, z_test = model_selection.train_test_split(
        x,
        y,
        w,
        z,
        test_size=1.0 - config["train_test_split"],
        random_state=int(config["seed"]))
    logger.info("Total number of events in training data: {}".format(tf.shape(x_train)[0]))
    del x, y, w
    # Add callbacks
    callbacks = []
    if "early_stopping" in config["model"]:
        logger.info("Stop early after %s tries.",
                    config["model"]["early_stopping"])
        callbacks.append(
            EarlyStopping(patience=config["model"]["early_stopping"]))

    logger.info("Saving tensorboard logs in {}".format(output_dir + "tensorboard_logs_fold{}".format(args.fold)))
    # dir for tensorboard
    path_tensorboard = os.path.join(
            output_dir,
            "tensorboard_logs_fold{}".format(args.fold))
    callbacks.append(
        TensorBoard(log_dir=path_tensorboard,
                    profile_batch="2, {}".format(
                        config["model"]["steps_per_epoch"])))

    path_model = os.path.join(output_dir,
                              "fold{}_keras_model.h5".format(args.fold))
    #This check is ignored when run on the cluster as the model is not imported into the job
    # if os.path.exists(path_model):
    #     logger.info("Path {} already exists! I will not overwrite it".format(
    #         path_model))
    #     raise Exception
    if "save_best_only" in config["model"]:
        if config["model"]["save_best_only"]:
            logger.info("Write best model to %s.", path_model)
            callbacks.append(
                ModelCheckpoint(path_model, save_best_only=True, verbose=1))

    if "reduce_lr_on_plateau" in config["model"]:
        logger.info("Reduce learning-rate after %s tries.",
                    config["model"]["reduce_lr_on_plateau"])
        callbacks.append(
            ReduceLROnPlateau(patience=config["model"]["reduce_lr_on_plateau"],
                              verbose=1))

    # Train model
    if not hasattr(keras_models, config["model"]["name"]):
        logger.fatal("Model %s is not implemented.", config["model"]["name"])
        raise Exception
    logger.info("Train keras model %s.", config["model"]["name"])

    if config["model"]["eventsPerClassAndBatch"] < 0:
        eventsPerClassAndBatch = x_train.shape[0] * len(classes)
    else:
        eventsPerClassAndBatch = config["model"]["eventsPerClassAndBatch"]

    ###
    classIndexDict = {
        label: np.where(y_train[:, i_class] == 1)[0]
        for i_class, label in enumerate(classes)
    }
    if "steps_per_epoch" in config["model"]:
        steps_per_epoch = int(config["model"]["steps_per_epoch"]) * len(eras)
        recommend_steps_per_epoch = int(
            min([len(classIndexDict[class_])
                 for class_ in classes]) / (eventsPerClassAndBatch)) + 1
        logger.info(
            "steps_per_epoch: Using {} instead of recommended minimum of {}".
            format(str(steps_per_epoch), str(recommend_steps_per_epoch)))
    else:
        logger.info("model: steps_per_epoch: Not found in {} ".format(
            args.config))
        raise Exception
    with distribution_strategy.scope():
        model_impl = getattr(keras_models, config["model"]["name"])
        model = model_impl(len(variables) + len_eras, len(classes))
        model.summary()

    if (args.balance_batches):
        logger.info("Running on balanced batches.")

        # 
        def calculateValidationWeights(x_test, y_test, w_test):
            testIndexDict = {
                era: {
                    label: np.where((z_test[:, i_era] == 1)
                                    & (y_test[:, i_class] == 1))[0]
                    for i_class, label in enumerate(classes)
                }
                for i_era, era in enumerate(eras)
            }

            y_collect = np.concatenate([
                y_test[testIndexDict[era][label]] for label in classes
                for era in eras
            ])
            x_collect = np.concatenate([
                x_test[testIndexDict[era][label], :] for label in classes
                for era in eras
            ])
            w_collect = np.concatenate([
                w_test[testIndexDict[era][class_]] *
                (len(x_test) / np.sum(np.absolute(w_test[testIndexDict[era][class_]])))
                for class_ in classes for era in eras
            ])
            w_collect = np.concatenate([
                w_test[testIndexDict[era][class_]] *
                (len(x_test) / np.sum(w_test[testIndexDict[era][class_]]))
                for class_ in classes for era in eras
            ])

            return x_collect, y_collect, w_collect

        # logger.info(x_test[:10])
        x_test_rew, y_test_rew, w_test_rew = calculateValidationWeights(
            x_test, y_test, w_test)
        # logger.info(x_test_rew[:10])


        # Loop over all eras and classes to divide the batch equally, defines the indices of the correct answers for each era/class combination
        eraIndexDict = {
            era: {
                label: np.where((z_train[:, i_era] == 1)
                                & (y_train[:, i_class] == 1))[0]
                for i_class, label in enumerate(classes)
            }
            for i_era, era in enumerate(eras)
        }
        # Generator to generate batches for balanced batch training
        from tensorflow.keras.utils import Sequence
        class balancedBatchGenerator(Sequence):
            def __init__(self):
                self.nperClass = int(eventsPerClassAndBatch)
                self.add_perClass = self.nperClass
                self.classes = classes
                self.eras = eras
                if config["model"]["cut_fraction"] > 0:
                    self.do_extbatch = True
                    self.decision_frac = config["model"]["cut_fraction"]
                    self.all_weights_mean_frac = [
                        self.decision_frac * np.absolute(np.sum(w_train[eraIndexDict[era][label]])) / len(w_train[eraIndexDict[era][label]])
                            for label in self.classes for era in self.eras
                        ]        
                else:
                    self.do_extbatch = False

            def __len__(self):
                return eventsPerClassAndBatch * len(self.classes) * max(1, len(self.eras)) * config["model"]["epochs"]
            def get_index_list(self, elem_per_era_class, prev_list={}):
                if prev_list=={}:
                    prev_list = {era: {label: [] for label in self.classes} for era in self.eras}
                selIdxDict = {
                    era: {
                        label: [*prev_list[era][label], *eraIndexDict[era][label][np.random.randint(
                            0, len(eraIndexDict[era][label]), elem_per_era_class)]]
                        for label in self.classes
                    }
                    for era in self.eras
                }
                return selIdxDict

            def __getitem__(self, idx):
                selIdxDict_todo = True
                # Choose eventsPerClassAndBatch events randomly for each class and era
                selIdxDict = {}
                tot_nperClass = self.nperClass
                next_nperClass = self.nperClass
                counter = 0
                if self.do_extbatch:
                    #Enforcing limit on lower bound of sum of weights
                    while selIdxDict_todo:
                        selIdxDict = self.get_index_list(next_nperClass, selIdxDict)
                        current_weight_sums = [
                            np.sum(w_train[selIdxDict[era][label]]) / len(w_train[selIdxDict[era][label]]) 
                                for label in self.classes for era in self.eras
                            ]
                        if not np.any(np.less(current_weight_sums, self.all_weights_mean_frac)):
                            selIdxDict_todo = False
                        else:
                            next_nperClass = self.add_perClass
                            tot_nperClass += self.add_perClass
                else:
                    #Previous implementation without lower limits
                    selIdxDict= self.get_index_list(self.nperClass, selIdxDict)
                y_collect = np.concatenate([
                    y_train[selIdxDict[era][label]] for label in self.classes
                    for era in self.eras
                ])
                x_collect = np.concatenate([
                    x_train[selIdxDict[era][label], :] for label in self.classes
                    for era in self.eras
                ])
                w_collect = np.concatenate([
                    w_train[selIdxDict[era][label]] *
                    (tot_nperClass /
                    np.sum(w_train[selIdxDict[era][label]]))
                    for label in self.classes for era in self.eras
                ])
                # return list of choosen values
                return  x_collect, y_collect, w_collect

        # Function to distribute batch generation across multiple CPU threads
        def multithreaded_from_generator(generator, output_signature, parallel_iterations=1):
            # Get batches normal if only a single CPU is available to reduce overhead
            if parallel_iterations<=1:
                return tf.data.Dataset.from_generator(generator, output_signature=output_signature)
            # Distribute batch generation with Round-robin scheduling
            else:
                def dataset_shard(index):
                    return tf.data.Dataset.from_generator(generator, output_signature=output_signature)
                return tf.data.Dataset.range(parallel_iterations).interleave(dataset_shard, 
                    cycle_length=parallel_iterations, num_parallel_calls=parallel_iterations)

        # define tf.data.Dataset for input generator data
        gen_output_signature = (tf.TensorSpec(
                                    shape=(None, len(variables) + len_eras),
                                    dtype=tf.float64),
                                tf.TensorSpec(
                                    shape=(None,
                                            len(classes)),
                                    dtype=tf.float64),
                                tf.TensorSpec(
                                    shape=None,
                                    dtype=tf.float64)
                                )

        # define tf.data.Dataset for training data
        tfx_train = tf.constant(x_train)
        tfy_train = tf.constant(y_train)
        tfw_train = tf.constant(w_train)
        IndexArray = [[np.where((z_train[:, i_era] == 1) & (y_train[:, i_class] == 1))[0] for i_class in range(len(classes))] for i_era in range(len(eras))]
        tfIndexArray = tf.ragged.constant(IndexArray, dtype=tf.int32)
        tf_len_classes = tf.constant(len(classes))
        tf_len_eras = tf.constant(len(eras))
      
        @tf.function
        def get_batch(input_para):
            tftotal_classes = tf_len_classes * tf_len_eras
            tfx_gatherer = tf.TensorArray(tf.float64, size=tftotal_classes)
            tfy_gatherer = tf.TensorArray(tf.float64, size=tftotal_classes)
            tfw_gatherer = tf.TensorArray(tf.float64, size=tftotal_classes)
            for perEraIndex_i in range(tfIndexArray.shape[0]):
                for perClassIndex_i in range(tfIndexArray[perEraIndex_i].shape[0]):
                    ind_of_class_era = tfIndexArray[perEraIndex_i][perClassIndex_i]
                    abs_index = perEraIndex_i * tf_len_classes + perClassIndex_i
                    tmp_rand_int = tf.random.uniform((eventsPerClassAndBatch,), dtype=tf.int32, minval=0, maxval=ind_of_class_era.shape[0])
                    gathered_ind = tf.gather(ind_of_class_era, tmp_rand_int)
                    gathered_w = tf.gather(tfw_train, gathered_ind)
                    tfx_gatherer = tfx_gatherer.write(abs_index, tf.gather(tfx_train, gathered_ind))
                    tfy_gatherer = tfy_gatherer.write(abs_index, tf.gather(tfy_train, gathered_ind))
                    tfw_gatherer = tfw_gatherer.write(abs_index, gathered_w * (eventsPerClassAndBatch / tf.math.reduce_sum(gathered_w)))
            return tfx_gatherer.concat(), tfy_gatherer.concat(), tfw_gatherer.concat()

        # data_length = range(eventsPerClassAndBatch * len(classes) * len(eras) * config["model"]["epochs"])
        # traindata = tf.data.Dataset.from_generator(lambda: data_length, tf.uint8)
        # traindata = traindata.map(get_batch, num_parallel_calls=m_threads)


        # gen_output_signature = (tf.TensorSpec(shape=(None),dtype=tf.int32))

        # traindata_single = multithreaded_from_generator(balancedBatchGenerator, gen_output_signature, 1)
        # traindata_multi = multithreaded_from_generator(balancedBatchGenerator, gen_output_signature, m_threads)



        use_multithread_generator = True
        if len(physical_GPU_devices)>0 & use_multithread_generator :
            logger.info("Using multithreaded generator")
            traindata = multithreaded_from_generator(balancedBatchGenerator, gen_output_signature, m_threads)
            # traindata = tf.data.Dataset.from_generator(balancedBatchGenerator, output_signature=gen_output_signature)
        else:
            logger.info("Using singlethreaded generator")
            traindata = tf.data.Dataset.from_generator(balancedBatchGenerator, output_signature=gen_output_signature)

        # define tf.data.Dataset for validation data
        validata = tf.data.Dataset.from_tensor_slices((x_test_rew, y_test_rew, w_test_rew))
        # collect all data into single batch
        validata = validata.batch(len(x_test_rew))

        # Prefetch datasets to CPU or GPU if available
        # Prefetch batches equal to the number of CPU cores to keep CPUs busy after initial startup delay
        num_prefetch_train = max(m_threads, 2)
        do_prefetch = False
        if do_prefetch:
            logger.info("Using prefetching")
            if len(physical_GPU_devices)>0:
                traindata = traindata.apply(
                    tf.data.experimental.prefetch_to_device(
                        tf.test.gpu_device_name(), num_prefetch_train))
                validata = validata.apply(
                    tf.data.experimental.prefetch_to_device(
                        tf.test.gpu_device_name(), tf.data.experimental.AUTOTUNE))
            else:
                traindata = traindata.prefetch(tf.data.experimental.AUTOTUNE)
                validata = validata.prefetch(tf.data.experimental.AUTOTUNE)
        else:
            logger.info("Not using prefetching")

        time_start = datetime.now()
        logger.info("Timestamp training start: {}".format(time_start.strftime("%H:%M:%S")))
        # Train model with prefetched datasets
        if len(physical_GPU_devices)>0:
            logger.info("Now training with {} GPUs.".format(len(physical_GPU_devices)))
            history = model.fit(traindata,
                                steps_per_epoch=steps_per_epoch,
                                epochs=config["model"]["epochs"],
                                callbacks=callbacks,
                                validation_data=validata,
                                verbose=2)
        else:
            logger.info("Now training with no GPUs.")
            history = model.fit(traindata,
                                steps_per_epoch=steps_per_epoch,
                                epochs=config["model"]["epochs"],
                                callbacks=callbacks,
                                validation_data=validata,
                                max_queue_size=10,
                                workers=5,
                                verbose=2)

    else:
        # Perform data transformations on gpu if available
        if len(physical_GPU_devices)>0:
            device = '/GPU:0'
        else:
            device = '/CPU'
        with tf.device(device):
            traindata = tf.data.Dataset.from_tensor_slices(
                (x_train, y_train, w_train))
            traindata = traindata.batch(eventsPerClassAndBatch * len(classes))
            validata = tf.data.Dataset.from_tensor_slices(
                (x_test_rew, y_test_rew, w_test))
            validata = validata.batch(len(x_test_rew))
            if len(physical_GPU_devices)>0:
                traindata = traindata.apply(
                    tf.data.experimental.prefetch_to_device(
                        tf.test.gpu_device_name(),
                        tf.data.experimental.AUTOTUNE))
                validata = validata.apply(
                    tf.data.experimental.prefetch_to_device(
                        tf.test.gpu_device_name(),
                        tf.data.experimental.AUTOTUNE))
            else:
                traindata = traindata.prefetch(tf.data.experimental.AUTOTUNE)
                validata = validata.prefetch(tf.data.experimental.AUTOTUNE)

        time_start = datetime.now()
        logger.info("Timestamp training start: {}".format(time_start.strftime("%H:%M:%S")))
        history = model.fit(traindata,
                            validation_data=validata,
                            epochs=config["model"]["epochs"],
                            shuffle=True,
                            callbacks=callbacks,
                            verbose=2)

    time_end = datetime.now()
    logger.info("Timestamp training end: {}".format(time_end.strftime("%H:%M:%S")))
    timediff = time_end-time_start
    overflow = 0
    if timediff.total_seconds() < 0:
        overflow = 24 * 60 * 60
    logger.info("The training took {} minutes.".format((timediff.total_seconds()+overflow)/60.))
    epochs = range(1, len(history.history["loss"]) + 1)
    axes = plt.gca()
    axes.set_ylim([0, history.history["val_loss"][0]+1])
    plt.plot(epochs, history.history["loss"], lw=3, label="Training loss")
    plt.plot(epochs,
             history.history["val_loss"],
             lw=3,
             label="Validation loss")
    plt.xlabel("Epoch"), plt.ylabel("Loss")
    path_plot = os.path.join(output_dir,
                             "fold{}_loss".format(args.fold))
    plt.legend()
    plt.savefig(path_plot + ".png", bbox_inches="tight")
    plt.savefig(path_plot + ".pdf", bbox_inches="tight")

    # Save model
    if not "save_best_only" in config["model"]:
        logger.info("Write model to %s.", path_model)
        model.save(path_model)

if __name__ == "__main__":
    args = parse_arguments()
    config_path = "{}/training_config.yaml".format(args.config_dir)
    config = parse_config(config_path)
    setup_logging(logging.INFO)
    main(args, config)
