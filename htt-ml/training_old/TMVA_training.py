#!/usr/bin/env python

import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True  # disable ROOT internal argument parser

import argparse
import yaml
import os

import keras_models


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train machine learning methods for Htt analyses")
    parser.add_argument("config", help="Path to training config file")
    parser.add_argument("fold", type=int, help="Select the fold to be trained")
    return parser.parse_args()


def parse_config(filename):
    return yaml.load(open(filename, "r"))


def main(args, config):
    ROOT.TMVA.Tools.Instance()
    ROOT.TMVA.PyMethodBase.PyInitialize()

    # Set up TMVA DataLoader
    factory = ROOT.TMVA.Factory(
        "TMVAMulticlass",
        ROOT.TFile.Open(
            os.path.join(config["output_path"], "fold{}_training.root".format(
                args.fold)), "RECREATE"),
        "!V:!Silent:Color:!DrawProgressBar:Transformations=None:AnalysisType=multiclass"
    )
    dataloader = ROOT.TMVA.DataLoader(
        os.path.join(config["output_path"], "fold{}_training".format(
            args.fold)))

    # Add variables
    for variable in config["variables"]:
        dataloader.AddVariable(variable)

    # Add classes with class weights and event weights
    input_file = ROOT.TFile(config["datasets"][args.fold])
    trees = {}
    for class_ in config["classes"]:
        trees[class_] = input_file.Get(class_)
        if trees[class_] == None:
            raise Exception("Tree for class {} does not exist.".format(class_))
        dataloader.AddTree(
            trees[class_], class_,
            config["class_weights"][class_] * config["global_weight_scale"])
        dataloader.SetWeightExpression(config["event_weights"], class_)

    # Set split of training dataset in training and validation set
    prepare_classes = ""
    for class_ in config["classes"]:
        prepare_classes += "TrainTestSplit_{}={}:".format(
            class_, config["train_test_split"])
    dataloader.PrepareTrainingAndTestTree(
        ROOT.TCut(""), prepare_classes + "SplitMode=Random:NormMode=None")

    # Set up Keras model
    model = keras_models.example(
        len(config["variables"]), len(config["classes"]))
    model.summary()
    model.save("fold{}_model.h5".format(args.fold))

    # Book MVA methods
    factory.BookMethod(
        dataloader, ROOT.TMVA.Types.kPyKeras, "PyKeras_fold{}".format(
            args.fold),
        "!H:!V:VarTransform=N:FilenameModel=fold{}_model.h5:".format(
            args.fold) + "SaveBestOnly=true:TriesEarlyStopping=-1:" +
        "NumEpochs={}:BatchSize={}".format(config["model"]["epochs"],
                                           config["model"]["batch_size"]))

    factory.BookMethod(
        dataloader, ROOT.TMVA.Types.kBDT, "BDT",
        "!H:!V:VarTransform=None:NTrees=1500:BoostType=Grad:Shrinkage=0.10:" +
        "UseBaggedBoost:BaggedSampleFraction=0.50:nCuts=50:MaxDepth=3:SeparationType=GiniIndex"
    )

    # Run training and evaluation
    factory.TrainAllMethods()
    factory.TestAllMethods()
    factory.EvaluateAllMethods()


if __name__ == "__main__":
    args = parse_arguments()
    config = parse_config(args.config)
    main(args, config)
