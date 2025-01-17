#!/usr/bin/env python

from XRootD import client
from XRootD.client.flags import QueryCode
import ROOT

ROOT.PyConfig.IgnoreCommandLineOptions = True  # disable ROOT internal argument parser
#  ROOT.ROOT.EnableImplicitMT(12); # Tell ROOT you want to go parallel
import argparse
import yaml
import os
from array import array
from re import findall
import os

import logging

logger = logging.getLogger("create_training_datashard")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

def check_if_remote_file_exists(fullpath):
    client_path = findall('(root://.*/)/', fullpath)[0]
    file_path = findall('root://.*/(/.*)', fullpath)[0]
    myclient = client.FileSystem(client_path)
    status, response = myclient.query(QueryCode.CHECKSUM, file_path, timeout=10)
    if status == '':
        logger.fatal("XRootD checksum query did not respond")
        raise Exception
    if status.status == 0:
        return 1
    else:
        return 0

def parse_arguments():
    logger.debug("Parse arguments.")
    parser = argparse.ArgumentParser(description="Create training dataset")
    parser.add_argument("--config", help="Datashard config file")
    parser.add_argument("--process", help="Process of datashard")
    parser.add_argument("--fold", help="Fold of datashard (0 or 1)")
    parser.add_argument("--output-dir", help="Root file created by this script")
    return parser.parse_args()


def parse_config(filename):
    logger.debug("Load YAML config: {}".format(filename))
    return yaml.load(open(filename, "r"), Loader=yaml.SafeLoader)


def main(args, config):
    logger.info(args.process)
    logger.debug("Collect events of process {} for fold {}.".format(args.process, args.fold))

    # Create output file
    output_filename = os.path.join(
        args.output_dir, "{}_datashard_fold{}.root".format(args.process, args.fold)
    )

    # Collect all files for this process in a chain. Create also chains for friend files
    chain = ROOT.TChain(config["tree_path"])  ## "mt_nominal/ntuple"
    friendchains = {}
    for friendPath in config["friend_paths"]:  ####/ceph/htautau/2017/nnscore_friends/
        friendTreeName = os.path.basename(os.path.normpath(friendPath))
        friendchains[friendTreeName] = ROOT.TChain(config["tree_path"])

    # for each file, add ntuple TTree to the chain and do the same for the the friendTrees
    for filename in config["processes"][args.process]["files"]:
        path = os.path.join(config["base_path"], filename )
        if not check_if_remote_file_exists(path):
            logger.fatal("File does not exist: {}".format(path))
            raise Exception
        else:
            logger.debug("File {} exists.".format(path))

        chain.AddFile(path)
        # Make sure, that friend files are put in the same order together
        for friendPath in config["friend_paths"]:
            friendFileName = os.path.join(friendPath, filename)
            if not check_if_remote_file_exists(friendFileName):
                logger.fatal("File does not exist: {}".format(friendFileName))
                raise Exception
            else:
                logger.debug("File {} exists.".format(friendFileName))

            friendTreeName = os.path.basename(os.path.normpath(friendPath))
            logger.debug(
                "Attaching friendtree for {}, filename{}".format(
                    friendTreeName, friendFileName
                )
            )
            friendchains[friendTreeName].AddFile(friendFileName)
    logger.debug("Joining TChains")
    for friendTreeName in friendchains.keys():
        logger.debug("Adding to mainchain: {}".format(friendTreeName))
        chain.AddFriend(friendchains[friendTreeName], friendTreeName)

    logger.debug("Calculationg number of events")
    # Disable branch "nickname" of type "string" to prevent futile searching
    chain.SetBranchStatus("nickname",0)
    rdf = ROOT.RDataFrame(chain)
    chain_numentries = rdf.Count().GetValue()
    if chain_numentries == 0:
        logger.fatal("Chain (before skimming) does not contain any events.")
        raise Exception
    logger.info("Found {} events for process {}.".format(chain_numentries, args.process))

    # Skim the events with the cut string
    cut_string = "({EVENT_BRANCH}%2=={NUM_FOLD})&&({CUT_STRING})".format(
        EVENT_BRANCH=config["event_branch"],
        NUM_FOLD=args.fold,
        CUT_STRING=config["processes"][args.process]["cut_string"],
    )
    logger.debug("Skim events with cut string: {}".format(cut_string))

    rdf = rdf.Filter(cut_string)

    chain_skimmed_numentries = rdf.Count().GetValue()
    if not chain_skimmed_numentries > 0:
        logger.fatal("Chain (after skimming) does not contain any events.")
        raise Exception
    logger.debug(
        "Found {} events for process {} after skimming.".format(
            chain_skimmed_numentries, args.process
        )
    )

    # # Write training weight to new branch
    logger.debug(
        "Add training weights with weight string: {}".format(
            config["processes"][args.process]["weight_string"]
        )
    )
    rdf = rdf.Define(
        config["training_weight_branch"],
        "(float)(" + config["processes"][args.process]["weight_string"] + ")",
    )

    opt = ROOT.ROOT.RDF.RSnapshotOptions()
    opt.fMode = "RECREATE"

    logger.info("Creating output file: {}".format(output_filename))
    rdf.Snapshot(config["processes"][args.process]["class"], output_filename, "^((?!nickname).)*$", opt)
    logger.info("snapshot created for process {}!".format(args.process))


if __name__ == "__main__":
    args = parse_arguments()
    config = parse_config(args.config)
    main(args, config)
