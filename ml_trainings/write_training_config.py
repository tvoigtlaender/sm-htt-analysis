#!/usr/bin/env python

import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
import argparse
import logging
import numpy as np
from get_processes import main as get_processes
logger = logging.getLogger("sum_training_weights")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

### YAML + ORDERED DICT MAGIC
from collections import OrderedDict
import yaml
from yaml import Loader, Dumper
from yaml.representer import SafeRepresenter
from functools import reduce

def dict_representer(dumper, data):
   return dumper.represent_dict(data.items())
Dumper.add_representer(OrderedDict, dict_representer)

def dict_constructor(loader, node):
    return OrderedDict(loader.construct_pairs(node))
Loader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, dict_constructor)

#Class to allow input arguments to be read as dict
class StoreDictKeyPair(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        self._nargs = nargs
        super(StoreDictKeyPair, self).__init__(option_strings, dest, nargs=nargs, **kwargs)
    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        # print("values: {}".format(values))
        for kv in values:
            k,v = kv.split("=")
            #Convert value from string to float or int if possible
            try:
                v = float(v)
                if int(v) == v:
                    v = int(v)
            except:
                pass
            my_dict[k] = v
        setattr(namespace, self.dest, my_dict)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Sum training weights of classes in training dataset.")
    parser.add_argument("--era", required=True, help="Experiment era")
    parser.add_argument("--channel", required=True, help="Analysis channel")
    parser.add_argument("--masses",nargs='+', required=True, help="Analysis channel")
    parser.add_argument("--batches",nargs='+', required=True, help="Analysis channel")
    parser.add_argument("--dataset-dir", type=str,required=True, help="Path to training datasets directory.")
    parser.add_argument("--config-dir", type=str,required=True, help="Path to training datasets directory.")

    parser.add_argument("--processes",nargs='+', type=str,required=True, help="List of datashards files")
    parser.add_argument("--training-template", type=str,required=False, help="Specifies the config file setting the model, used variables...")
    parser.add_argument("--write-weights", type=bool, default=True, help="Overwrite inverse weights to ml/$era_$channel_training.yaml")
    parser.add_argument("--output-path", type=str, help="Branch with weights.")
    parser.add_argument("--overwrite-configs", action=StoreDictKeyPair, nargs="*", metavar="KEY=VAL")
    
    return parser.parse_args()


def dictToString(exdict):
    return str(["{} : {}".format(key, value) for key, value in sorted(exdict.items(), key=lambda x: str(x[1]))])

# Function to merge two dicts
def merge(a, b, path=None):
    "merges b into a"
    if path is None: path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge(a[key], b[key], path + [str(key)])
            elif a[key] == b[key]:
                pass # same leaf value
            else:
                raise Exception('Conflict at %s' % '.'.join(path + [str(key)]))
        else:
            a[key] = b[key]
    return a

def main(args):
    print(args)
    folds = ["0", "1"]
    # Create dict containing containing the files, datasets and classes of all processes
    all_dict = {
        process: {
            "config_file": "{process}_datashard_config.yaml".format(process=process),
            "datasets": ["{process}_datashard_fold{fold}.root".format(process=process, fold=fold) for fold in folds],
            "class": {}
        } for process in args.processes
    }
    # Add sum of weights to dict for all processes
    for process in args.processes:
        logger.info("Process datasets %s.", all_dict[process]["datasets"])
        f = [ROOT.TFile.Open("{}/{}".format(args.dataset_dir, filename), "READ") for filename in all_dict[process]["datasets"]]
        with open(args.config_dir + "/" +all_dict[process]["config_file"], "r") as f_dict:
            process_dict = yaml.load(f_dict, Loader=yaml.SafeLoader)
        class_name = process_dict["processes"][process]["class"]
        all_dict[process]["class"] = class_name
        sum_ = 0.0
        for f_ in f:
            tree = f_.Get(class_name)
            for event in tree:
                evw=getattr(event, process_dict["training_weight_branch"])
                if np.isnan( evw):
                    logger.fatal("Fatal: no event weight in class {} with ID {}".format(name,getattr(event, "event")))
                    raise Exception
                else:
                    sum_ += evw
        all_dict[process]["weight"] = sum_

    mass_batch_dict = {}
    # Loop over all valid mass/batch combinations
    for mass in args.masses:
        mass_batch_dict[mass] = {}
        for batch in args.batches:
            mass = str(mass)
            if mass in ["240", "280"]:
                max_batch = 3
            elif mass in ["320", "360", "400", "450"]:
                max_batch = 4
            elif mass in ["500", "550", "600"]:
                max_batch = 5
            elif mass in ["700", "800", "heavier"]:
                max_batch = 6
            elif mass in ["900", "1000"]:
                max_batch = 7
            else:
                raise Exception("Provided mass {} is not valid.".format(mass))
            if int(batch) > max_batch:
                continue

            logger.info("Processing era {}, channel {}, mass {}, batch {}".format(
                args.era, args.channel, mass, batch
            ))
            out_file = args.config_dir +"_{}_{}/training_config.yaml".format(
                mass, batch
            )
            # get list of processes and their classes used for the mass/batch
            mass_batch_processes, mass_batch_classes = zip(*get_processes(
                channel=args.channel,
                mass=mass,
                batch=batch,
                training_z_estimation_method="emb",
                training_jetfakes_estimation_method="ff"
            ))
            # Get subset dict, that only contains processes for the current mass/batch
            subset_dict = {process: all_dict[process] for process in mass_batch_processes}

            # Merge the dicts for all used shards
            data_dicts = []
            for key in subset_dict:
                data_dict_path = "{}/{}".format(args.config_dir, subset_dict[key]["config_file"])
                data_dicts.append(yaml.load(open(data_dict_path, "r"), Loader=yaml.SafeLoader))
            dsConfDict = reduce(merge, data_dicts)

            ### use the classes that have processes mapped to them
            classes = set(mass_batch_classes)

            if args.training_template == None:
                args.training_template= "ml/templates/{}_{}_training.yaml".format(args.era, args.channel)
            trainingTemplateDict=yaml.load(open(args.training_template, "r"), Loader=yaml.SafeLoader)

            ### Weight Calculation
            counts = []
            sum_all = 0.0
            for name in classes:
                logger.debug("Process class %s.", name)
                sum_ = 0.0
                for process in mass_batch_processes:
                    if subset_dict[process]["class"] == name:
                        sum_ += evw
                sum_all += sum_
                counts.append(sum_)

            ### Weight printing
            for i, name in enumerate(classes):
                logger.info(
                    "Class {} (sum, fraction, inverse): {:g}, {:g}, {:g}".format(
                        name, counts[i], counts[i] / sum_all, sum_all / counts[i]))

            logger.info( "{}-{}: Class weights before update: {}".format(args.era, args.channel,dictToString(trainingTemplateDict["class_weights"])))

            newWeightsDict={}
            for i, name in enumerate(classes):
                newWeightsDict[name]=sum_all / counts[i]

            ### Warning for big changes
            if set(list(trainingTemplateDict["class_weights"].keys()))==set(list(newWeightsDict.keys())):
                for i, name in enumerate(classes):
                    oldweight=trainingTemplateDict["class_weights"][name]
                    newweight=newWeightsDict[name]
                    if newweight/oldweight > 2 or newweight/oldweight < .5:
                        logger.warning( "{}-{}: Class weights for {} changing by more than a factor of 2".format(args.era, args.channel,name))
            else:
                logger.warning("Training classes in {} and {} differ".format(out_file, args.training_template))

            ## Sort the clases, so testing plots/... are easierer to compare
            priolist=["qqh","ggh","emb","ztt","tt","db","misc","zll","w","noniso","ss","ff"]
            odDict=OrderedDict({})
            for key in priolist:
                if key in newWeightsDict:
                    odDict[key]=newWeightsDict[key]
                    del newWeightsDict[key]
            ## add classes that are not in the priolist at the end
            for key in sorted(newWeightsDict):
                odDict[key]=newWeightsDict[key]
            del newWeightsDict

            ###
            # attach the weights dict with the classes to the dsConfDict
            dsConfDict["classes"]=list(odDict.keys())
            dsConfDict["class_weights"]=odDict


            ############ Logic for merging the configs
            if "classes" in list(trainingTemplateDict.keys()):
                if set(trainingTemplateDict["classes"])!=set(classes):
                    logger.warning("Training classes in {} and {} differ".format(
                        out_file, args.training_template
                    ))

            ## Merge dicts "classes" and "class_weights" are overwritten
            ## Anything provided by the "--overwrite-configs" argument has priority
            mergeddict=OrderedDict({})
            for key in trainingTemplateDict.keys():
                if key in ["classes", "class_weights"]:
                    mergeddict[key] = dsConfDict[key]
                else:
                    mergeddict[key] = trainingTemplateDict[key]
            mergeddict["processes"] = list(dsConfDict["processes"].keys())
            if args.overwrite_configs:
                logger.info("Given overwrite values:")
                logger.info(args.overwrite_configs)
            for key in args.overwrite_configs.keys():
                if key in mergeddict["model"].keys():
                    mergeddict["model"][key] = args.overwrite_configs[key]
                else:
                    mergeddict[key] = args.overwrite_configs[key]

            #Write config
            with open(out_file,"w") as f:
                yaml.dump(mergeddict, f,Dumper=Dumper, default_flow_style=False)

            logger.info( "{}-{}: Resulting training configs".format(args.era, args.channel))
            print(dictToString({key: value for key, value in mergeddict.items() if key != "processes"}))

            logger.info( "{}-{}: Class weights after update: {}".format(args.era, args.channel,dictToString(mergeddict["class_weights"])))


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
