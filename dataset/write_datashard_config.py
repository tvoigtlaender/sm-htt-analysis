#!/usr/bin/env python

import argparse
import logging
import re
import yaml
import importlib
from shape_producer.channel import *


from shape_producer.cutstring import Weights, Weight


logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--channel", 
        required=True, 
        help="Analysis channel"
    )
    parser.add_argument(
        "--era", 
        required=True, 
        help="Experiment era"
    )
    parser.add_argument(
        "--processes",
        nargs='+',
        default=[],
        help="Process of which the config will be created."
    )
    parser.add_argument(
        "--process-classes",
        nargs='+',
        default=[],
        help="Class of process of which the config will be created."
    )
    parser.add_argument(
        "--base-path", 
        required=True, 
        help="Path to Artus output files"
    )
    parser.add_argument(
        "--friend-paths", 
        nargs='+', 
        default=[], 
        help="Additional paths to Artus output friend files"
    )
    parser.add_argument(
        "--event-branch", 
        required=True, 
        help="Branch with event numbers"
    )
    parser.add_argument(
        "--training-weight-branch",
        required=True,
        help="Branch with training weights"
    )
    parser.add_argument(
        "--database", 
        required=True, 
        help="Kappa datsets database."
    )
    parser.add_argument(
        "--tree-path", 
        required=True, 
        help="Path to tree in ROOT files"
    )
    parser.add_argument(
        "--output-config-base", 
        required=True, 
        help="Output dataset config file"
    )
    return parser.parse_args()

def main(args):
    logger.debug("Parsed arguments: {}".format(args))
    assert (len(args.processes)==len(args.process_classes)), \
    "Number of processes and process classes is not equal({},{})".format(
        len(args.processes),len(args.process_classes)
    )
    output_config = {}
    output_config["base_path"] = args.base_path.format(channel=args.channel)
    output_config["friend_paths"] = [path.format(channel=args.channel) for path in args.friend_paths]
    output_config["event_branch"] = args.event_branch
    output_config["training_weight_branch"] = args.training_weight_branch
    output_config["tree_path"] = args.tree_path

    # print(output_config)

    if "2016" in args.era:
        import shape_producer.estimation_methods_2016 as estimation_methods
        from shape_producer.era import Run2016
        era = Run2016(args.database)

    if "2017" in args.era:
        import shape_producer.estimation_methods_2017 as estimation_methods
        from shape_producer.era import Run2017
        era = Run2017(args.database)

    if "2018" in args.era:
        import shape_producer.estimation_methods_2018 as estimation_methods
        from shape_producer.era import Run2018
        era = Run2018(args.database)

    channelDict = {}
    channelDict["2016"] = {"mt": MTSM2016(), "et": ETSM2016(), "tt": TTSM2016(), "em": EMSM2016()}
    channelDict["2017"] = {"mt": MTSM2017(), "et": ETSM2017(), "tt": TTSM2017(), "em": EMSM2017()}
    channelDict["2018"] = {"mt": MTSM2018(), "et": ETSM2018(), "tt": TTSM2018(), "em": EMSM2018()}
    
    
    for process, process_class in zip(args.processes, args.process_classes):
        logger.info("Channel: {}, Era: {}, Process: {}".format(args.channel, args.era, process))
        
        channel = channelDict[args.era][args.channel]
        additional_cuts = Cuts()
        output_config["processes"] = {}
        additionalWeights = Weights("", "default")

        if re.findall("(NMSSM_[0-9]+_125_[0-9]+)", process):
            heavy_mass, light_mass = re.findall("NMSSM_([0-9]+)_125_([0-9]+)", process)[0]
            estimation = estimation_methods.NMSSMEstimation(
                era,
                args.base_path,
                channel,
                heavy_mass=heavy_mass,
                light_mass=light_mass
            )

        elif process == "EMB":
            estimation = estimation_methods.ZTTEmbeddedEstimation(era, args.base_path, channel)

        elif process == "EWKZ":
            estimation = estimation_methods.EWKZEstimation(era, args.base_path, channel)

        elif process == "TTL":
            estimation = estimation_methods.TTLEstimation(era, args.base_path, channel)

        elif process == "VVL":
            estimation = estimation_methods.VVLEstimation(era, args.base_path, channel)

        elif process == "ZL":
            estimation = estimation_methods.ZLEstimation(era, args.base_path, channel)

        elif process == "ff":
            estimation = estimation_methods.DataEstimation(era, args.base_path, channel)
            if args.channel in ["et", "mt"]:
                channel_Cut = Cut(
                    "byMediumDeepTau2017v2p1VSjet_2<0.5&&byVVVLooseDeepTau2017v2p1VSjet_2>0.5",
                    "tau_aiso")
                fakeWeightstring = "ff2_nom"
                channel.cuts.remove("tau_iso")
            elif args.channel == "tt":
                channel_Cut = Cut(
                    "(byMediumDeepTau2017v2p1VSjet_2>0.5&&byMediumDeepTau2017v2p1VSjet_1<0.5&&byVVVLooseDeepTau2017v2p1VSjet_1>0.5)||(byMediumDeepTau2017v2p1VSjet_1>0.5&&byMediumDeepTau2017v2p1VSjet_2<0.5&&byVVVLooseDeepTau2017v2p1VSjet_2>0.5)",
                    "tau_aiso")
                fakeWeightstring = "(0.5*ff1_nom*(byMediumDeepTau2017v2p1VSjet_1<0.5)+0.5*ff2_nom*(byMediumDeepTau2017v2p1VSjet_2<0.5))"
                channel.cuts.remove("tau_1_iso")
                channel.cuts.remove("tau_2_iso")
            channel.cuts.add(channel_Cut)
            additionalWeights = Weights(Weight(fakeWeightstring, "fake_factor"))

        elif process == "ggH125":
            estimation = estimation_methods.ggHEstimation("ggH125", era, args.base_path, channel)

        elif process == "qqH125":
            estimation = estimation_methods.qqHEstimation("qqH125", era, args.base_path, channel)

        elif process == "ZTT":
            estimation = estimation_methods.ZTTEstimation(era, args.base_path, channel)

        elif process == "TTT":
            estimation = estimation_methods.TTTEstimation(era, args.base_path, channel)

        elif process == "VVT":
            estimation = estimation_methods.VVTEstimation(era, args.base_path, channel)

        elif process == "VVJ":
            estimation = estimation_methods.VVJEstimation(era, args.base_path, channel)

        elif process == "ZJ":
            estimation = estimation_methods.ZJEstimation(era, args.base_path, channel)

        elif process == "TTJ":
            estimation = estimation_methods.TTJEstimation(era, args.base_path, channel)

        elif process == "W":
            estimation = estimation_methods.WEstimation(era, args.base_path, channel)

        elif process == "QCD":
            # Same sign selection for data-driven QCD
            estimation = estimation_methods.DataEstimation(era, args.base_path, channel)
            if args.channel != "tt":
                ## os= opposite sign
                channel.cuts.get("os").invert()
            # Same sign selection for data-driven QCD
            else:
                channel.cuts.remove("tau_2_iso")
                channel.cuts.add(
                    Cut("byMediumDeepTau2017v2p1VSjet_2<0.5", "tau_2_iso"))
                channel.cuts.add(
                    Cut("byMediumDeepTau2017v2p1VSjet_2>0.5", "tau_2_iso_loose"))
        
        else:
            logger.fatal(
                "Process {} with class {} not recognized.".format(
                    process, process_class
                )
            )
            raise Exception

        # Additional cuts for NMSSM
        additional_cuts = Cuts(Cut("nbtag>0", "nbtag"))
        logger.debug("Use additional cuts: %s", additional_cuts.expand())

        #Gather process dict
        output_config["processes"][process] = {
            "files": [
                str(f).replace(args.base_path.rstrip("/") + "/", "")
                for f in estimation.get_files()
            ],
            "cut_string": (estimation.get_cuts() + channel.cuts +
                            additional_cuts).expand(),
            "weight_string":
                (estimation.get_weights() + additionalWeights).extract(),
            "class": process_class
        }

        #Write config to file
        logger.info("Writing config to {}".format("{}/{}_datashard_config.yaml".format(args.output_config_base, process)))
        yaml.dump(output_config, open("{}/{}_{}/{}_datashard_config.yaml".format(
            args.output_config_base, 
            args.era,
            args.channel,
            process), 'w'
        ), default_flow_style=False)

if __name__ == "__main__":
    args = parse_arguments()
    main(args)