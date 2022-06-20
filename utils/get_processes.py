#This script currently only supports the NMSSM analysis
#Other process behaviours can be extracted from write_dataset_config.py

import argparse
import logging

logger = logging.getLogger("write_dataset_config")
logger.setLevel(logging.DEBUG)
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
        "--mass", 
        required=True, 
        help="Mass of mH"
    )
    parser.add_argument(
        "--batch", 
        required=True, 
        help="Batch to select mh' mass"
    )
    parser.add_argument(
        "--training-z-estimation-method",
        required=True,
        help="Estimate the Z bkg with emb (embedding) or mc (Monte Carlo) ?"
    )
    parser.add_argument(
        "--training-jetfakes-estimation-method",
        required=True,
        help="Estimate the jet fakes with ff (FakeFactor) or mc (Monte Carlo) ?"
    )
    return parser.parse_args()


def main(channel, mass, batch, training_z_estimation_method, training_jetfakes_estimation_method):
    #check for conflicting arguments
    if (training_z_estimation_method != "mc" and 
        training_z_estimation_method != "emb"):
        logger.fatal(
            "No valid training-z-estimation-method! "\
            "Options are emb, mc. Argument was {}".format(
                training_z_estimation_method))
        raise Exception
    
    if (training_jetfakes_estimation_method != "ff" and 
        training_jetfakes_estimation_method != "mc"):
        logger.fatal(
            "No valid training-jetfakes-estimation-method! "\
            "Options are ff, mc. Argument was {}".format(
            training_jetfakes_estimation_method))
        raise Exception

    if (training_jetfakes_estimation_method == "ff" and
        channel == "em"):
        logger.warn("ff+em: using mc for em channel")

    process_dict = {}

    #NMSSM processes
    mass = int(mass)
    batch = int(batch)
    batches = {}
    if mass<=1000:
        batches[1] = [60, 70, 75, 80]
        batches[2] = [85, 90, 95, 100]
        batches[3] = [110, 120, 130, 150]
        batches[4] = [170, 190, 250, 300]
        batches[5] = [350, 400, 450, 500]
        batches[6] = [550, 600, 650, 700]
        batches[7] = [750, 800, 850]
    else:
        batches[1] = [60, 70, 80, 90, 100]
        batches[2] = [120, 150, 170, 190, 250, 300]
        batches[3] = [350, 400, 450, 500, 550, 600, 650, 700]
        batches[4] = [800, 900, 1000, 1100, 1200]
        batches[5] = [1300, 1400, 1600, 1800]
        batches[6] = [2000, 2200, 2400, 2600, 2800]
    light_masses = [val for val in batches[batch] if val+125<mass]
    for light_mass in light_masses:
        process_dict[
            "NMSSM_{}_125_{}".format(mass, light_mass)
        ] = "NMSSM_MH{}_{}".format(mass, batch)

    #EWKZ process
    process_dict["EWKZ"] = "misc"

    #VVL process
    process_dict["VVL"] = "misc"

    #TTL process
    process_dict["TTL"] = "tt"

    #ZL process
    process_dict["ZL"] = "misc"

    #ggH125 process
    process_dict["ggH125"] = "misc"
    
    #ggH125 process
    process_dict["qqH125"] = "misc"

    #EMB process
    if training_z_estimation_method == "emb":
        process_dict["EMB"] = "emb"

    #ZTT process
    if training_z_estimation_method == "mc":
        process_dict["ZTT"] = "ztt"

    #TTT process
    if training_z_estimation_method == "mc":
        if channel == "tt":
            process_dict["TTT"] = "misc"
        else:
            process_dict["TTT"] = "tt"

    #VTT process
    if training_z_estimation_method == "mc":
        if channel == "em":
            process_dict["VTT"] = "db"
        else:
            process_dict["VTT"] = "misc"

    #ff process
    if (training_jetfakes_estimation_method == "ff" and 
        channel != "em"):
        process_dict["ff"] = "ff"

    #TTJ process
    if (training_jetfakes_estimation_method == "mc" or 
        channel == "em"):
        if channel == "tt":
            process_dict["TTJ"] = "misc"
        elif channel != "em":
            process_dict["TTJ"] = "tt"

    #ZJ process
    if (training_jetfakes_estimation_method == "mc" or 
        channel == "em"):
        if channel == "tt":
            process_dict["TTJ"] = "misc"
        elif channel != "em":
            process_dict["TTJ"] = "zll"

    #VVJ process
    if (training_jetfakes_estimation_method == "mc" and 
        channel != "em"):
        process_dict["VVJ"] = "misc"

    #W process
    if (training_jetfakes_estimation_method == "mc" or 
        channel == "em"):
        if channel in ["et", "mt"]:
            process_dict["W"] = "w"
        else:
            process_dict["W"] = "misc"

    #QCD process
    if (training_jetfakes_estimation_method == "mc" or 
        channel == "em"):
        if channel == "tt":
            process_dict["QCD"] = "noniso"
        else:
            process_dict["QCD"] = "ss"

    # return list of (process, class) tuples of all processes used for the analysis
    return [(process, process_dict[process]) for process in process_dict]
    
if __name__ == "__main__":
    args = parse_arguments()
    print(main(args.channel, args.mass, args.batch, args.training_z_estimation_method, args.training_jetfakes_estimation_method))