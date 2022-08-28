### Collection of functions to merge configs for composit trainings ###

# Function to collect information on the process configs
def get_id_configs(config, t_name, recursion_list=[]):
    # Check for loops in recursive resolution
    if t_name in recursion_list:
        print(
            "Loop found in recursive training definition! {} has already been called.".format(
                t_name
            )
        )
        raise Exception("Consistency error in training config.")
    else:
        tmp_list = recursion_list + [t_name]
    t_config = config[t_name]
    if "composite" in t_config.keys() and t_config["composite"]:
        # If training consists of sub-trainings
        #   Merge identifiers of all sub-trainings
        id_path = []
        for sub_t in t_config["trainings"]:
            id_path += get_id_configs(config, sub_t, tmp_list)
        # Check for identifiers with the same name, but different config dirs
        unique_id_path = set(id_path)
        ids, paths = zip(*unique_id_path)
        if len(ids) != len(set(ids)):
            print("Entry with same identifier, but different config dir found!")
            data_found = []
            for data in unique_id_path:
                id_, path = data
                if id_ in data_found:
                    print("Identifier {} is affected.".format(id_))
                else:
                    data_found.append(id_)
            raise Exception("Consistency error in training config.")
        # Return combined list of identifiers
        return list(id_path)
    else:
        # Create identifier from identification parameters
        id_string = "_".join(t_config["identifier"].values())
        # Return together with process config path
        return [(id_string, t_config["processes_config"])]


# Function to collect information on the training model configs
def get_model_configs(config, t_name, recursion_list=[]):
    # Check for loops in recursive resolution
    if t_name in recursion_list:
        print(
            "Loop found in recursive training definition! {} has already been called.".format(
                t_name
            )
        )
        raise Exception("Consistency error in training config.")
    else:
        tmp_list = recursion_list + [t_name]
    t_config = config[t_name]
    if "composite" in t_config.keys() and t_config["composite"]:
        # If training consists of sub-trainings
        #   Collect training config information of all sub-trainings
        tmp_conf_dict = {}
        for training in t_config["trainings"]:
            tmp_conf_dict[training] = get_model_configs(config, training, tmp_list)
        # Collect keys of all sub-trainings and the overwrite
        if "model" in t_config.keys() and t_config["model"]:
            overwrite = True
            all_model_keys = list(t_config["model"].keys())
        else:
            overwrite = False
            all_model_keys = []
        for conf in tmp_conf_dict.values():
            all_model_keys.extend(conf.keys())
        all_model_keys = list(set(all_model_keys))
        # Create new combined training config
        new_model_config = {}
        for key in all_model_keys:
            if overwrite and key in t_config["model"].keys():
                # Use overwrite when possible
                new_model_config[key] = t_config["model"][key]
            else:
                # Check if parameter is present in all sub-trainings
                for conf in tmp_conf_dict.values():
                    if key not in conf.keys():
                        print(
                            "Training parameter {} is not present in all sub-trainings and is not overwritten!".format(
                                key
                            )
                        )
                        raise Exception("Consistency error in training config.")
                # Check if there are differences in the parameters of the sub-trainings
                t_para = set([conf[key] for conf in tmp_conf_dict.values()])
                if len(t_para) == 1:
                    new_model_config[key] = list(t_para)[0]
                else:
                    print(
                        "Training parameter {} is not the same for all sub-trainings and is not overwritten!".format(
                            key
                        )
                    )
                    raise Exception("Consistency error in training config.")
        return new_model_config
    else:
        # If training does not consists of sub-trainings
        #   Simply return the training parameters
        return t_config["model"]


# Function to collect information on processes, classes, mapping and variables
def get_pvcm(config, t_name, recursion_list=[]):
    # Check for loops in recursive resolution
    if t_name in recursion_list:
        print(
            "Loop found in recursive training definition! {} has already been called.".format(
                t_name
            )
        )
        raise Exception("Consistency error in training config.")
    else:
        tmp_list = recursion_list + [t_name]
    t_config = config[t_name]
    new_dict = {}
    if "composite" in t_config.keys() and t_config["composite"]:
        # If training consists of sub-trainings
        #   Collect information of sub-trainings
        sub_trainings = t_config["trainings"]
        sub_conf = [get_pvcm(config, name, tmp_list) for name in sub_trainings]
        # Check if processes, variables and classes are the same for all sub-trainings
        for para in ["processes", "variables", "classes"]:
            unique_pvc = set(frozenset(training[para]) for training in sub_conf)
            if len(unique_pvc) != 1:
                print("The {} of the sub-trainings are not equal!".format(para))
                raise Exception("consistency error in training config.")
        # Check if the mapping is the same for all sub-trainings
        unique_m = [
            training["mapping"] == sub_conf[0]["mapping"] for training in sub_conf
        ]
        if len(unique_pvc) != 1 or not all(unique_m):
            print("The mappings of the sub-trainings are not equal!")
            raise Exception("consistency error in training config.")
        # Return entries of first sub-training (values are equal to all others)
        new_dict["processes"] = sub_conf[0]["processes"]
        new_dict["variables"] = sub_conf[0]["variables"]
        new_dict["classes"] = sub_conf[0]["classes"]
        new_dict["mapping"] = sub_conf[0]["mapping"]
    else:
        # If training does not consists of sub-trainings
        #   Return entries of config file for the training
        new_dict["processes"] = t_config["processes"]
        new_dict["variables"] = t_config["variables"]
        new_dict["classes"] = t_config["classes"]
        new_dict["mapping"] = t_config["mapping"]
    return new_dict


def get_misc(config, t_name, recursion_list=[]):
    # Check for loops in recursive resolution
    if t_name in recursion_list:
        print(
            "Loop found in recursive training definition! {} has already been called.".format(
                t_name
            )
        )
        raise Exception("Consistency error in training config.")
    else:
        tmp_list = recursion_list + [t_name]
    t_config = config[t_name]
    new_dict = {}
    # List of miscellaneous parameters
    misc_args = ["event_weights"]
    if "composite" in t_config.keys() and t_config["composite"]:
        for arg in misc_args:
            usub_args = set(
                [
                    get_misc(config, train, tmp_list)[arg]
                    for train in t_config["trainings"]
                ]
            )
            if len(usub_args) != 1:
                print("The {} of the sub-trainings are not equal!".format(arg))
                raise Exception("consistency error in training config.")
            new_dict[arg] = list(usub_args)[0]
    else:
        for arg in misc_args:
            new_dict[arg] = t_config[arg]
    return new_dict


def get_merged_config(config, t_name):
    new_config = {}
    new_config["parts"] = {}
    id_conf = get_id_configs(config, t_name)
    for entry in id_conf:
        identifier, path = entry
        new_config["parts"][identifier] = path
    new_config["model"] = get_model_configs(config, t_name)
    pvcm_conf = get_pvcm(config, t_name)
    for var in ["processes", "variables", "classes", "mapping"]:
        new_config[var] = pvcm_conf[var]
    misc_dict = get_misc(config, t_name)
    for key in misc_dict.keys():
        new_config[key] = misc_dict[key]
    return new_config
