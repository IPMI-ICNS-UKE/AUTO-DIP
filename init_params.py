"""This module parses and filters parameters."""
from pprint import pprint
import sys
import copy
import yaml
import numpy as np

sys.path.insert(0, './aaparam_finder')
from aaparam_finder.param_finder import ParamFinder

def get_parameters(config_file : str) -> tuple[dict, bool]:
    """load parameter file, fill missing parameters with default ones

    Args:
        config_file (str): path to config file

    Returns:
        dict: parameters for AUTO-DIP
        bool: image_needed, does the image need to be loaded to find transfer params?
    """

    custom_params = load_custom_parameters(config_file)

    default_params = load_default_parameters()

    if "path" not in custom_params['image']:
        print("There is no image path given in the config file.")
        sys.exit()

    image_needed = False
    if "transfer_params" in custom_params:
        custom_params, image_needed = add_transfer_params(custom_params)
        print(custom_params)

    
    combined_params = fill_with_default_parameters(custom_params, default_params)

    filtered_params = filter_unknown_parameters(combined_params, default_params)
    final_params = filter_unused_parameters(filtered_params)

    verify_params(final_params)


    if final_params["save_and_log"]["verbosity"] > 0:
        pprint(final_params)

    return final_params, image_needed

def load_custom_parameters(config_file : str) -> dict:
    """load parameters from custom config file

    Args:
        config_file (str): path to config file

    Returns:
        dict: dict with custom parameters
    """
    # load custom parameters
    try:
        with open(config_file, 'r') as read_file:  
            custom_params = yaml.safe_load(read_file)
        return custom_params
    except OSError:
        print(f"config file {config_file} does not exist")
        sys.exit()
    except yaml.YAMLError as exc:
        print(exc)
        sys.exit()

def load_default_parameters() -> dict:
    """load default parameters

    Returns:
        dict: dict with default parameters
    """
    # load default parameters
    default_params_file = "./default_parameters.yaml"
    try:
        with open(default_params_file, "r") as read_file:
            default_params = yaml.safe_load(read_file)
            return default_params
    except OSError:
        print(f"default parameters file '{default_params_file}' does not exist")
        sys.exit()
    except yaml.YAMLError as exc:
        print(exc)

def add_transfer_params(custom_params: dict, image: (np.ndarray|None) = None) -> tuple[dict, bool]:
    """add transfer params to custum params

    Args:
        custom_params (dict): custom parameters
        image (np.ndarray | None, optional): image for parameter transfer. Defaults to None.

    Returns:
        dict: dict with combined parameters
        bool: image needed is needed for transfer but not given 
    """

    transfer_param_conf = custom_params["transfer_params"]

    microscope = None
    specimen = None
    if "microscope" in transfer_param_conf:
        microscope = transfer_param_conf["microscope"]
    if "specimen" in transfer_param_conf:
        specimen  = transfer_param_conf["specimen"]

    try:
        param_finder = ParamFinder(microscope=microscope, specimen=specimen, 
                                   image=image, calibration_data_path="transfer_params")
    except AssertionError:
        # image is needed to get parameters
        print("image is needed to get parameters")
        return custom_params, True

    params = param_finder.get_best_params()

    if "net" not in custom_params:
        custom_params["net"] = {}

    custom_params["net"] = add_transfer_net_param_to_custom_param(
        custom_params["net"], params, "num_iter", "num_iter")
    custom_params["net"] = add_transfer_net_param_to_custom_param(
        custom_params["net"], params, "skip_n33d", "width")
    custom_params["net"] = add_transfer_net_param_to_custom_param(
        custom_params["net"], params, "skip_n33u", "width")
    custom_params["net"] = add_transfer_net_param_to_custom_param(
        custom_params["net"], params, "skip_n11", "skip_n11")
    custom_params["net"] = add_transfer_net_param_to_custom_param(
        custom_params["net"], params, "num_scales", "depth")

    custom_params["transfer_params"]["closest_image"] = params["closest_image"] 
    if "representative_img" in transfer_param_conf:
        custom_params["transfer_params"]["representative_img"] = transfer_param_conf["representative_img"] 

    return custom_params, False

def add_transfer_net_param_to_custom_param(custom_net_params : dict, transfer_params : dict, 
                                           param_name : str, param_name_transfer : str) -> dict:
    """add transfer parameter to custom parameters without overwriting custom parameters

    Args:
        custom_net_params (dict): custom net parameters 
        transfer_params (dict): transfer parameters
        param_name (str): parameter name in custom parameters
        param_name_transfer (str): parameter name in transfer parameters

    Returns:
        dict: combined parameters
    """

    if param_name not in custom_net_params:
        custom_net_params[param_name] = transfer_params[param_name_transfer]
    else:
        print(f"WARNING: Transfer parameter '{param_name_transfer}' ({transfer_params[param_name_transfer]}) " +
              f"is ignored, because you manually defined '{param_name}'={custom_net_params[param_name]}")
    return custom_net_params

def fill_with_default_parameters(custom_params : dict, default_params : dict) -> dict:
    """fill custom parameters with defaults where they are not specified

    Args:
        custom_params (dict): custom parameters
        default_params (dict): default parameters

    Returns:
        dict: combined paramters
    """
    for param_group in default_params:
        if param_group not in custom_params:
            custom_params[param_group] = default_params[param_group]
        for param in default_params[param_group]:
            if param not in custom_params[param_group]:
                custom_params[param_group][param] = default_params[param_group][param]

    return custom_params

def filter_unknown_parameters(params : dict, default_params : dict) -> dict:
    """filter parameters that are not known

    Args:
        params (dict): parameters for the run
        default_params (dict): default parameters

    Returns:
        dict: filtered parameters
    """
    filtered_params = copy.deepcopy(params)

    # check for unknown params
    for param_group in params:
        if param_group not in default_params:
            print(f"parameter group {param_group} is not recognized and thus ignored.")
            filtered_params.pop(param_group)
            continue
        for param in params[param_group]:
            if param not in default_params[param_group]:
                if param_group == "image" and param == "path":
                    continue
                print(f"parameter {param} in {param_group} is not recognized and thus ignored.")
                filtered_params[param_group].pop(param)

    return filtered_params

def filter_unused_parameters(params : dict) -> dict:
    """filter parameters that are not needed dependent on other parameter values

    Args:
        params (dict): parameters for the run

    Returns:
        dict: filtered parameters
    """
    filtered_params = copy.deepcopy(params)

    # throw out params that are not needed
    if not params["time_series"]["is_series"]:
        filtered_params["time_series"].pop("learning_rate")
        filtered_params["time_series"].pop("num_iter")
        filtered_params["time_series"].pop("back_and_forth")

    if params["superresolution"]["superres_factor"] == 1:
        filtered_params["superresolution"].pop("downsample_kernel")

    if params["loss"]["loss_type_main"] != "psf" and params["loss"]["loss_type2"] != "psf":
        filtered_params["psf"] = None

    if not params["loss"]["loss_type2"]:
        filtered_params["loss"].pop("loss2_fact")
        filtered_params["loss"].pop("loss2_incr_fact")

    if not params["loss"]["regularizer"]:
        filtered_params["loss"].pop("regularizer_fact")
        filtered_params["loss"].pop("regularizer_incr_fact")

    return filtered_params

def verify_params(final_params : dict):
    """verify parameters 

    Args:
        final_params (dict): parameter dict

    Raises:
        ValueError: crop region does not have a length of 4
        ValueError: left is higher than right in crop region
        ValueError: top is higher than bottom in crop region
    """

    if "crop_region" in final_params["image"] and final_params["image"]["crop_region"] is not None:

        if len(final_params["image"]["crop_region"]) != 4:
            raise ValueError("crop_region must be a list with length 4.")
        
        if not final_params["image"]["crop_region"][0] < final_params["image"]["crop_region"][2]:
            raise ValueError("crop_region must have the form: [left, top, right, bottom]. " +
                             "Left cannot be higher than right.")
        
        if not final_params["image"]["crop_region"][1] < final_params["image"]["crop_region"][3]:
            raise ValueError("crop_region must have the form: [left, top, right, bottom]. " +
                             "Top cannot be higher than bottom.")
