#!/usr/bin/env python

import os
import sys
import argparse
import torch
import numpy as np

from init_params import get_parameters, add_transfer_params
from fm_image_sequence import ImageSequence
from training import Training

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.manual_seed(0)


def get_config_files() -> list[str]:
    """parse config file names from arguments

    Returns:
        list: a list of config file names
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--param_path',
                        nargs=1,
                        default=['parameters.yaml'],
                        help='Path to parameter file or folder with parameter files',
                        type=str)
    param_path = parser.parse_args().param_path[0]

    if os.path.isfile(param_path):
        configs = [param_path]
    elif os.path.isdir(param_path):
        configs = [f"{param_path}/{config_file}" for config_file
                   in sorted(os.listdir(param_path)) if config_file.endswith(".yaml")]
    else:
        print(f"file/directory {param_path} does not exist")
        sys.exit()

    return configs


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    config_list = get_config_files()

    for cfg_num, config_file in enumerate(config_list):

        print(f"processing config number {cfg_num}: {config_file}")

        parameters, image_needed_for_transfer = get_parameters(config_file)
        parameters['save_and_log']['save_imgs'] = True

        image_sequence = ImageSequence(
            path_noisy=parameters['image']['path'],
            time_series_params=parameters['time_series'],
            frame_range=parameters['image']['frame_range'],
            path_gt=parameters['image']['path_gt'],
            crop_region=parameters['image']['crop_region'],
            n_channels = parameters['net']['n_channels'],
            transfer_representative_img = parameters['transfer_params']['representative_img'])
        
        if image_needed_for_transfer:
            input_image = image_sequence.get_image_for_param_transfer()
            parameters, _ = add_transfer_params(parameters, image = input_image)

        training = Training(parameters)

        training.start_training(image_sequence)

        del image_sequence
