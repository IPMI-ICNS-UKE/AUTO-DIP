"""This module provides a class to save an image with the corresponding ground truth and 
other parameters"""
import os
import PIL
import torch
import numpy as np
import tifffile as tf
import matplotlib.pyplot as plt

from dip.utils import denoising_utils

from pathlib import Path

dtype = torch.cuda.FloatTensor


class Image():
    """Image with corresponding ground truth (if available), paths, crop regions, ...
    """
    def __init__(self,
                 data_noisy : np.ndarray,
                 path_noisy : str,
                 data_gt : (np.ndarray|None) = None,
                 path_gt : (str|None) = None,
                 crop_region : (list[int]|None) = None,
                 frame_number : (int|None) = None,
                 img_mode : (str|None) = None):
        """initialize image

        Args:
            data_noisy (np.ndarray): array with pixel values of noisy image
            path_noisy (str): path to noisy image
            data_gt (np.ndarray | None, optional): array with pixel values of ground truth image. Defaults to None.
            path_gt (str | None, optional): path to ground truth image. Defaults to None.
            crop_region (list[int] | None, optional): list with the crop region: [left, top, right, bottom].. Defaults to None.
            frame_number (int | None, optional): frame number of the image in the file. Defaults to None.
            img_mode (str | None, optional): pillow image mode of noisy image. Defaults to None.
        """

        self.data_noisy = data_noisy
        self.img_noisy_torch = denoising_utils.np_to_torch(data_noisy.astype('float')).type(dtype)
        self.data_gt = data_gt
        if self.data_gt is not None:
            self.img_gt_torch = denoising_utils.np_to_torch(data_gt.astype('float')).type(dtype)
        else:
            self.img_gt_torch = None
        self.path_noisy = path_noisy
        self.path_gt = path_gt
        self.crop_region = crop_region
        self.frame_number = frame_number

        self.base_file_name = self.get_base_file_name()

        if path_noisy.endswith(".png"):
            self.file_type = "png"
        elif path_noisy.endswith(".tif") or path_noisy.endswith(".tiff") or path_noisy.endswith(".tf2"):
            self.file_type = "tif"
        else:
            self.file_type = "unknown"

        self.save_file_noisy = None
        self.data_denoised = None
        self.save_file_denoised = None

        self.img_mode = img_mode
    
    def scale_all_img_versions_0_1(self, min : float, max : float):
        """scale noisy image and ground truth image if available with the same scale to [0,1]

        Args:
            min (float): minimum value for normalization
            max (float): maximum value for normalization
        """

        self.images_min = min
        self.images_max = max

        self.data_noisy = self.scale_to_0_1(self.data_noisy, min, max)
        self.img_noisy_torch = denoising_utils.np_to_torch(self.data_noisy.astype('float')).type(dtype)
        
        if self.data_gt is not None:
            self.data_gt = self.scale_to_0_1(self.data_gt, min, max)
            self.img_gt_torch = denoising_utils.np_to_torch(self.data_gt.astype('float')).type(dtype)
        else:
            self.img_gt_torch = None

    def scale_to_0_1(self, img : np.ndarray, min: float, max : float) -> np.ndarray:
        """scale an image with min-max normalization

        Args:
            img (np.ndarray): image to be normalized
            min (float): minimum value used for the normalization
            max (float): maximum value used for the normalization

        Returns:
            np.ndarray: normalized image
        """

        img = (img - min) / (max - min)
        return img

    def get_base_file_name(self):
        """extract base file name from path

        Returns:
            str: file name
        """
        file_name_prefix = Path(self.path_noisy).stem
        if self.crop_region:
            file_name = (f"{file_name_prefix}_{self.crop_region[0]}_{self.crop_region[1]}" +
                         f"_{self.crop_region[2]}_{self.crop_region[3]}")
        else:
            file_name = file_name_prefix
        if self.frame_number:
            file_name = f"{file_name}_frame{self.frame_number:03}"
        return file_name
