import pickle
import tifffile
from pathlib import Path

from collections.abc import Callable
from typing import Concatenate

import polars as pl
import numpy as np

import torch
import ignite.metrics
import lpips
from scipy.ndimage import sobel

from transfer_method import TransferMethod


class ClosestCalibrationImage():
    """Find the image with minimal distance to the input image in the calibration set
    """

    def __init__(self, image : (np.ndarray|torch.Tensor),
                 transfer_method : (TransferMethod),
                 best_image_params_df : (pl.DataFrame),
                 image_source_path : (Path),
                 umap_source_path : (Path|None) = None):
        """initializes class to find closest image in calibration set

        Args:
            image (np.ndarray | torch.Tensor): input image
            transfer_method (TransferMethod): parameter transfer method
            best_image_params_df (pl.DataFrame): dataframe with optimal parameters for each calibration image
            image_source_path (Path): path to the file with calibration set images
            umap_source_path (Path | None, optional): path to the files with the umap. Defaults to None.
        """

        self.best_image_params_df = best_image_params_df
        valid_tiff_idx = self.get_intra_group_img_idx(transfer_method, best_image_params_df)

        images =  tifffile.imread(image_source_path)
        images = images[valid_tiff_idx]
        self.images = torch.tensor(images[:, 0, :, :] / 255, dtype=torch.float)

        self.image = torch.tensor(image[0], dtype=torch.float)

        self.distance_metric = transfer_method.distance_metric

        if umap_source_path and self.distance_metric == "umap":
            with open(umap_source_path, "rb") as file:
                self.umap = pickle.load(file)

        self.dist_func = self.get_dist_func()

        dists = self.calc_distances()

        min_dist_img_idx = int(dists.argmin())

        self.index_in_tiff = valid_tiff_idx[min_dist_img_idx]
                
        print("closest calibration image:", self.index_in_tiff)

    def get_intra_group_img_idx(self, method : TransferMethod, best_image_params_df : pl.DataFrame) -> int:
        """get image indices of images that are in the group (or all if not restricted by group)

        Args:
            method (TransferMethod): method used to transfer the parameters
            best_image_params_df (pl.Dataframe): dataframe with calibration image information

        Returns:
            list: list of valid image indices
        """
        if method.group_based:
            best_image_params_df = best_image_params_df.filter(
                pl.col(method.group_type) == method.group_name
            )
        
        return best_image_params_df["index_in_tiff"].unique()

    def calc_distances(self) -> np.ndarray:
        """calculate distances between input image and images in the calibration set

        Returns:
            list[float]: distances to all calibration set images
        """

        dists = np.full(len(self.images), fill_value=None)
        args = [None]
        if self.distance_metric == "umap":
            umap_pos_input = self.get_umap_pos_curr_img()
            print("umap pos input img:", umap_pos_input)

        for i, calibration_image in enumerate(self.images):
            if self.distance_metric == "umap":
                umap_pos_calibration = self.get_umap_pos_calibration(i)
                args = umap_pos_input[0, 0], umap_pos_input[0, 1], umap_pos_calibration[0], umap_pos_calibration[1]

            dists[i] = self.dist_func(self.image, calibration_image, *args)

        return dists
    
    def get_umap_pos_calibration(self, img_idx : (int)) -> tuple[float, float] :
        """get the umap position of a calibration set image

        Args:
            img_idx (int): index of the calibration set image

        Returns:
            tuple(float, float): position in the umap (x, y)
        """

        img_rows = self.best_image_params_df.filter(pl.col("index_in_tiff") == img_idx)
        comp_rows = img_rows[["Component 1", "Component 2"]]

        umap_pos = comp_rows.unique().row(0)

        return umap_pos
    
    def get_umap_pos_curr_img(self):
        """get the position of the input image in the UMAP

        Returns:
            tuple(float, float): position in the UMAP
        """

        image_flat = torch.flatten(self.image)

        umap_pos = self.umap.transform([image_flat])

        return umap_pos
    

    def get_dist_func(self) -> Callable[Concatenate[torch.Tensor, torch.Tensor, ...], float]:
        """get the loss function

        Returns:
            fm_loss.LossFunction: loss funtion
        """
        match self.distance_metric:
            case 'mse':
                return mse_dist
            case 'l1':
                return l1_dist
            case 'ssim':
                return ssim_dist
            case 'psnr':
                return psnr_dist
            case 'lpips':
                return lpips_dist
            case 'umap':
                return umap_dist
            case 'mean_gradient':
                return mean_gradient_dist
            case _:
                print(self.distance_metric, "is not among the available transfer methods.")
                assert(False)


def mse_dist(im1 : torch.Tensor, im2 : torch.Tensor, *args) -> float:
    return torch.mean(((im1 - im2)**2).float()).item()

def l1_dist(im1 : torch.Tensor, im2 : torch.Tensor, *args) -> float:
    return torch.mean((torch.abs(im1 - im2)).float()).item()

def ssim_dist(im1 : torch.Tensor, im2 : torch.Tensor, *args) -> float:
    ssim_fun = ignite.metrics.SSIM(data_range=1)
    ssim_fun.update((im1[torch.newaxis, :], im2[torch.newaxis, :]))
    ssim = torch.tensor(ssim_fun.compute()).item()
    return ssim

def psnr_dist(im1 : torch.Tensor, im2 : torch.Tensor, *args) -> float:
    psnr_fun = ignite.metrics.PSNR(data_range=1)
    psnr_fun.update((im1[torch.newaxis, :], im2[torch.newaxis, :]))
    psnr = torch.tensor(psnr_fun.compute()).item()
    return psnr

def lpips_dist(im1 : torch.Tensor, im2 : torch.Tensor, *args) -> float:
    loss_fn = lpips.LPIPS(net='alex', verbose=False)
    lpips_val = loss_fn.forward(im1, im2).item()
    return lpips_val

def mean_gradient_dist(im1 : torch.Tensor, im2 : torch.Tensor, *args) -> float:
    mg1 = mean_gradient(im1)
    mg2 = mean_gradient(im2)

    return np.abs(mg1 - mg2)

def mean_gradient(image : (torch.Tensor|np.ndarray)) -> float:
    # Compute the gradients along the x and y axes
    gradient_x = sobel(image, axis=1)
    gradient_y = sobel(image, axis=2)

    # Compute the mean gradient (MG)
    _, shape_x, shape_y = image.shape
    gradient_magnitude = np.sqrt((gradient_x**2 + gradient_y**2) / 2)
    result = np.sum(gradient_magnitude) / (shape_x * shape_y)

    return result

def umap_dist(im1, im2, im1_comp1 : float, im1_comp2 : float, im2_comp1 : float, im2_comp2 : float) -> float:
    
    return np.mean((np.array([im1_comp1, im1_comp2]) - np.array([im2_comp1, im2_comp2]))**2)









