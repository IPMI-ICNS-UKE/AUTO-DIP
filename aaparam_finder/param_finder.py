
import torch
import yaml

import polars as pl
from pathlib import Path
import numpy as np

from transfer_method import TransferMethod
from aaparam_finder.closest_calibration_image import ClosestCalibrationImage


class ParamFinder():
    """finds the optimal parameters for the input image based on the image and/or microscope and specimen
    """

    def __init__(self, calibration_data_path : (Path|str), image : (np.ndarray|torch.Tensor|None) = None, 
                 microscope : (str|None) = None, specimen : (str|None) = None):
        """initialize info about image, microscope type and specimen

        Args:
            calibration_data_path (Path | str): path to folder with calibration data 
            image (np.ndarray | torch.Tensor | None, optional): noisy input image for which we choose parameters. Defaults to None.
            microscope (str | None, optional): microscope type, can be confocal, twophoton, widefield or None. Defaults to None.
            specimen (str | None, optional): specimen, can be nucleus, actin, mito or None. Defaults to None.
        """
        
        # image microscope type and object must be given if there is no image
        assert(not(image is None and (microscope is None or specimen is None)))
        if type(calibration_data_path) is str:
            calibration_data_path = Path(calibration_data_path)

        self.method = TransferMethod(microscope, specimen)

        self.closest_image_idx = None

        if self.method.image_based:

            self.best_image_params_df = pl.read_csv(calibration_data_path / "opt_params_per_image_src_with_umap.csv")

            umap_source_path = Path("transfer_params/umap_py3.12.pkl")

            closest_image = ClosestCalibrationImage(
                image = image, 
                transfer_method = self.method,
                best_image_params_df = self.best_image_params_df,
                image_source_path = calibration_data_path / "calibration_images.tiff",
                umap_source_path = umap_source_path)

            self.closest_image_idx = closest_image.index_in_tiff

        with open(calibration_data_path / "best_param_iter_comb_dict.yaml", "r") as file:
            self.best_group_params_dict = yaml.safe_load(file)

        # metric with which we measure performance on the calibration set
        self.metric = "lpips_psnr"

    def get_best_params(self) -> dict[str, (str|int)]:
        """returns optimal parameters for the image

        Returns:
            dict: optimal parameters
        """

        if self.method.image_based:
            best_params = self.get_best_image_params()
        else:
            best_params = self.get_group_params()

        # make width an int in case it's not "asc_to512"
        try:
            best_params["width"] = int(best_params["width"])
        except ValueError:
            pass
        if best_params["width"] == "asc_to512":
            best_params["width"] = [2**i for i in range(10-best_params["depth"], 10)]

        if self.closest_image_idx:
            best_params["closest_image"] = self.closest_image_idx
        else:
            best_params["closest_image"] = None

        return best_params

    def get_group_params(self) -> dict[str, (str|int)]:
        """get optimal parameters for the given microscope object group

        Returns:
            dict: optimal parameters
        """

        group_params_metric = self.best_group_params_dict[f"{self.metric}_gt"]

        group_params_group_type = group_params_metric[self.method.group_type]

        group_params_group = group_params_group_type[self.method.group_name]

        group_params_group_best = group_params_group["best"]

        return group_params_group_best

    def get_best_image_params(self) -> dict[str, (str|int)]:
        """get optimal parameters for the image based on the image and group if given

        Returns:
            dict: optimal parameters
        """

        best_image_params_df_row = self.best_image_params_df.filter(
            (pl.col("metric") == f"{self.metric}_gt") &
            (pl.col("index_in_tiff") == self.closest_image_idx)
        )

        assert(len(best_image_params_df_row) == 1)
        if self.method.group_based:
            assert(best_image_params_df_row[self.method.group_type][0] == self.method.group_name)

        best_params_dict = best_image_params_df_row[
            "depth", "width", "skip_n11", "num_iter"].to_dicts()[0]
        
        return best_params_dict
