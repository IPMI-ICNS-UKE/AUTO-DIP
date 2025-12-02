import os
from pathlib import Path

import numpy as np
import polars as pl
from abstract_dataset import AbstractDataset
from PIL import Image
from torchvision import transforms


class FMDTestMixDataset(AbstractDataset):
    def __init__(
        self,
        root_dir: Path,
        transform: (transforms.Lambda | None) = None,
        downsize_method: str = "split",
        init: bool = True,
    ):
        """initialize FMD Test Mix Dataset

        Args:
            root_dir (Path): path to the dataset
            transform (transforms, optional):  How to transform each image.
                Defaults to None.
            downsize_method (str, optional): How to make images standard size.
                Defaults to "split".
            init (bool, optional): _descWhether to initialize the dataset.
                Defaults to True.
        """
        super().__init__(root_dir, transform, downsize_method=downsize_method)
        self.name = "fmd_test_mix"
        if init:
            self.prepare_dataset()

    def load_dataset(self):
        """Populate the DataFrame with image metadata from the dataset directory
        structure.
        """
        data = []

        # directory iteration (avg2, avg4, avg8, avg16, gt, raw)
        for folder_name in ["avg2", "avg4", "avg8", "avg16", "gt", "raw"]:
            folder_path = self.root_dir / folder_name
            if os.path.isdir(folder_path):
                data = self.process_sub_directory(folder_path, folder_name, data)

        self.df = pl.DataFrame(data)

    def process_sub_directory(self, dir_path, noise_level, data):
        """Process each subdirectory and add image information to the data list."""
        img_paths = [f for f in sorted(dir_path.iterdir()) if f.suffix == ".png"]

        image = Image.open(img_paths[0]).convert("L")
        image = np.array(image)
        width, height = image.shape

        num_tiles = self.calculate_num_tiles_per_img(width, height)

        if "avg" in noise_level or noise_level == "raw":
            image_quality = "noisy"
        else:
            image_quality = "gt"

        for img_index, img_path in enumerate(img_paths):
            image = self.load_image(image_info={"path": str(img_path)})
            mean_gradient = self.mean_gradient(image)
            img_name = img_path.stem
            # (expected format: <microscope_type>_<specimen>_<R|G|B>?_<fov>)
            parts = img_name.split("_")
            microscope_type, obj = parts[0], parts[1]
            channel = parts[2] if len(parts) == 4 else "None"
            fov = parts[-1]
            for tile_index in range(num_tiles):
                data.append(
                    {
                        "path": str(img_path),
                        "microscope_type": self.microscope_name_conv(microscope_type),
                        "specimen": obj
                        if obj != "BPAE"
                        else self.channel_specimen_conv(channel),
                        "channel": channel,
                        "image_quality": image_quality,
                        "noise_level": noise_level,
                        "img_index": img_index,
                        "fov": fov,
                        "tile_index": tile_index,
                        "mean_gradient": mean_gradient,
                    }
                )

        return data

    @staticmethod
    def microscope_name_conv(name):
        match name:
            case "Confocal":
                return "confocal"
            case "TwoPhoton":
                return "twophoton"
            case "WideField":
                return "widefield"

    @staticmethod
    def channel_specimen_conv(channel):
        match channel:
            case "R":
                return "mito"
            case "G":
                return "F-actin"
            case "B":
                return "nucleus"

    def load_image(self, image_info: dict) -> np.ndarray:
        """load image with pillow and convert to numpy array

        Args:
            image_info (dict): info about image, containing "path"

        Returns:
            np.ndarray: image as numpy array
        """
        image = Image.open(image_info["path"]).convert("L")
        image = np.array(image)
        return image

    def create_img_name(self, row: dict, gt: bool = False, **kwargs) -> str:
        """create a name for the config file

        Args:
            row (dict): row from the dataset dataframe
            gt (bool, optional): if it is the ground truth image image.
                Defaults to False.

        Returns:
            str: name of the config file
        """

        name = f"{row['noise_level']}_{Path(row['path']).stem}"

        if gt:
            name = name + "_gt"

        return name

    def get_params_for_dip(self, img_dict: dict) -> dict:
        """generate info about image to use as parameters for DIP

        Args:
            img_dict (dict): image dict, containing "path", "part_index",
            "microscope_type", "specimen", "fov", "channel"

        Returns:
            dict: image parameters for DIP
        """
        params = {}
        params["path"] = img_dict["path"]
        params["frame_range"] = None
        params["crop_region"] = None

        match_dict = {
            "image_quality": "gt",
            "noise_level": "gt",
            "tile_index": img_dict["tile_index"],
            "microscope_type": img_dict["microscope_type"],
            "specimen": img_dict["specimen"],
            "fov": img_dict["fov"],
        }

        if img_dict["channel"] is not None:
            match_dict["channel"] = img_dict["channel"]

        condition = pl.all_horizontal([self.df[k] == v for k, v in match_dict.items()])
        gt_row = self.df.filter(condition)

        assert len(gt_row) == 1

        params["path_gt"] = gt_row[0]["path"][0]

        assert "gt" in params["path_gt"]

        return params
