import os
import re
from pathlib import Path

import numpy as np
import polars as pl
import tifffile as tf
from abstract_dataset import AbstractDataset
from torchvision import transforms


class HagenTestDataset(AbstractDataset):
    def __init__(
        self,
        root_dir: Path,
        transform: (transforms.Lambda | None) = None,
        downsize_method: (str | None) = None,
        init: bool = True,
        tile_overlap: int = 128,
    ):
        """initialize Hagen Test Dataset

        Args:
            root_dir (Path): path to the dataset
            transform (torchvision.transforms, optional): How to transform each image.
                Defaults to None.
            downsize_method (str, optional): How to make images standard size.
                Defaults to "split".
            init (bool, optional): Whether to initialize the dataset. Defaults to True.
            tile_overlap (int) : Number of pixel that overlap when splitting the image
                into tiles.
        """
        super().__init__(root_dir, transform, downsize_method, tile_overlap)
        self.name = "hagen_test"
        self.filename_pattern1 = re.compile(
            r"(?P<specimen>.+)-(?P<magnification>.+)x-noise(?P<noise_level>.+)-(?P<snr>.+)snr.tif"
        )
        self.filename_pattern2 = re.compile(
            r"(?P<specimen>.+)-(?P<microscope_type>.+)-(?P<snr>.+)snr.tif"
        )
        self.filename_pattern3 = re.compile(r"(?P<specimen>.+)-(?P<snr>.+)snr.tif")
        if init:
            self.prepare_dataset()

    def load_dataset(self):
        """Load the dataset and populate the DataFrame with image metadata."""
        data = []

        img_index = 0

        for filename in os.listdir(self.root_dir):
            if filename.endswith(".tif"):
                file_path = self.root_dir / filename
                attributes = self.get_attributes(filename)

                with tf.TiffFile(file_path) as tif:
                    num_images = len(tif.pages)
                    # last 10% of images in each file are test images
                    test_range_begin = int(0.9 * num_images)
                    for frame_num in range(test_range_begin, num_images):
                        page = tif.pages[frame_num]
                        width, height = page.shape[:2]
                        num_tiles = self.calculate_num_tiles_per_img(width, height)

                        microscope_type = attributes.get("microscope_type", "widefield")
                        specimen = attributes["specimen"]
                        magnification = attributes.get("magnification", None)

                        if magnification is None:
                            if microscope_type == "confocal":
                                magnification = 63
                            elif specimen == "nucleus" or specimen == "membrane":
                                magnification = 100

                        for tile_index in range(num_tiles):
                            data.append(
                                {
                                    "path": file_path,
                                    "microscope_type": microscope_type,
                                    "specimen": specimen,
                                    "image_quality": attributes["image_quality"],
                                    "noise_level": attributes.get("noise_level", None),
                                    "magnification": magnification,
                                    "width": width,
                                    "height": height,
                                    "frame_num": frame_num,
                                    "tile_index": tile_index,
                                    "img_index": img_index,
                                }
                            )
                            img_index += 1
        self.df = pl.DataFrame(data)

    def get_attributes(self, filename: str) -> dict:
        """get attributes for the image file

        Args:
            filename (str): file name

        Returns:
            dict: dict with the attributes
        """

        match = self.filename_pattern1.search(filename)
        if not match:
            match = self.filename_pattern2.search(filename)
        if not match:
            match = self.filename_pattern3.search(filename)

        if match:
            attributes = match.groupdict()

            assert "snr" in attributes
            attributes["image_quality"] = (
                "gt" if attributes["snr"] == "high" else "noisy"
            )
            return attributes
        else:
            exit("no match found")

    # TODO: if we need this: make it for image tile / frame
    def load_image(self, image_info: dict) -> np.ndarray:
        """load the image

        Args:
            image_info (dict): info about the image

        Returns:
            np.ndarry: image
        """
        path = image_info["path"]
        img_index_in_file = int(image_info["img_index_in_file"])

        with tf.TiffFile(path) as tif:
            image = tif.pages[img_index_in_file].asarray()

        return image

    def create_img_name(self, row, gt=False, repr_img=""):
        """create a name for the config file

        Args:
            row (dict): row from the dataset dataframe
            gt (bool, optional): if it is the ground truth image image. Defaults to False.
            repr_img (str, optional): type of the representative image ("downsize" or
            "centercrop" or ""). Defaults to "".

        Returns:
            str: name of the config file
        """

        name = Path(row["path"]).stem

        if repr_img:
            name += f"_{repr_img}"
        name += f"_part{row['tile_index']}"

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
        params["frame_range"] = [img_dict["frame_num"], img_dict["frame_num"] + 1]
        params["crop_region"] = self.get_crop_region(
            img_dict["width"], img_dict["height"], tile_num=img_dict["tile_index"]
        )

        match_dict = {
            "image_quality": "gt",
            "microscope_type": img_dict["microscope_type"],
            "specimen": img_dict["specimen"],
            "tile_index": img_dict["tile_index"],
            "frame_num": img_dict["frame_num"],
        }

        if img_dict["magnification"]:
            match_dict["magnification"] = img_dict["magnification"]
        if img_dict["noise_level"]:
            match_dict["noise_level"] = img_dict["noise_level"]

        condition = pl.all_horizontal([self.df[k] == v for k, v in match_dict.items()])
        gt_row = self.df.filter(condition)

        assert len(gt_row) == 1

        params["path_gt"] = gt_row[0]["path"][0]

        return params
