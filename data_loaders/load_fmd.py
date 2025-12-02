import re
from pathlib import Path

import numpy as np
import polars as pl
from abstract_dataset import AbstractDataset
from PIL import Image
from torchvision import transforms


class FMDDataset(AbstractDataset):
    def __init__(
        self,
        root_dir: Path,
        transform: (transforms.Lambda | None) = None,
        downsize_method: (str | None) = None,
        init: bool = True,
        select_imgs_file: (Path | None) = None,
    ):
        """initialize FMD Dataset

        Args:
            root_dir (Path): path to the dataset
            transform (torchvision.transforms, optional): How to transform each image.
                Defaults to None.
            downsize_method (str, optional): How to make images standard size.
                Defaults to None.
            init (bool, optional): Whether to initialize the dataset. Defaults to True.
            select_imgs_file (Path|None, optional): path to a csv file with a selection
                of images to limit the dataset to. Defaults to None.
        """
        super().__init__(root_dir, transform, downsize_method=downsize_method)
        self.name = select_imgs_file.stem if select_imgs_file else "fmd"
        self.foldername_pattern = re.compile(
            r"(?P<microscope>[A-Za-z]+)_(?P<specimen>[A-Za-z]+)(_(?P<channel>[A-Z]+))?"
        )
        self.filename_pattern = re.compile(r".*(?P<capture_idx>\d\d).png")

        self.select_imgs_file = select_imgs_file
        if init:
            self.prepare_dataset()

            if self.select_imgs_file:
                self.limit_dataset_to_sel_imgs()

    def load_dataset(self):
        """Populate the DataFrame with image metadata from the dataset directory
        structure."""
        data = []

        for folder_path in sorted(self.root_dir.iterdir()):
            if folder_path.name not in ["test_mix"] and folder_path.is_dir():
                attributes = self.get_attributes(folder_path.name)
                microscope_type = attributes["microscope"]
                specimen = attributes["specimen"]

                # Subdirectory iteration (avg2, avg4, avg8, avg16, gt, raw)
                for sub_folder in ["avg2", "avg4", "avg8", "avg16", "gt", "raw"]:
                    sub_folder_path = folder_path / sub_folder
                    if sub_folder_path.is_dir():
                        data = self.process_sub_directory(
                            sub_folder_path, microscope_type, specimen, sub_folder, data
                        )

        self.df = pl.DataFrame(data, infer_schema_length=None)

    def process_sub_directory(
        self,
        dir_path: Path,
        microscope_type: str,
        specimen: str,
        noise_level: str,
        data: dict,
    ) -> dict:
        """Process each subdirectory and add image information to the data list.

        Args:
            dir_path (Path): path to the directory
            microscope_type (str): microscope type
            specimen (str): specimen
            noise_level (str): noise level
            data (dict): previously processed data

        Returns:
            data: dict with image info
        """
        num_parts = self.calculate_num_tiles_per_img(512, 512)

        if "avg" in noise_level or noise_level == "raw":
            image_quality = "noisy"
        else:
            image_quality = "gt"

        for fov_path in sorted(dir_path.iterdir()):
            if not fov_path.is_dir():
                continue

            fov = int(fov_path.name)

            for img_path in fov_path.iterdir():
                if img_path.suffix != ".png":
                    continue

                capture_idx = int(
                    self.filename_pattern.search(img_path.name)["capture_idx"]
                )

                for part_idx in range(num_parts):
                    data.append(
                        {
                            "path": str(img_path.absolute()),
                            "microscope_type": microscope_type,
                            "specimen": specimen,
                            "noise_level": noise_level,
                            "image_quality": image_quality,
                            "fov": fov,
                            "capture_idx": capture_idx,
                            "part_index": part_idx,
                        }
                    )

        return data

    @staticmethod
    def microscope_name_conv(name: str) -> str:
        """map the microscope name in the folder name to a standard one

        Args:
            name (str): microscope name in folder name

        Returns:
            str: standardized microscope name
        """
        match name:
            case "Confocal":
                return "confocal"
            case "TwoPhoton":
                return "twophoton"
            case "WideField":
                return "widefield"

    @staticmethod
    def channel_specimen_conv(channel: str) -> str:
        """map the channel letter to the specimen

        Args:
            channel (str): channel letter

        Returns:
            str: specimen
        """
        match channel:
            case "R":
                return "mito"
            case "G":
                return "F-actin"
            case "B":
                return "nucleus"

    def get_attributes(self, foldername: str) -> dict:
        """get attibutes based on the folder name

        Args:
            foldername (str): name of the folder

        Returns:
            dict: dict with attributes, containing "microscope" and "specimen"
        """

        match = self.foldername_pattern.search(foldername)

        attributes = match.groupdict()

        attributes["microscope"] = self.microscope_name_conv(attributes["microscope"])

        if attributes["channel"]:
            attributes["specimen"] = self.channel_specimen_conv(attributes["channel"])

        return attributes

    def load_image(self, image_info: dict) -> np.ndarray:
        """load image to numpy array

        Args:
            image_info (dict): dict with image info, containing the "path"

        Returns:
            np.ndarray: image as ndarray
        """
        image = Image.open(image_info["path"]).convert("L")
        image = np.array(image)
        return image

    def get_params_for_dip(self, img_dict: dict) -> dict:
        """generate info about image to use as parameters for DIP

        Args:
            img_dict (dict): image dict, containing "path", "part_index",
            "microscope_type", "specimen", "fov"

        Returns:
            dict: image parameters for DIP
        """
        params = {}
        params["path"] = img_dict["path"]
        params["frame_range"] = None
        params["crop_region"] = None

        match_dict = {
            "noise_level": "gt",
            "image_quality": "gt",
            "fov": img_dict["fov"],
            "part_index": img_dict["part_index"],
            "microscope_type": img_dict["microscope_type"],
            "specimen": img_dict["specimen"],
        }

        condition = pl.all_horizontal([self.df[k] == v for k, v in match_dict.items()])
        gt_row = self.df.filter(condition)

        assert len(gt_row) == 1

        params["path_gt"] = gt_row[0]["path"][0]

        assert "gt" in params["path_gt"]

        return params

    def create_img_name(self, row: dict, **kwargs) -> str:
        """create a name for the config file

        Args:
            row (dict): row from the dataset dataframe

        Returns:
            str: name of the config file
        """

        name = (
            f"{row['microscope_type']}_{row['specimen']}_{row['noise_level']}_"
            + f"{row['fov']}_{row['capture_idx']}_{row['part_index']}"
        )

        return name

    def limit_dataset_to_sel_imgs(self):
        """limit the dataset to to the images specified in the csv file
        in self.select_imgs_file.
        The corresponding ground truth images to the selected noisy images are
        automatically added.
        """

        sel_imgs_df = pl.read_csv(self.select_imgs_file)

        df_sel_noisy = self.df.join(
            sel_imgs_df,
            on=["microscope_type", "specimen", "noise_level", "fov", "capture_idx"],
            how="inner",
        )

        gt_df = self.df.filter(pl.col("image_quality") == "gt")

        df_sel_gt = gt_df.join(
            sel_imgs_df.drop(("noise_level", "capture_idx")),
            on=["microscope_type", "specimen", "fov"],
            how="inner",
        ).unique()

        self.df = pl.concat((df_sel_noisy, df_sel_gt))
