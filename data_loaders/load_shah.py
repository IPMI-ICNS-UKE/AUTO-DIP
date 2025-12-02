import os
import re
from pathlib import Path

import numpy as np
import polars as pl
import tifffile as tf
from abstract_dataset import AbstractDataset
from PIL import Image
from torchvision import transforms


class ShahDataset(AbstractDataset):
    def __init__(
        self,
        root_dir: Path,
        transform: (transforms.Lambda | None) = None,
        downsize_method: str = "split",
        init: bool = True,
        select_imgs_file: (Path | None) = None,
    ):
        """initialize Shah Dataset

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
        self.name = select_imgs_file.stem if select_imgs_file else "shah"
        self.filename_pattern = re.compile(
            r".*_FOV_(?P<fov>\d+)_Timestamp_(?P<timestamp>\d+)_number_(?P<number>\d+).tif"
        )
        self.select_imgs_file = select_imgs_file
        if init:
            self.prepare_dataset()

            if self.select_imgs_file:
                if self.create_avg_dir():
                    self.limit_dataset_to_sel_imgs()
                else:
                    self.df = pl.read_csv(self.df_path)

    def load_dataset(self):
        """Populate the DataFrame with image metadata from the dataset directory
        structure."""

        data_noisy = self.process_images(
            self.root_dir / "DataSet3" / "Noisy_ImageStacks", image_quality="noisy"
        )
        data_gt = self.process_images(
            self.root_dir / "DataSet3" / "Reference_Images", image_quality="gt"
        )

        data = data_noisy + data_gt

        self.df = pl.DataFrame(data)

    def get_attributes(self, filename: str) -> dict:
        """Get image attributes.

        Args:
            filename (str): name of the image file

        Returns:
            dict: attributes for the image
        """

        match = self.filename_pattern.search(filename)

        return match

    def process_images(self, folder_path: Path, image_quality: str) -> list[dict]:
        """Process info about the images.

        Args:
            folder_path (Path): path to the dataset subfolder
            image_quality (str): either 'noisy' or 'gt' (ground truth)

        Returns:
            list[dict]: list of dicts with info about each image
        """

        data = []

        for img_path in sorted(folder_path.iterdir()):
            image = Image.open(img_path)
            width, height = image.size
            num_frames = image.n_frames
            num_tiles = self.calculate_num_tiles_per_img(width, height)

            attributes = self.get_attributes(img_path.name)

            for frame in range(num_frames):
                for tile_index in range(num_tiles):
                    data.append(
                        {
                            "path": str(img_path.absolute()),
                            "microscope_type": "widefield",
                            "specimen": "Tubulin",
                            "image_quality": image_quality,
                            "fov": int(attributes["fov"]),
                            "timestamp": int(attributes["timestamp"]),
                            "img_num": int(attributes["number"]),
                            "frame_num": frame,
                            "tile_index": tile_index,
                        }
                    )

        return data

    def get_params_for_dip(self, img_dict: dict) -> dict:
        """Generate info about image to use as parameters for DIP.

        Args:
            img_dict (dict): image dict, containing 'path', 'fov', 'img_num',
            'tile_index', 'microscope_type', and optionally 'frame_num'

        Returns:
            dict: image parameters for DIP
        """
        params = {}
        params["path"] = img_dict["path"]
        params["frame_range"] = (
            [img_dict["frame_num"], img_dict["frame_num"] + 1]
            if "frame_num" in img_dict
            else None
        )
        params["crop_region"] = None

        match_dict = {
            "image_quality": "gt",
            "fov": img_dict["fov"],
            "img_num": img_dict["img_num"],
            "tile_index": img_dict["tile_index"],
            "microscope_type": img_dict["microscope_type"],
        }

        condition = pl.all_horizontal([self.df[k] == v for k, v in match_dict.items()])
        gt_row = self.df.filter(condition)

        assert len(gt_row) == 1

        params["path_gt"] = gt_row[0]["path"][0]

        return params

    def load_image(self, image_info: dict) -> np.ndarray:
        """load the image

        Args:
            image_info (dict): info about the image

        Returns:
            np.ndarry: image
        """
        path = image_info["path"][0]
        image = tf.imread(path)
        image = image.astype(np.float32)

        return image

    def create_img_name(self, img_row: dict, **kwargs) -> str:
        """Create individual name for one image.

        Args:
            img_row (dict): dict with image info

        Returns:
            str: name for the image
        """

        if "avg_num_frames" in self.df.columns:
            return (
                f"FOV_{img_row['fov']}_num_{img_row['img_num']}"
                + f"_avg_{img_row['avg_num_frames']}"
            )
        else:
            return (
                f"FOV_{img_row['fov']}_num_{img_row['img_num']}"
                + f"_frame_{img_row['frame_num']}"
            )

    def limit_dataset_to_sel_imgs(self):
        """Limit the dataset to the images specified in the csv file in
        self.select_imgs_file.
        """

        sel_imgs_df = pl.read_csv(self.select_imgs_file)

        df_sel = self.df.filter(pl.col("image_quality") == "noisy")

        df_sel.drop_in_place("image_quality")

        df_sel = df_sel.join(
            sel_imgs_df,
            on=["microscope_type", "specimen", "fov", "timestamp", "img_num"],
            how="inner",
        )

        self.df = df_sel

        if "avg_num_frames" in self.df:
            self.build_averages()

    def create_avg_dir(self):
        """Create directory to save the averaged images.

        Returns:
            bool: whether or not the directories where created (if false they existed
            already)
        """

        self.avg_dir = self.root_dir / f"{self.select_imgs_file.stem}_avgs"

        self.df_path = self.avg_dir / self.df_path.name

        if not self.avg_dir.exists():
            os.makedirs(self.avg_dir)

            return True

        return False

    @staticmethod
    def percentile_clipping(image: np.ndarray) -> np.ndarray:
        """Clip extreme pixel values in the image using percentile norm.

        Args:
            image (np.ndarray): numpy array containing the iamge

        Returns:
            np.ndarray: image with clipped extreme values
        """

        min_perc = np.percentile(image, 0.01)
        max_perc = np.percentile(image, 99.9)

        return np.clip(image, min_perc, max_perc)

    def build_averages(self):
        """Build average images to mimic different noise levels."""

        if "avg_num_frames" not in self.df.columns:
            print("avg_num_frames must be defined to build image averages.")
            return

        self.df = self.df.drop("frame_num")
        self.df = self.df.unique()

        file_grouped_df = self.df.group_by("path")

        new_data_list = []

        for _, data in file_grouped_df:
            image = self.load_image(data[0])

            avg_nums = data["avg_num_frames"].unique()

            for avg_num in avg_nums:
                img_dict = data.filter(pl.col("avg_num_frames") == avg_num)[
                    0
                ].to_dicts()[0]
                img_dict["avg_num_frames"] = avg_num

                image_avg = np.mean(image[0:avg_num, :, :, 0], axis=0)
                image_avg = self.percentile_clipping(image_avg)

                file_name = (
                    f"FOV_{data['fov'][0]}_num_{data['img_num'][0]}_avg_{avg_num}.tif"
                )
                img_dict["path"] = str((self.avg_dir / file_name).absolute())

                tf.imwrite(self.avg_dir / file_name, image_avg)

                new_data_list.append(img_dict.copy())

        self.df = pl.DataFrame(new_data_list)

        self.df.write_csv(self.df_path)
