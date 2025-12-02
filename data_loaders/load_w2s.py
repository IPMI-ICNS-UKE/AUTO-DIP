import os
import re
from pathlib import Path

import numpy as np
import polars as pl
from abstract_dataset import AbstractDataset
from PIL import Image
from torchvision import transforms


class W2SDataset(AbstractDataset):
    def __init__(
        self,
        root_dir: Path,
        transform: (transforms.Lambda | None) = None,
        downsize_method: str = "split",
        init: bool = True,
        select_imgs_file: (Path | None) = None,
    ):
        """initialize W2S Test Dataset

        Args:
            root_dir (Path): path to the dataset
            transform (torchvision.transforms, optional): How to transform each image.
                Defaults to None.
            downsize_method (str, optional): How to make images standard size.
                Defaults to "split".
            init (bool, optional): Whether to initialize the dataset. Defaults to True.
            select_imgs_file (Path|None, optional): path to a csv file with a selection
                of images to limit the dataset to. Defaults to None.
        """
        super().__init__(root_dir, transform, downsize_method=downsize_method)
        self.name = select_imgs_file.stem if select_imgs_file else "w2s"
        self.filename_pattern = re.compile(r"wf_channel_(?P<channel>\d).npy")
        self.foldername_pattern = re.compile(r"Image(?P<fov>\d+)")
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

        data = self.process_images(self.root_dir / "raw")

        self.df = pl.DataFrame(data)

    def process_images(self, folder_path: Path) -> list[dict]:
        """Iterate through image folders and get info for every image.

        Args:
            folder_path (Path): path to dataset

        Returns:
            list[dict]: list with info dicts for every image
        """

        data = []

        for img_folder in sorted(folder_path.iterdir()):
            if not img_folder.is_dir():
                continue

            for channel in [0, 1, 2]:
                img_name = f"wf_channel{channel}.npy"

                img_path = img_folder / img_name

                image = np.load(img_path)
                num_frames, width, height = image.shape
                num_tiles = self.calculate_num_tiles_per_img(width, height)

                fov = self.foldername_pattern.search(img_folder.name).groupdict()["fov"]

                for frame in range(num_frames):
                    for tile_index in range(num_tiles):
                        data.append(
                            {
                                "path": str(img_path.absolute()),
                                "microscope_type": "widefield",
                                "specimen": None,
                                "fov": int(fov),
                                "channel": channel,
                                "frame_num": frame,
                                "tile_index": tile_index,
                            }
                        )

        return data

    def get_params_for_dip(self, img_dict):
        """generate info about image to use as parameters for DIP

        Args:
            img_dict (dict): image dict, containing 'path', 'tile_index',
            'microscope_type', 'fov', 'channel' and optionally 'frame_num'

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
            "channel": img_dict["channel"],
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
        path = image_info["path"]
        image = np.load(path)

        return image

    def create_img_name(self, img_row: dict, **kwargs) -> str:
        """create unique name for image

        Args:
            img_row (dict): info about image

        Returns:
            str: unique name for the image
        """
        if "avg_num_frames" in self.df.columns:
            return (
                f"FOV_{img_row['fov']}_channel_{img_row['channel']}"
                + f"_avg_{img_row['avg_num_frames']}"
            )
        else:
            return (
                f"FOV_{img_row['fov']}_channel_{img_row['channel']}"
                + f"_frame_{img_row['frame_num']}"
            )

    def limit_dataset_to_sel_imgs(self):
        """Limit the dataset to the images specified in the csv file in
        self.select_imgs_file.
        """

        sel_imgs_df = pl.read_csv(self.select_imgs_file)

        df_sel = self.df.join(sel_imgs_df, on=["fov", "channel"], how="inner")

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
            image = self.load_image(data[0].to_dicts()[0])

            avg_nums = data["avg_num_frames"].unique()

            for avg_num in avg_nums:
                img_dict = data.filter(pl.col("avg_num_frames") == avg_num)[
                    0
                ].to_dicts()[0]
                img_dict["avg_num_frames"] = avg_num

                image_avg = np.mean(image[0:avg_num, :, :], axis=0)
                image_avg = self.percentile_clipping(image_avg)

                file_name = (
                    f"FOV_{data['fov'][0]}_channel_{data['channel'][0]}_"
                    + f"avg_{avg_num}.tif"
                )
                img_dict["path"] = str((self.avg_dir / file_name).absolute())

                Image.fromarray(image_avg).save(self.avg_dir / file_name)

                new_data_list.append(img_dict.copy())

        self.df = pl.DataFrame(new_data_list)

        self.df.write_csv(self.df_path)
