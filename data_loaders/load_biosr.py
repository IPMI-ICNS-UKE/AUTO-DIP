import os
import re
from pathlib import Path

import numpy as np
import polars as pl
from abstract_dataset import AbstractDataset
from read_mrc import read_mrc, write_mrc
from torchvision import transforms


class BioSRDataset(AbstractDataset):
    def __init__(
        self,
        root_dir: Path,
        transform: (transforms.Lambda | None) = None,
        downsize_method: (str | None) = None,
        init: bool = True,
        tile_overlap: int = 128,
        select_imgs_file: (Path | None) = None,
    ):
        """initialize BioSR Test Dataset

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
        super().__init__(root_dir, transform, downsize_method, tile_overlap)
        self.name = select_imgs_file.stem if select_imgs_file else "biosr"
        self.filename_pattern = re.compile(
            r"RawSIMData_((?P<gt>gt)|(level_(?P<level>\d+))).mrc"
        )
        self.filename_pattern_ER = re.compile(r".*_level_(?P<level>\d+).mrc")
        self.folder_name_pattern = re.compile(r"Cell_(?P<cell_num>\d+)")

        self.select_imgs_file = select_imgs_file

        self.limited = False

        if init:
            self.prepare_dataset()

            if self.select_imgs_file:
                if self.create_percentile_clip_dir():
                    self.limit_dataset_to_sel_imgs()
                else:
                    self.df = pl.read_csv(self.df_path)
                self.limited = True

    def load_dataset(self):
        """Populate the DataFrame with image metadata from the dataset directory
        structure."""
        data = []

        img_index = 0

        specimens = ["CCPs", "F-actin", "ER", "F-actin_Nonlinear", "Microtubules"]
        for specimen in specimens:
            specimen_dir = self.root_dir / specimen
            for cell_path in sorted(specimen_dir.iterdir()):
                if cell_path.is_dir():
                    if specimen == "ER":
                        data, img_index = self.process_cell_directory_ER(
                            cell_path, data, img_index, specimen
                        )
                    else:
                        data, img_index = self.process_cell_directory(
                            cell_path, data, img_index, specimen
                        )

        self.df = pl.DataFrame(data, infer_schema_length=100000)

    def process_cell_directory(self, dir_path, data, img_index, specimen):
        """Process each subdirectory and add image information to the data list."""

        cell_num = int(self.folder_name_pattern.search(dir_path.name)["cell_num"])

        for i, file in enumerate(sorted(dir_path.iterdir())):
            if file.suffix == ".mrc" and file.name not in [
                "SIM_gt.mrc",
                "SIM_gt_a.mrc",
                "SIM_gt_b.mrc",
            ]:
                file_info = self.filename_pattern.search(file.name)
                if file_info["gt"] == "gt":  # in file_info.groupdict():
                    image_quality = "gt"
                    noise_level = None
                else:
                    image_quality = "noisy"
                    noise_level = int(file_info["level"])

                if i == 0:
                    header, image = read_mrc(file)
                    width, height, num_frames = image.shape
                    num_tiles = self.calculate_num_tiles_per_img(width, height)

                for frame in range(num_frames):
                    for tile_index in range(num_tiles):
                        data.append(
                            {
                                "path": str(file.absolute()),
                                "microscope_type": "widefield",  # SIM widefield
                                "image_quality": image_quality,
                                "noise_level": noise_level,
                                "img_index": img_index,
                                "cell_num": cell_num,
                                "tile_index": tile_index,
                                "frame_num": frame,
                                "specimen": specimen,
                                "width": width,
                                "height": height,
                            }
                        )
                        img_index += 1

        return data, img_index

    def process_cell_directory_ER(self, dir_path, data, img_index, specimen="ER"):
        """Process each subdirectory and add image information to the data list."""

        map_path_to_noise_level = {"RawGTSIMData": "gt", "RawSIMData": "noisy"}
        cell_num = int(self.folder_name_pattern.search(dir_path.name)["cell_num"])

        for folder in ["RawGTSIMData", "RawSIMData"]:
            quality_folder = dir_path / folder
            for i, file in enumerate(sorted(quality_folder.iterdir())):
                if file.suffix == ".mrc":
                    image_quality = map_path_to_noise_level[folder]
                    noise_level = int(
                        self.filename_pattern_ER.search(file.name)["level"]
                    )
                    # noise_level =  int((re.findall(r'\d+', filename))[0])

                    if i == 0:
                        header, image = read_mrc(file)
                        width, height, num_frames = image.shape
                        num_tiles = self.calculate_num_tiles_per_img(width, height)

                    for frame in range(num_frames):
                        for tile_index in range(num_tiles):
                            data.append(
                                {
                                    "path": str(file.absolute()),
                                    "microscope_type": "widefield",  # SIM widefield
                                    "image_quality": image_quality,
                                    "noise_level": noise_level,
                                    "img_index": img_index,
                                    "cell_num": cell_num,
                                    "tile_index": tile_index,
                                    "frame_num": frame,
                                    "specimen": specimen,
                                    "width": width,
                                    "height": height,
                                }
                            )
                            img_index += 1

        return data, img_index

    def load_image(self, image_info, with_header=False):
        """Load the image from file.

        Args:
            image_info (dict): info about image containing 'path' and 'frame_num'
            with_header (bool, optional): wheter to also return the header.
            Defaults to False.

        Returns:
            np.ndarray: image
        """
        header, image = read_mrc(image_info["path"])

        image = image[:, :, int(image_info["frame_num"])]

        image = np.array(image)

        if with_header:
            return image, header
        else:
            return image

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
            "cell_num": img_dict["cell_num"],
            "frame_num": img_dict["frame_num"],
        }

        condition = pl.all_horizontal([self.df[k] == v for k, v in match_dict.items()])
        gt_row = self.df.filter(condition)

        assert len(gt_row) == 1

        params["path_gt"] = gt_row[0]["path"][0]

        return params

    def create_img_name(self, row: dict, gt: bool = False, **kwargs) -> str:
        """create a name for the config file

        Args:
            row (dict): row from the dataset dataframe
            gt (bool, optional): if it is the ground truth image image.
                Defaults to False.

        Returns:
            str: name of the config file
        """

        if self.limited:
            name = f"{Path(row['path']).stem}"
        else:
            name = (
                f"{row['specimen']}_Cell_{row['cell_num']:03}_{Path(row['path']).stem}"
            )

        if gt:
            name = name + "_gt"

        return name

    def limit_dataset_to_sel_imgs(self):
        """Limit the dataset to the images specified in the csv file in
        self.select_imgs_file.
        """

        sel_imgs_df = pl.read_csv(self.select_imgs_file)

        df_sel = self.df.filter(pl.col("image_quality") == "noisy")

        df_sel.drop_in_place("image_quality")

        df_sel = df_sel.join(
            sel_imgs_df,
            on=["microscope_type", "specimen", "noise_level", "cell_num", "frame_num"],
            how="inner",
        )

        self.df = df_sel

        self.save_percentile_normalized()

    def create_percentile_clip_dir(self):
        """Create directory to save the clipped images.

        Returns:
            bool: whether or not the directories where created (if false they existed
            already)
        """

        self.clipped_dir = self.root_dir / f"{self.select_imgs_file.stem}_clip"

        self.df_path = self.clipped_dir / self.df_path.name

        if not self.clipped_dir.exists():
            os.makedirs(self.clipped_dir)

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

    def save_percentile_normalized(self):
        """Clip extreme pixel values of each selected image and save it separately"""
        for i in range(len(self.df)):
            image_dict = self.df[i].to_dicts()[0]

            image, header = self.load_image(image_dict, with_header=True)
            image_clipped = self.percentile_clipping(image)

            file_name = self.create_img_name(image_dict)
            self.df[i, "path"] = str((self.clipped_dir / file_name).absolute())

            # changes because we only select one frame
            header[0][2] = 1
            image_clipped = image_clipped[:, :, np.newaxis]
            write_mrc(self.clipped_dir / file_name, image_clipped, header=header)

        self.df.write_csv(self.df_path)
