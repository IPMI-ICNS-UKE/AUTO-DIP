import math
import os
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import polars as pl
import torch
from PIL import Image
from scipy.ndimage import sobel
from torch.utils.data import Dataset
from torchvision import transforms


class AbstractDataset(Dataset, ABC):
    def __init__(
        self,
        root_dir: Path,
        transform: (transforms.Lambda | None) = None,
        downsize_method: str = "split",
        tile_overlap: int = 0,
    ):
        """initialize abstract dataset class

        Args:
            root_dir (Path): path to dataset
            transform (torchvision.transforms, optional): How to transform each image.
                Defaults to None.
            downsize_method (str, optional): How to make images standard size.
                Defaults to "split".
            tile_overlap (int, optional): Number of pixel that overlap when splitting
                the image into tiles. Defaults to 0.
        """
        self.root_dir = root_dir
        if transform == "default":
            self.transform = transforms.Compose(
                [
                    transforms.Lambda(lambda x: torch.from_numpy(x).float()),
                    transforms.Lambda(lambda x: torch.flatten(x)),
                ]
            )

        else:
            self.transform = transform

        self.downsize_method = downsize_method

        self.tile_overlap = tile_overlap

        self.standard_img_size = 512

        self.df_path = None

    def set_df_path(self):
        """set the path where to save the dataframe with dataset info"""
        csv_name = (
            f"df_{self.name}_{self.downsize_method}"
            + f"_imgsize{self.standard_img_size}_overlap{self.tile_overlap}.csv"
        )
        self.df_path = self.root_dir / csv_name

    def prepare_dataset(self):
        """
        Load dataset info from file if it exists, else parse dataset.
        """
        self.set_df_path()
        if not self.load_df_from_file():
            self.load_dataset()
            self.save_df_to_file()

    @abstractmethod
    def load_dataset(self):
        """
        Load dataset specific files and labels.
        This method needs to be implemented by each subclass.
        """
        pass

    def __len__(self):
        """get the length of the dataset

        Returns:
            int: length of the dataset
        """
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, dict]:
        """Get image from dataset

        Args:
            idx (int): index in dataset

        Raises:
            IndexError: if index is out of dataset range

        Returns:
            tuple[np.ndarray, dict]: image, info about image
        """
        if idx >= len(self.df):
            raise IndexError("Index out of bounds for the dataset.")

        image_info = self.df[idx]

        image = self.get_image(image_info)

        labels = image_info.to_dict()

        labels["dataset"] = self.name

        return image, labels

    def get_image(self, image_info):
        """get image in standard size based on image info

        Args:
            image_info (dict): line of the dataframe with the images info

        Returns:
            np.ndarray: the image as numpy array
        """
        image = self.load_image(image_info)

        # If the image needs to be split into parts, handle that
        if image.shape != (
            (self.standard_img_size, self.standard_img_size)
            and self.downsize_method is not None
        ):
            image = self.get_image_with_standard_size(image, image_info["tile_index"])

        if self.transform:
            image = self.transform(image)

        return image

    @abstractmethod
    def load_image(self, image_info):
        """load the image based the given info

        Args:
            image_info (dict): dataframe row with info about the image
        """
        pass

    def get_image_with_standard_size(
        self, image: np.ndarray, img_inner_idx: int = None
    ) -> np.ndarray | list[np.ndarray]:
        """get image with standard size

        Args:
            image (np.ndarray): image
            img_inner_idx (int, optional): Tile index in the image to return.
                Defaults to None.

        Returns:
            np.ndarray | list[np.ndarray]: reized image or image tile corresponding to
                index if given else list with image tiles
        """
        if self.downsize_method == "split":
            image = self.split_image(image, img_inner_idx)
        else:
            image = self.resize_image(image)
        return image

    def calculate_num_tiles_per_img(self, width: int, height: int):
        """Calculate the number of parts an image should be split into, so that each
        tile has the standard size.

        Args:
            width (int): width of the image in pixels
            height (int): height of the image in pixels

        Returns:
            int: number of tile when splitting the image into tile of standard size
        """
        if self.downsize_method == "split":
            step = self.standard_img_size - self.tile_overlap
            tiles_across = max(math.ceil((width - self.tile_overlap) / step), 1)
            tiles_down = max(math.ceil((height - self.tile_overlap) / step), 1)
            return tiles_across * tiles_down
        else:
            return 1

    def get_crop_region(self, img_width: int, img_height: int, tile_num: int):
        """Get the crop region for the image tile with the given tile number

        Args:
            img_width (int): width of the image in pixels
            img_height (int): height of the image in pixels
            part_num (int): nuber of the image tile

        Raises:
            ValueError: if the tile number is out of range

        Returns:
            list[int]: list with the crop region of the form [left, top, right, bottom]
        """
        step = self.standard_img_size - self.tile_overlap
        cols = max(math.ceil((img_width - self.tile_overlap) / step), 1)
        rows = max(math.ceil((img_height - self.tile_overlap) / step), 1)
        total_regions = cols * rows

        if tile_num >= total_regions or tile_num < 0:
            raise ValueError(
                "Region number out of range."
                + f"Must be between 0 and {total_regions - 1}."
            )

        row = tile_num // cols
        col = tile_num % cols

        x_start = col * step
        y_start = row * step
        x_end = min(x_start + self.standard_img_size, img_width)
        y_end = min(y_start + self.standard_img_size, img_height)

        return [x_start, y_start, x_end, y_end]  # [left, top, right, bottom]

    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize the image to standard_img_size x standard_img_size pixels,
        cropping if necessary.

        Args:
            image (np.ndarray): input image

        Returns:
            np.ndarray: resized image
        """
        # Crop to make the image square
        min_dim = min(image.shape[:2])
        start_x = (image.shape[1] - min_dim) // 2
        start_y = (image.shape[0] - min_dim) // 2
        image = image[start_y : start_y + min_dim, start_x : start_x + min_dim]

        # Resize to self.standard_img_sizexself.standard_img_size
        image = Image.fromarray(image)
        image = image.resize((self.standard_img_size, self.standard_img_size))
        image_np = np.array(image)
        return image_np

    def split_image(
        self, image: np.ndarray, index: int = None
    ) -> np.ndarray | list[np.ndarray]:
        """Split the image into multiple standard_img_size x standard_img_size images,
        using mirroring if dimensions are less than standard_img_size.

        Args:
            image (np.ndarray): input image
            index (int|None): index of the image tile to output. Defaults to 'None'.

        Returns:
            np.ndarray | list[np.ndarray]: image tile corresponding to index if given
                else list with image tiles
        """
        original_height, original_width = image.shape

        # Apply mirroring if necessary
        if original_height < self.standard_img_size:
            pad_top = (self.standard_img_size - original_height) // 2
            pad_bottom = self.standard_img_size - original_height - pad_top
            image = np.pad(image, ((pad_top, pad_bottom), (0, 0)), mode="reflect")
        if original_width < self.standard_img_size:
            pad_left = (self.standard_img_size - original_width) // 2
            pad_right = self.standard_img_size - original_width - pad_left
            image = np.pad(image, ((0, 0), (pad_left, pad_right)), mode="reflect")

        # Crop to make dimensions divisible by self.standard_img_size
        new_height = (image.shape[0] // self.standard_img_size) * self.standard_img_size
        new_width = (image.shape[1] // self.standard_img_size) * self.standard_img_size
        start_x = (image.shape[1] - new_width) // 2
        start_y = (image.shape[0] - new_height) // 2
        image = image[start_y : start_y + new_height, start_x : start_x + new_width]

        # Split the image
        images = []
        for y in range(0, new_height, self.standard_img_size):
            for x in range(0, new_width, self.standard_img_size):
                img_part = image[
                    y : y + self.standard_img_size, x : x + self.standard_img_size
                ]
                images.append(img_part)

        if index is not None:
            if index < len(images):
                return images[index]
            else:
                raise ValueError("Index is too high")
        else:
            return images

    def save_df_to_file(self):
        """Save the dataframe with dataset info to df_path."""
        self.df.write_csv(self.df_path)

    def load_df_from_file(self):
        """Load the dataframe with dataset info from a file

        Returns:
            bool: if the file existed and the loading was successful
        """
        if os.path.isfile(self.df_path):
            self.df = pl.read_csv(
                self.df_path, null_values="None", infer_schema_length=None
            )
            return True
        else:
            return False

    def mean_gradient(self, image: np.ndarray) -> float:
        """Calculate the mean gradient of an image.

        Args:
            image (np.ndarray): image as numpy array

        Returns:
            float: the mean gradient of the image
        """
        # Compute the gradients along the x and y axes
        gradient_x = sobel(image, axis=1)
        gradient_y = sobel(image, axis=0)

        # Compute the mean gradient (MG)
        shape_x, shape_y = image.shape
        gradient_magnitude = np.sqrt((gradient_x**2 + gradient_y**2) / 2)
        result = np.sum(gradient_magnitude) / (shape_x * shape_y)

        return result
