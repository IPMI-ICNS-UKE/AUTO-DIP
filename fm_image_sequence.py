"""This module defines an Image Sequence that contains images of type fm_image."""
import os
import sys
import PIL.Image
import numpy as np
import tifffile as tf
from typing import Iterator

import fm_image

class ImageSequence():
    """Image sequences containing multiple images of type fm_image
    """
    def __init__(self,
                 path_noisy : str,
                 time_series_params : dict,
                 frame_range : (list|None) = None,
                 path_gt : (str|None) = None,
                 crop_region : (list[int]|None) = None,
                 n_channels : int = 1,
                 transfer_representative_img : (str|None) = None):
        """initialize image sequence

        Args:
            path_noisy (str): path to noisy image
            time_series_params (dict): time series parameters 
            frame_range (list | None, optional): limit denoising to a frame range in file. Defaults to None.
            path_gt (str | None, optional): path to ground truth image. Defaults to None.
            crop_region (list[int] | None, optional): list with the crop region: [left, top, right, bottom]. Defaults to None.
            n_channels (int, optional): number of channels in the input image. Defaults to 1.
            transfer_representative_img (str | None, optional): method to make the size of input image 512x512. Defaults to None.
        """

        self.images = []
        self.path_noisy = path_noisy

        if path_gt == "NULL":
            self.path_gt = None
        else:
            self.path_gt = path_gt
        if crop_region == "NULL":
            self.crop_region = None
        else:
            self.crop_region = crop_region

        self.is_time_series = time_series_params["is_series"]
        if not self.is_time_series:
            self.back_and_forth = False
            self.joint_normalize = False
        else:
            self.back_and_forth = time_series_params["back_and_forth"]
            self.joint_normalize = time_series_params["joint_normalize"]

        if frame_range == "NULL":
            frame_range = None
        assert(frame_range is None or len(frame_range) == 2)
        self.frame_range = frame_range

        self.n_channels = n_channels

        self.transfer_representative_img = transfer_representative_img

        self.load_images()

    def load_images(self):
        """load tif or png images (with their corresponding ground truth image if available)
        """
        # both, noisy images and ground truths are folders
        if os.path.isdir(self.path_noisy) and (self.path_gt and os.path.isdir(self.path_gt)):
            img_paths = list(self.get_image_paths(self.path_noisy))
            img_gt_paths = list(self.get_image_paths(self.path_noisy))

            for i, (file, file_gt) in enumerate(zip(img_paths, img_gt_paths)):
                assert(f"{file}"[-6:-4] == f"{file_gt}"[-6:-4] or file_gt == "NULL")
                img_list = self.prepare_imgs(file,
                                        file_gt = file_gt,
                                        frame_number = i)
                self.images.extend(img_list)

        # there are multiple noisy images in one folder, ground truth not available
        elif os.path.isdir(self.path_noisy) and not self.path_gt:
            img_paths = list(self.get_image_paths(self.path_noisy))
            for i, file in enumerate(img_paths):
                img_list = self.prepare_imgs(file,
                                        file_gt = None,
                                        frame_number = i)
                self.images.extend(img_list)

        # noisy images are in one folder and there is only one ground truth image
        elif os.path.isdir(self.path_noisy) and not os.path.isdir(self.path_gt):
            img_paths = list(self.get_image_paths(self.path_noisy))
            for i, file in enumerate(img_paths):
                img_list = self.prepare_imgs(file,
                                        self.path_gt,
                                        frame_number = i)
                self.images.extend(img_list)

        # there is only one noisy image (and optiionally a ground truth)
        else:
            img_list = self.prepare_imgs(self.path_noisy,
                                         self.path_gt)
            self.images.extend(img_list)

        # adjust frame range
        if self.frame_range:
            self.images = self.images[self.frame_range[0] : self.frame_range[1]]

        self.normalize_images()

    def load_png(self, path : str, crop_region : (list[int]|None) = None) -> tuple[np.array, str]:
        """load a png image

        Args:
            path (str): path to image
            crop_region (list[int], optional): list with the crop region: [left, top, right, bottom].
                Defaults to None.

        Returns:
            np.Array: numpy array with the image, shape: (1, shape_x, shape_y)
            str: pillow image mode of the loaded image
        """
        img = PIL.Image.open(path)
        img_mode = img.mode 
        if crop_region:
            # crop region: [left, top, right, bottom]
            img = img.crop(tuple(crop_region))        
        
        if self.n_channels == 1:
            img_np = np.array(img)[None, ...].astype(np.float32)
        else:
            img_np = np.moveaxis(np.array(img), -1, 0).astype(np.float32)

        return img_np, img_mode

    def load_tif(self, path : str, crop_region : (list[int]|None) = None):
        """load tif image

        Args:
            path (str): path to tif file
            crop_region (list[int], optional): list with the crop region: [left, top, right, bottom].
                Defaults to None.

        Returns:
            (List of np.Array): list of images as np.Array(1, shape_x, shape_y)
        """
        imgs = tf.imread(path)
        imgs = imgs.astype(np.float32)
        if len(imgs.shape) < 3:
            imgs = imgs[None, ...]
        if crop_region:
            imgs = imgs[:, crop_region[1]:crop_region[3],
                        crop_region[0]:crop_region[2]]

        img_list = []
        num_imgs = imgs.shape[0]

        for i in range(num_imgs):
            img = imgs[i][None, ...]
            img_list.append(img)
        return img_list

    def prepare_imgs_png(self, file_noisy : str, file_gt : (str|None) = None,
                         frame_number : (int|None) = None) -> list[fm_image.Image]:
        """create an fm_image for the image

        Args:
            save_dir_noisy (str): path where to save the noisy image
            save_dir_denoised (str): path where to save the denoised image
            frame_number (int): number of the image

        Returns:
            List[fm_image.Image]: list with one fm_image
        """
        img_noisy, img_mode = self.load_png(file_noisy, self.crop_region)

        img_gt = None
        if file_gt:
            assert file_noisy.endswith(".png")
            img_gt, _ = self.load_png(file_gt, self.crop_region)

        image = fm_image.Image(data_noisy = img_noisy,
                                path_noisy = file_noisy,
                                data_gt = img_gt,
                                path_gt = file_gt,
                                crop_region = self.crop_region,
                                frame_number = frame_number,
                                img_mode=img_mode)
        return [image]

    def prepare_imgs_tif(self, file_noisy : str) -> list[fm_image.Image]:
        """create fm_images for each image

        Args:
            file_noisy (str): path to noisy images

        Returns:
            (List[fm_image.Image]): list of fm_images
        """
        imgs_noisy = self.load_tif(file_noisy, self.crop_region)
        if self.path_gt:
            assert file_noisy.endswith(".tif") or file_noisy.endswith(".tf2") or file_noisy.endswith(".tiff")
            imgs_gt = self.load_tif(self.path_gt, self.crop_region)

            images = []
            for i, (img_noisy, img_gt) in enumerate(zip(imgs_noisy, imgs_gt)):
                image = fm_image.Image(data_noisy = img_noisy,
                        path_noisy = file_noisy,
                        data_gt = img_gt,
                        path_gt = self.path_gt,
                        crop_region = self.crop_region,
                        frame_number = i)
                images.append(image)
        else:
            images = []
            for i, img_noisy in enumerate(imgs_noisy):
                image = fm_image.Image(data_noisy = img_noisy,
                        path_noisy = file_noisy,
                        data_gt = None,
                        path_gt = self.path_gt,
                        crop_region = self.crop_region,
                        frame_number = i)
                images.append(image)

        return images

    def prepare_imgs(self, path_noisy : str, file_gt : (str|None) = None,
                     frame_number : (int|None) = None) -> list[fm_image.Image]:
        """create an fm_image for each image

        Args:
            path_noisy (str): path to the noisy image
            save_dir_noisy (str): path to where the noisy images are stored
            save_dir_denoised (str): path to where the denoised images are stored
            frame_number (int, optional): number of the image/frame. Defaults to 0.

        Returns:
            List[fm_image.Image]: list of fm_images
        """
        if path_noisy.endswith(".png"):
            image_list = self.prepare_imgs_png(path_noisy,
                                               file_gt, frame_number)
            return image_list
        if path_noisy.endswith(".tif") or path_noisy.endswith(".tf2") or path_noisy.endswith(".tiff"):
            image_list = self.prepare_imgs_tif(path_noisy)
            return image_list
        print("error: file ends neither with png nor tif")
        sys.exit(1)

    def get_image_paths(self, path : str, rev : bool = False) -> Iterator[str]:
        """iterator over all png and tif files in a directory

        Args:
            path (str): path to the directory
            rev (bool, optional): iterate in reverse alphabetic order. Defaults to False.

        Yields:
            str: path to png/tif file
        """
        if rev:
            files = reversed(sorted(os.listdir(path)))
        else:
            files = sorted(os.listdir(path))
        for file in files:
            if os.path.isfile(os.path.join(path, file)) and (f"{file}".endswith(".png") or
                                                             f"{file}".endswith(".tif")):
                yield f"{path}/{file}"

    def get_image_for_param_transfer(self) -> (np.ndarray):
        """returns image for param transfer (to look for closest image among training data)

        Args:

        Returns:
            np.array : noisy image 
        """

        if self.transfer_representative_img == "input":
            image = self.images[0].data_noisy
            assert(image.shape == (512, 512))
            return image

        if os.path.isfile(self.path_noisy):
            img_path = self.path_noisy
        else:
            img_paths = list(self.get_image_paths(self.path_noisy))
            img_path = img_paths[0]

        if img_path.endswith(".png"):
            image, _ = self.load_png(img_path)
        else:
            assert img_path.endswith(".tif") or img_path.endswith(".tf2") or img_path.endswith(".tiff")
            imagelist, _, _ = self.load_tif(img_path)
            if self.frame_range:
                imagelist = imagelist[self.frame_range[0] : self.frame_range[1]]
            image = imagelist[0]

        if self.transfer_representative_img == "centercrop":

            assert(image.shape[1] >= 512 and image.shape[2] >= 512)
            _, h, w = image.shape
            image_cropped = image[:, 
                                  h // 2 - 256 : h // 2 + 256, 
                                  w // 2 - 256 : w // 2 + 256]
            assert(image_cropped.shape == (1, 512, 512))

            return image_cropped

        if self.transfer_representative_img == "resize":

            image_pil = PIL.Image.fromarray(image[0])
            image_downsized =  image_pil.resize((512, 512))
            image_downsized_np = np.array(image_downsized)[None, ...]
            assert(image_downsized_np.shape == (1, 512, 512))
            return image_downsized_np
        
        else:
            print(f"Representative image variant {self.transfer_representative_img} was not found!")
            assert False

    def normalize_images(self):
        """Normalize images to [0,1] range. 
        Usually normalizes each image separately, except joint normalize fro time series is enabled.
        """

        if self.joint_normalize:

            imgs_max = max(x.data_noisy.max() for x in self.images)
            imgs_min = min(x.data_noisy.min() for x in self.images)

            for image in self.images:
                image.scale_all_img_versions_0_1(imgs_min, imgs_max)

        else:

            for image in self.images:
                imgs_min = image.data_noisy.min()
                imgs_max = image.data_noisy.max()
                image.scale_all_img_versions_0_1(imgs_min, imgs_max)
            