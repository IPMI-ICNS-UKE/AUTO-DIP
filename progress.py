import os
import torch
import mlflow
import datetime
import numpy as np
from PIL import Image
from pathlib import Path
from dip.utils.common_utils import torch_to_np

import fm_loss
import fm_image


os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "false"

class NetStatus():
    def __init__(self, 
                 params: dict,
                 loss_function: fm_loss.LossFunction,
                 image: fm_image.Image) -> None:
        """initialize net status

        Args:
            params (dict): net parameters
            loss_function (fm_loss.LossFunction): loss function
            image (fm_image.Image): input image
        """
        
        self.params = params
        self.exp_weight = self.params["net"]["exp_weight"]
        self.log_with_mlflow = params["save_and_log"]["mlflow"]
        self.exp_name = params["save_and_log"]["mlflow_exp_name"]
        self.run_name = params["save_and_log"]["mlflow_run_name"]
        self.loss_types = params["save_and_log"]["mlflow_log_loss_types"]
        self.save_img_path = params["save_and_log"]["save_path"]

        self.out_avg = None
        self.psnr_last = 0

        self.last_net = None
        
        self.loss_function = loss_function

        self.save_imgs = {}
        self.loss_list_noisy_dict = {}
        self.loss_list_gt_dict = {}

        self.image = image

        self.best_losses = {}
        self.best_imgs = {}

    def reset(self, image : fm_image.Image):
        """reset saved losses and images and set a new input image

        Args:
            image (fm_image.Image): new input image
        """
        self.save_imgs = {}
        self.loss_list_noisy_dict = {}
        self.loss_list_gt_dict = {}

        self.image = image

        self.best_losses = {}
        self.best_imgs = {}
        self.save_imgs = {}
        self.loss_list_noisy_dict = {}
        self.loss_list_gt_dict = {}

    def update_smoothed_output(self, out : torch.Tensor):
        """update smoothed output image

        Args:
            out (torch.Tensor): smoothed output image
        """
        if self.out_avg is None:
            self.out_avg = out
        else:
            self.out_avg = self.out_avg * \
                self.exp_weight + out * (1 - self.exp_weight)
        
    def mlflow_log(self):
        """log all info from the class in MLflow:
            - loss values for every iteration and different measures
            - all saved images 
            - the noisy image 
            - the ground truth image if it exists 
        """
        if not self.params["save_and_log"]["mlflow"]:
            return

        best_metrics = self.get_best_metrics()
        static_metrics = self.get_static_metrics()

        mlflow.set_tracking_uri("http://localhost:50002")
        mlflow.set_experiment(self.exp_name)
        with mlflow.start_run(run_name = f"{self.run_name}"):

            for param_set in self.params:
                if self.params[param_set]:
                    mlflow.log_params(self.params[param_set])

            for loss_type in self.loss_list_noisy_dict:
                for i, loss in enumerate(self.loss_list_noisy_dict[loss_type]):
                    mlflow.log_metric(loss_type, loss, step=i)

            if self.loss_list_gt_dict:
                for loss_type in self.loss_list_gt_dict:
                    for i, loss in enumerate(self.loss_list_gt_dict[loss_type]):
                        mlflow.log_metric(f"{loss_type}_gt", loss, step=i) 

            for key, item in static_metrics.items():
                mlflow.log_metric(key, item, step=0)

            for key, item in best_metrics.items():
                mlflow.log_metric(key, item["best_value"], step=item["epoch"])

            if self.params["save_and_log"]["mlflow_log_images"]:
                for num_img in self.save_imgs:
                    mlflow.log_image(self.save_imgs[num_img], f"out_{num_img:05d}.png")
            
                mlflow.log_image(np.moveaxis(self.image.data_noisy, 0, -1), "noisy.png")

                if self.image.data_gt is not None:
                    mlflow.log_image(np.moveaxis(self.image.data_gt, 0, -1), "gt.png")

                    for loss_type, img in self.best_imgs.items():
                        img = np.moveaxis(img, 0, -1)
                        mlflow.log_image(img, f"best_{loss_type}_gt.png")

    def save_images(self):
        """save all images from the class in local directory
            - all saved images 
            - the noisy image 
            - the ground truth image if it exists 
        """
        if not self.params["save_and_log"]["save_imgs"]:
            return
        
        # get new unique folder named after image and time
        save_path_base = Path(self.save_img_path)
        folder_name_base = self.image.get_base_file_name()
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_path = save_path_base / f"{folder_name_base}_{current_time}"
        os.makedirs(save_path)

        for num_img in self.save_imgs:
            img_pil = self.get_img_for_saving(self.save_imgs[num_img][:, :, 0])
            img_pil.save(save_path / f"out_{num_img:05d}.png")
    
        img_pil = self.get_img_for_saving(self.image.data_noisy[0, :, :])
        img_pil.save(save_path / "noisy.png")

        if self.image.data_gt is not None:
            img_pil = self.get_img_for_saving(self.image.data_gt[0, :, :])
            img_pil.save(save_path / "gt.png")

            for loss_type, img in self.best_imgs.items():
                img_pil = self.get_img_for_saving(img[0, :, :])
                img_pil.save(save_path / f"best_{loss_type}_gt.png")

    def get_img_for_saving(self, image : np.ndarray) -> Image:
        """de-normalize image to original range, choose the correct format and adjust axis

        Args:
            image (np.ndarray): output image

        Returns:
            PIL.Image: de-normalized image, ready for saving
        """
        
        img_scaled = image * (self.image.images_max - self.image.images_min) + self.image.images_min

        if self.image.images_max > 255 or (self.image.img_mode and self.image.img_mode == "I;16"):
            img_scaled = img_scaled.astype(np.int16)
            mode = "I;16"
        else:
            img_scaled = img_scaled.astype(np.int8)
            mode = "L"
        img_pil = Image.fromarray(img_scaled, mode=mode)

        return img_pil

    def log_current_loss(self, output : torch.Tensor, num_epoch : int):
        """log the loss between output and noisy image (also gt image if available) with different metrics

        Args:
            output (torch.Tensor): network output   
            num_epoch (int): number of the epoch.
        """
        loss_types = self.loss_types
        loss_dict = self.loss_function.get_losses(output, self.image.img_noisy_torch, num_epoch, loss_types)
        loss_dict = loss_dict | self.loss_function.get_measures(output, img_postfix = "")
        self.loss_list_noisy_dict = append_to_list_dict(self.loss_list_noisy_dict, loss_dict)

        if self.image.img_gt_torch is not None:
            loss_dict = self.loss_function.get_losses(output, self.image.img_gt_torch, num_epoch, loss_types)
            self.loss_list_gt_dict = append_to_list_dict(self.loss_list_gt_dict, loss_dict)

            if num_epoch > 100:
                self.track_best_loss_values(loss_dict, loss_types, output, num_epoch)

    def track_best_loss_values(self, loss_dict : dict, loss_types : list[str], output : torch.Tensor, 
                               num_epoch : int):
        """track best loss values

        Args:
            loss_dict (dict): dict with losses
            loss_types (list[str]): list of loss types to be tracked
            output (torch.Tensor): network output
            num_epoch (int): number of the current epoch
        """
        if not self.best_losses:
            for loss_type in loss_types:
                self.best_losses[loss_type] = {"best_value" : loss_dict[loss_type],
                                               "epoch": num_epoch}

                self.best_imgs[loss_type] = torch_to_np(output)
        else:
            for loss_type, item in loss_dict.items():
                if loss_type in ["ssim", "psnr"]:
                    def is_better(a, b):
                        return True if a > b["best_value"] else False
                else:
                    def is_better(a, b):
                        return False if a > b["best_value"] else True
                if is_better(item, self.best_losses[loss_type]):
                    self.best_losses[loss_type]["best_value"] = item
                    self.best_losses[loss_type]["epoch"] = num_epoch
                    
                    self.best_imgs[loss_type] = torch_to_np(output)

    def get_static_metrics(self):
        """get single image metrics for noisy and gt image if avaible

        Returns:
            dict: dict with metrics
        """
        metrics_dict = {}
        metrics_dict = metrics_dict | self.loss_function.get_measures(
            self.image.img_noisy_torch, img_postfix = "_noisy")
        if self.image.img_gt_torch is not None:
            metrics_dict = metrics_dict | self.loss_function.get_measures(
                self.image.img_gt_torch, img_postfix = "_gt")

        return metrics_dict

    def get_best_metrics(self):
        """get best metrics

        Returns:
            dict: dict with best values and epochs for each metric
        """
        metrics_dict = {}
        for key in self.loss_list_noisy_dict:
            if key in ['ssim', 'psnr']:
                def best_fun(a):
                    return max(enumerate(a), key=lambda x: x[1])
            else:
                def best_fun(a):
                    return min(enumerate(a), key=lambda x: x[1])
            
            metrics_dict[f"{key}_best"] = {}
            metrics_dict[f"{key}_best"]["epoch"], metrics_dict[f"{key}_best"]["best_value"] = best_fun(self.loss_list_noisy_dict[key])

        for loss_type, item in self.best_losses.items():
            metrics_dict[f"{loss_type}_gt_best"] = {}
            metrics_dict[f"{loss_type}_gt_best"]["best_value"] = item["best_value"]
            metrics_dict[f"{loss_type}_gt_best"]["epoch"] = item["epoch"]
        return metrics_dict

    def log_output(self, image : torch.Tensor, iteration : int):
        """save image in list for logging in MLflow

        Args:
            image (torch.Tensor): image to save
            iteration (int): number of the iteration
        """
        if self.params["save_and_log"]["mlflow_log_images"] or self.params["save_and_log"]["save_imgs"]:
            self.save_imgs[iteration] = np.moveaxis(torch_to_np(image), 0, -1)

    def check_pnsr_developement(self, fm_image : fm_image.Image, output_image : torch.Tensor, 
                                net : torch.nn.modules.container.Sequential, epoch : int, verbosity : int = 0):
        """check if the psnr got better since the last call, 
        resets the network to last checkpoint if the PSNR got worse by more than 5

        Args:
            fm_image (fm_image.Image): fm_image containing the noisy image
            output_image (torch.Tensor): network output
            net (torch.nn.modules.container.Sequential): network 
            epoch (int): number of the current epoch
            verbosity (int, optional): verbosity. Defaults to 0.
        """
        psnr = self.loss_function.psnr(output_image, fm_image.img_noisy_torch)

        if psnr - self.psnr_last < -5:
            if epoch == 200:
                self.out_avg = None
            if verbosity > 0:
                print("Falling back to previous checkpoint.")
            for new_param, net_param in zip(self.last_net, net.parameters()):
                net_param.data.copy_(new_param.cuda())
        else:
            self.last_net = [x.detach().cpu() for x in net.parameters()]
            self.psnr_last = psnr


def append_to_list_dict(list_dict : dict, single_dict : dict) -> dict:
    """append the entries of a dict (single_dict) to a dict that has lists as values

    Args:
        list_dict (dict): dict with lists as values
        single_dict (dict): dict with the same keys and single values

    Returns:
        dict: list_dict with appended items from single_dict
    """
    if list_dict:
        for key, val in single_dict.items():
            list_dict[key].append(val)
    else:
        for key, val in single_dict.items():
            list_dict[key] = [val]
    return list_dict
