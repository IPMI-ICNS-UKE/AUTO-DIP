"""This module provides a loss function containing MSE, a convolution with a PSF 
or the Lucy-Richardson functional"""
import sys
import numpy as np
from torchmetrics.image import TotalVariation
import torch
import ignite
import lpips
from collections.abc import Callable
from typing import Concatenate


from psf import psf

class LossFunction(torch.nn.Module):
    """loss function

    Args:
        torch (nn.Module): torch module
    """
    def __init__(self, dtype : str, loss_type_main : str, loss_type2 : str, loss2_fact : float = 0, 
                 loss2_incr_fact : float = 0, regularizer : ( str|None) = None, regularizer_fact : float = 0, 
                 regularizer_incr_fact : float = 0, psf_params : (dict|None) = None, superres_factor : int = 1):
        """initialize loss function

        Args:
            dtype (str): dtype for loss calculation
            loss_type_main (str): loss type, can be: mse, psf, rl, kl, l1, lpips
            loss_type2 (str): second loss type, can be: mse, psf, rl, kl, l1, lpips
            loss2_fact (float, optional): influence of the second loss. Defaults to 0.
            loss2_incr_fact (float, optional): decrease/increase the loss2 factor with every epoch. Defaults to 0.
            regularizer (str | None, optional): optional explicit regularizer, can be tv or tm. Defaults to None.
            regularizer_fact (float, optional): impact of the regularizer . Defaults to 0.
            regularizer_incr_fact (float, optional): decrease/increase the regularizer factor with every epoch. Defaults to 0.
            psf_params (dict | None, optional): parameters for the point spread funtion. Defaults to None.
            superres_factor (int, optional): super-resolution factor. Defaults to 1.
        """
        super().__init__()

        self.superres_factor = superres_factor

        self.dtype = dtype
        if psf_params:
            self.dims = 2
            self.psf_params = psf_params
            self.init_psf()

        self.loss_type_main = loss_type_main

        self.loss_func_main = self.get_loss_func(loss_type_main)
        self.loss_func2 = self.get_loss_func(loss_type2)

        self.loss2_incr_fact = loss2_incr_fact
        self.loss2_fact = loss2_fact

        self.regularizer = self.get_regularizer(regularizer)
        self.regularizer_fact = regularizer_fact
        self.regularizer_incr_fact = regularizer_incr_fact

        self.psnr_fun = None
        self.ssim_fun = None
        self.lpips_fun = None

    def forward(self, output : torch.Tensor, target : torch.Tensor, **kwargs) -> torch.Tensor:
        """calculate the loss between output and target

        Args:
            output (torch.Tensor): network output
            target (torch.Tensor): target

        Returns:
            torch.Tensor: loss
        """
        loss = self.general_loss_func(output, target, **kwargs)
        return loss

    def general_loss_func(self, output : torch.Tensor, target : torch.Tensor, **kwargs) -> torch.Tensor:
        """calls the configured loss function(s)

        Args:
            output (torch.Tensor): network output
            target (torch.Tensor): target
            epoch (int): number of the epoch

        Returns:
            torch.Tensor: loss
        """
        loss_main = self.loss_func_main(output, target, optim_loss = True, **kwargs)
        if self.loss_func2:
            loss2 = self.loss_func2(output, target, optim_loss = True, **kwargs)
        else:
            loss2 = 0
        if self.regularizer:
            regularization = self.regularizer(output)
        else:
            regularization = 0

        fact_loss2 = self.loss2_fact + (self.loss2_incr_fact * kwargs["num_epoch"])
        fact_loss_main = 1 - fact_loss2
        regularizer_fact = self.regularizer_fact + (self.regularizer_incr_fact * kwargs["num_epoch"])
        loss = fact_loss_main * loss_main + fact_loss2 * loss2 +  regularizer_fact * regularization

        return loss

    ## loss functions
    def loss_mse(self, output : torch.Tensor, target : torch.Tensor, **kwargs) -> torch.Tensor:
        """calculate the mean squared error between output and target

        Args:
            output (torch.Tensor): network output
            target (torch.Tensor): target

        Returns:
            torch.Tensor: MSE
        """
        return self.mse(output, target)
    
    def loss_l1(self, output : torch.Tensor, target : torch.Tensor, **kwargs) -> torch.Tensor:
        """calculate the l1 norm of the difference between output and target

        Args:
            output (torch.Tensor): network output
            target (torch.Tensor): target

        Returns:
            torch.Tensor: L1 norm of output-target distance
        """
        return torch.mean(torch.abs(output - target))
    
    def loss_psf(self, output : torch.Tensor, target : torch.Tensor, **kwargs) -> torch.Tensor:
        """Calculates the MSE between the target and the output convolved with the PSF

        Args:
            output (torch.Tensor): network output
            target (torch.Tensor): target

        Returns:
            torch.Tensor: MSE(output * PSF, target)
        """
        conv = self.conv_with_psf(output)
        return self.mse(torch.squeeze(conv), target)
    

    def loss_richardson_lucy(self, output : torch.Tensor, target : torch.Tensor, **kwargs) -> torch.Tensor:
        """calculate the Richardson-Lucy functional

        Args:
            output (torch.Tensor): network output
            target (toch.Tensor): target

        Returns:
            torch.Tensor: RL functional result
        """
        conv = torch.squeeze(self.conv_with_psf(output))
        integrant = conv - target * torch.log(conv + sys.float_info.epsilon)
        return torch.sum(integrant)
    
    def ssim(self, output : torch.Tensor, target : torch.Tensor, **kwargs) -> torch.Tensor:
        """Calculate the structural similarity index between output and target
        Either call the optimize version that is differentiable
        or call the faster and more stable comparison version.

        Args:
            output (np.ndarray): network output
            target (np.ndarray): target

        Returns:
            torch.Tensor: SSIM
        """

        if "optim_loss" in kwargs and kwargs["optim_loss"]:
            return self.ssim_optim(output, target, **kwargs)
        else:
            return self.ssim_comp(output, target, **kwargs)

    def ssim_optim(self, output : torch.Tensor, target : torch.Tensor, **kwargs) -> torch.Tensor:
        """Calculate the structural similarity index between output and target
        Differentiable version that can be used for net optimization

        Args:
            output (np.ndarray): network output
            target (np.ndarray): target

        Returns:
            torch.Tensor: SSIM
        """
        # constants
        window_size=11
        C1=0.01**2
        C2=0.03**2

        # Gaussian kernel
        def create_window(window_size, channel, device, dtype):
            coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
            g = torch.exp(-(coords**2) / (2 * (1.5**2)))
            g = g / g.sum()
            window = g[:, None] @ g[None, :]
            window = window.expand(channel, 1, window_size, window_size)
            return window

        channel = output.size(1)
        window = create_window(window_size, channel, output.device, output.dtype)

        # Local means
        mu_x = torch.nn.functional.conv2d(output, window, padding=window_size//2, groups=channel)
        mu_y = torch.nn.functional.conv2d(target, window, padding=window_size//2, groups=channel)

        mu_x2 = mu_x ** 2
        mu_y2 = mu_y ** 2
        mu_xy = mu_x * mu_y

        # Local variances
        sigma_x2 = torch.nn.functional.conv2d(output * output, window, padding=window_size//2, groups=channel) - mu_x2
        sigma_y2 = torch.nn.functional.conv2d(target * target, window, padding=window_size//2, groups=channel) - mu_y2
        sigma_xy = torch.nn.functional.conv2d(output * target, window, padding=window_size//2, groups=channel) - mu_xy

        # SSIM formula
        ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2))
        ssim = ssim_map.mean()

        return 1 - ssim  # use as a loss (lower = better)

    def ssim_comp(self, output : torch.Tensor, target : torch.Tensor, **kwargs) -> torch.Tensor:
        """Calculate the structural similarity index between output and target

        Args:
            output (np.ndarray): network output
            target (np.ndarray): target

        Returns:
            torch.Tensor: SSIM
        """
        if not self.ssim_fun:
            self.ssim_fun = ignite.metrics.SSIM(data_range=1.0)
        else:
            self.ssim_fun.reset()
        self.ssim_fun.update((output, target))

        ssim = torch.tensor(self.ssim_fun.compute())

        return ssim
    
    def psnr(self, output : torch.Tensor, target : torch.Tensor, **kwargs) -> torch.Tensor:
        """calculate the peak-signal-to-noise ratio between the output and the target
        Either call the optimize version that is differentiable
        or call the faster and more stable comparison version

        Args:
            output (torch.Tensor): network output
            target (torch.Tensor): target

        Returns:
            torch.Tensor: PSNR
        """

        if "optim_loss" in kwargs and kwargs["optim_loss"]:
            return self.psnr_optim(output, target, **kwargs)
        else:
            return self.psnr_comp(output, target, **kwargs)

    def psnr_comp(self, output : torch.Tensor, target : torch.Tensor, **kwargs) -> torch.Tensor:
        """calculate the peak-signal-to-noise ratio between the output and the target

        Args:
            output (torch.Tensor): network output
            target (torch.Tensor): target

        Returns:
            torch.Tensor: PSNR
        """
        if not self.psnr_fun:
            self.psnr_fun = ignite.metrics.PSNR(data_range=1.0, device='cuda')
        else:
            self.psnr_fun.reset()
        self.psnr_fun.update((output, target))

        psnr = torch.tensor(self.psnr_fun.compute())

        return psnr

    def psnr_optim(self, output : torch.Tensor, target : torch.Tensor, **kwargs) -> torch.Tensor:
        """calculate the peak-signal-to-noise ratio between the output and the target
        Differentiable version that can be used in net optimization

        Args:
            output (torch.Tensor): network output
            target (torch.Tensor): target

        Returns:
            torch.Tensor: PSNR
        """
        mse = torch.mean((output - target) ** 2)
        max_val = torch.tensor(1, dtype=output.dtype, device=output.device)
        psnr = 20 * torch.log10(max_val) - 10 * torch.log10(mse)
        return -psnr  # negative because we minimize loss
    
    def lpips(self, output : torch.Tensor, target : torch.Tensor, **kwargs) -> torch.Tensor:
        """calculate the LPIPS between the output and the target

        Args:
            output (torch.Tensor): network output
            target (torch.Tensor): target

        Returns:
            torch.Tensor: LPIPS
        """
        if not self.lpips_fun:
            if "optim_loss" in kwargs and kwargs["optim_loss"]:
                # VGG is better suited for net optimization 
                self.lpips_fun = lpips.LPIPS(net='vgg', verbose=False).to(device="cuda")
            else:
                # Alex is faster and better for forward comparison
                self.lpips_fun = lpips.LPIPS(net='alex', verbose=False).to(device="cuda")

        lpips_result = self.lpips_fun.forward(output, target)

        return lpips_result

    ## regularizer
    def tv_norm(self, image : np.ndarray) -> float:
        """calculate the TV-norm of an image

        Args:
            image (np.ndarray): image as numpy array

        Returns:
            float: tv norm
        """
        tv = 0
        for i in range(len(image)-1):
            for j in range(len(image[0])-1):
                tv += torch.sqrt((image[i, j+1] - image[i, j])**2 +
                                 (image[i+1, j] - image[i, j])**2)
        return tv

    def tikhonov_miller(self, img : np.ndarray) -> float:
        """calculate the Tikhonov-Miller norm of an image

        Args:
            img (np.ndarray): image as numpy array

        Returns:
            float: Tikhonov-Miller norm of the image
        """
        diff1 = (img[..., 1:, :] - img[..., :-1, :])**2
        diff2 = (img[..., :, 1:] - img[..., :, :-1])**2

        res1 = diff1.abs().sum([1, 2, 3])
        res2 = diff2.abs().sum([1, 2, 3])
        tm = res1 + res2
        return tm


    ## helper functions
    def get_loss_func(self, loss_type : str) -> Callable[Concatenate[torch.Tensor, torch.Tensor, ...], torch.Tensor]:
        """choose the correct loss function

        Args:
            loss_type (str): loss type string, possible values: mse, psf, kl, rl

        Returns:
            fm_loss.LossFunction: loss funtion
        """
        match loss_type:
            case 'mse':
                return self.loss_mse
            case 'l1':
                return self.loss_l1
            case 'psf':
                return self.loss_psf
            case 'rl':
                return self.loss_richardson_lucy
            case 'ssim':
                return self.ssim
            case 'psnr':
                return self.psnr
            case 'lpips':
                return self.lpips
            case _:
                return None

    def get_regularizer(self, regularizer : str) -> Callable[[np.ndarray], float]:
        """get the regularizer function

        Args:
            regularizer (str): regularizer type, possible values: tv, tm, sparse

        Returns:
            fm_loss.regularizer: regularizer function
        """
        match regularizer:
            case 'tv':
                tv = TotalVariation().to(device='cuda')
                return tv
            case 'tm':
                return self.tikhonov_miller
            case '_':
                return None

    def init_psf(self) -> torch.Tensor:
        """initialize point spread function

        Returns:
            np.ndarray: PSF as numpy array
        """
        psf_obj = psf.PSF(self.dims, **self.psf_params)
        psf_arr = psf_obj.data
        if self.superres_factor > 1:
            psf_arr = np.kron(psf_arr, np.ones((self.superres_factor, self.superres_factor)))
        psf_arr_torch = torch.tensor(psf_arr, device='cuda:0').type(self.dtype)
        self.psf_arr = psf_arr_torch
        return psf_arr_torch

    def mse(self, output : torch.Tensor, target : torch.Tensor) -> torch.Tensor:
        """Calculate the mean squared error

        Args:
            output (torch.Tensor): network output
            target (torch.Tensor): target

        Returns:
            torch.Tensor: MSE
        """
        return torch.mean((output - target)**2 + sys.float_info.epsilon)

    def conv_with_psf(self, image : torch.Tensor) -> torch.Tensor:
        """convolve the given image with a PSF

        Args:
            image (torch.Tensor): image

        Returns:
            torch.Tensor: image convolved with the PSF
        """
        psf_size = self.psf_params['xysize'] * self.superres_factor
        in_psf = self.psf_arr.view(1, 1, psf_size, psf_size).repeat(1, 1, 1, 1)
        in_img = image.repeat(1, 1, 1, 1)
        conv = torch.real(torch.fft.ifftshift(torch.fft.ifftn(torch.fft.fftn(in_img) *
                                                              torch.fft.fftn(in_psf))))
        return conv

    def get_convolved_orig_mix(self, image : torch.Tensor) -> torch.Tensor:
        """Returns a mix between the original image and the image convolved with the psf, 
        loss2_fact weighs the two images

        Args:
            image (torch.Tensor): network output

        Returns:
            torch.Tensor: image 
        """
        conv = self.conv_with_psf(image)
        mix = torch.squeeze(conv) * self.loss2_fact + (1 - self.loss2_fact) * image
        return mix

    def get_losses(self, output : torch.Tensor, target : torch.Tensor, num_epoch : int, 
                   loss_types : list[str] = ['mse', 'ssim', 'psnr']) -> dict:
        """calculate different losses between output and target: MSE, the configured one, 
        PSNR, SSIM, tv norm of the output and tv norm of the target

        Args:
            output (torch.Tensor): network output
            target (torch.Tensor): target
            num_epoch (int): number of the epoch

        Returns:
            dict: dict with different losses
        """
        loss_dict = {}
        for loss_type in loss_types:
            if loss_type == "configured":
                loss_dict[loss_type] =  self.forward(output, target, num_epoch = num_epoch).item()
            else:
                loss_func = self.get_loss_func(loss_type)
                if loss_func:
                    loss_dict[loss_type] =  loss_func(output, target).item()
        return loss_dict

    def get_measures(self, image : torch.Tensor, img_postfix : str = "", 
                     measure_types : list[str] = ["tv"]) -> dict:
        """get measures for input image

        Args:
            image (torch.Tensor): image
            img_postfix (str, optional): image type postfix to put behind measure name for identification. Defaults to "".
            measure_types (list[str], optional): measures to measure for the input image. Defaults to ["tv"].

        Returns:
            dict: dict with measures
        """
        measure_dict = {}
        for measure_type in measure_types:
            measure_fun = self.get_regularizer(measure_type)
            if measure_fun:
                measure_dict[f"{measure_type}{img_postfix}"] =  measure_fun(image).item()
        return measure_dict
