from dip.utils import denoising_utils
import torch

import fm_image

dtype = torch.cuda.FloatTensor


class NetInput():
    def __init__(self, net_params : dict, image : fm_image.Image, superres_factor : int = 1, 
                seed : (int|None) = None):
        """initialize net input

        Args:
            net_params (dict): network parameters
            image (fm_image.Image): image
            superres_factor (int, optional): super-resolution factor. Defaults to 1.
            seed (int | None, optional): seed. Defaults to None.
        """
        # define a generator for setting always the same seed
        if seed:
            generator = torch.Generator()
            generator.manual_seed(seed)
            torch.manual_seed(seed)
        else:
            generator = None

        img_shape = image.data_noisy.shape[1] * superres_factor, image.data_noisy.shape[2] * superres_factor
        
        # get random net input
        self.net_input = denoising_utils.get_noise(input_depth=net_params['input_depth'],
                                                   method=net_params["INPUT"],
                                                   spatial_size=img_shape,
                                                   generator=generator).type(dtype).detach()
        
        self.net_input_saved = self.net_input.detach().clone()
        self.net_input_orig = self.net_input.detach().clone()
        
        self.reg_noise_std = net_params["reg_noise_std"]


    def update_net_input(self):
        """update the network input
        """
        if self.reg_noise_std > 0:
            self.net_input = self.net_input_saved + \
                (self.net_input_orig.normal_() * self.reg_noise_std)
        else:
            self.net_input = self.net_input_saved