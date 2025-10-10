import torch

from dip import models
from dip.utils import denoising_utils
from dip.models.downsampler import Downsampler

import fm_loss
import fm_image
import fm_image_sequence
from progress import NetStatus
from net_input import NetInput

dtype = torch.cuda.FloatTensor


class Training():
    def __init__(self, parameters : dict):
        """initialize training

        Args:
            parameters (dict): parameter dict
        """
        self.params = parameters

        superres_params = parameters['superresolution']
        self.net_params = parameters['net']
        self.superres_factor = superres_params['superres_factor']

        self.downsampler = None
        if superres_params['superres_factor'] > 1:
            downsample_kernel = superres_params['downsample_kernel']
            # define downsampler for superresolution
            self.downsampler = Downsampler(n_planes=1, factor=self.superres_factor,
                                    kernel_type=downsample_kernel, phase=0.5,
                                    preserve_size=True).type(dtype)
            

        self.loss = fm_loss.LossFunction(dtype, **parameters['loss'],
                                        psf_params=parameters['psf'],
                                        superres_factor=self.superres_factor)
                
        self.net_status = None
        self.net = None

    def start_training(self, image_sequence : fm_image_sequence.ImageSequence):
        """start training

        Args:
            image_sequence (fm_image_sequence.ImageSequence): image sequence to be denoised
        """
        num_imgs = len(image_sequence.images)

        if image_sequence.is_time_series:
            for r in range(self.params['image']['num_runs']):
                if image_sequence.back_and_forth:
                    self.params['save_and_log']['save_imgs'] = False

                print("Fitting net to first image ...")
                self.run(image=image_sequence.images[0], first_image=True)

                print("Iterating through images in chronological order ...")
                for i, image in enumerate(image_sequence.images[1:]):
                    print(f"Fitting net to image {i+2}/{num_imgs} ...")
                    self.run(image=image)

                self.params['save_imgs'] = True
                if image_sequence.back_and_forth:
                    print("Iterating through images in reverse order...")
                    for i, image in enumerate(reversed(image_sequence.images[:-1])):
                        print(f"Fitting net to image number {num_imgs-i-1}" +
                              f"(finished ({i+1}/{num_imgs}))")
                        self.run(image=image)
        else:
            for i, image in enumerate(image_sequence.images):
                for r in range(self.params['image']['num_runs']):
                    print(f"Fitting net to image {i+1}/{num_imgs} (run {r+1}) ...")
                    self.run(image)                

    def init_network(self, params : dict) -> torch.nn.modules.container.Sequential:
        """initialize network

        Args:
            params (dict): parameters for network

        Returns:
            torch.nn.modules.container.Sequential : neural network
        """
        self.net = models.get_net(input_depth=params['input_depth'],
                                    NET_TYPE=params['net_type'],
                                    pad=params['pad'],
                                    n_channels=params['n_channels'],
                                    act_fun=params['act_fun'],
                                    skip_n33d=params['skip_n33d'],
                                    skip_n33u=params['skip_n33u'],
                                    skip_n11=params['skip_n11'],
                                    num_scales=params['num_scales'],
                                    upsample_mode=params['upsample_mode'],
                                    downsample_mode=params['downsample_mode']).type(dtype)

    def closure(self, image : fm_image.Image, iteration : int):
        """perform one iteration of training

        Args:
            image (fm_image.Image): image of class fm_image to be reconstructed
            iteration (int): number of the iteration
        """

        self.net_input.update_net_input()

        out = self.net(self.net_input.net_input)

        self.net_status.update_smoothed_output(out.detach())

        if self.superres_factor > 1:
            out_comp = self.downsampler(out)
        else:
            out_comp = out

        total_loss = self.loss.forward(out_comp, image.img_noisy_torch, num_epoch = iteration, net = self.net)

        total_loss.backward()

        self.net_status.log_current_loss(output = out_comp, num_epoch = iteration)
        
        if iteration % self.net_params["checkpoint_interval"] == 0 or iteration + 1 == self.net_params["num_iter"]:
            if self.params["save_and_log"]["verbosity"] > 0:
                print(f"Iteration {iteration} Loss {total_loss.item()} PSNR_noisy: {self.net_status.psnr_last}")
            
            self.net_status.log_output(out, iteration)

            if iteration > 5:
                self.net_status.check_pnsr_developement(image, 
                                                        out_comp, 
                                                        self.net, 
                                                        iteration,
                                                        self.params["save_and_log"]["verbosity"])
        
    
    def run(self, image : fm_image.Image, first_image : bool = False):
        """load everything and start training the network

        Args:
            image (fm_image.Image): image
            first_image (bool, optional): defines if it is the first image of a series. 
                Defaults to False.
        """

        # init network for the first picture or if we do not have a time series
        if first_image or self.params['time_series']['is_series'] != "True":
            # configure network
            self.init_network(self.net_params)

            self.net_input = NetInput(self.net_params, 
                                     image, 
                                     superres_factor=self.superres_factor,
                                     seed = 0)
        
            self.net_status = NetStatus(params = self.params,
                                loss_function = self.loss,
                                image = image)
        
        else:
            self.net_status.reset(image)

        # get parameters for optimizer
        params_optimizer = denoising_utils.get_params(
            self.net_params['OPT_OVER'], self.net, self.net_input.net_input)

        # set learning rate and number of iterations depending on if it is a time series or not
        if first_image or self.params['time_series']['is_series'] != "True":
            self.learning_rate = self.net_params["learning_rate"]
            self.num_iter = self.net_params["num_iter"]
        else:
            series_params = self.params["time_series"]
            self.learning_rate = series_params["learning_rate"]
            self.num_iter = series_params["num_iter"]

        self.optimize(params_optimizer, image)

        # get final output
        out = self.net_status.out_avg
        image.data_denoised = denoising_utils.torch_to_np(out)

        self.net_status.mlflow_log()
        self.net_status.save_images()
    
    def optimize(self, parameters_optimizer : dict, fm_image : fm_image.Image):
        """Runs optimization loop.

        Args:
            parameters_optimizer (dict): parameters for the optimizer
            fm_image (fm_image.Image): image

        """
        if self.params["save_and_log"]["verbosity"] > 0:
            print('Starting optimization with ADAM')
        
        optimizer = torch.optim.Adam(parameters_optimizer, lr=self.learning_rate)

        for j in range(self.num_iter):
            optimizer.zero_grad()
            self.closure(fm_image, j)
            optimizer.step()
    