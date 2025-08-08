import torch
import tqdm
from collections import OrderedDict
from util import get_clamped_psnr
from util import to_coordinates_and_features
from util import apply_idct2, apply_dct2
import torch_dct as dct
import kornia
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torchvision.transforms as T
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.fft as fft
from laplacianpyramid import LaplacianPyramid
from torchmetrics import StructuralSimilarityIndexMeasure
import higher   
import copy
from copy import deepcopy
import numpy as np


class Trainer():
    def __init__(self, representation, lr=1e-3, print_freq=1,with_base_learner = False):
        """Model to learn a representation of a single datapoint.

        Args:
            representation (siren.Siren): Neural net representation of image to
                be trained.
            lr (float): Learning rate to be used in Adam optimizer.
            print_freq (int): Frequency with which to print losses.
        """
        self.representation = representation
        self.optimizer = torch.optim.Adam(self.representation.parameters(), lr=lr)
        self.print_freq = print_freq
        self.steps = 0  # Number of steps taken in training
        self.loss_func = torch.nn.MSELoss()
        self.best_vals = {'psnr': 0.0, 'loss': 1e8}
        self.logs = {'psnr': [], 'loss': []}
        # Store parameters of best model (in terms of highest PSNR achieved)
        self.best_model = OrderedDict((k, v.detach().clone()) for k, v in self.representation.state_dict().items())

        if with_base_learner:
            self.base_learner_optimizer = torch.optim.Adam(self.representation.base_learner.parameters(), lr=lr)
            self.correction_optimizer = torch.optim.Adam(self.representation.corrector.parameters(), lr=lr)
            self.best_base_learner = OrderedDict((k, v.detach().clone()) for k, v in self.representation.base_learner.state_dict().items())
            self.base_model_logs ={'psnr': [], 'loss': []}
            self.base_model_best_vals = {'psnr': 0.0, 'loss': 1e8}

    def chroma_upsample(self,chroma, target_size):
        """
        Upsample chroma channels to the target size using bilinear interpolation.
        
        Args:
            chroma (torch.Tensor): Tensor of shape (B, C, H_down, W_down) representing downsampled chroma channels.
            target_size (tuple): The target (height, width) to upsample to.
        
        Returns:
            torch.Tensor: Upscaled chroma tensor of shape (B, C, target_height, target_width).
        """
        # Bilinear interpolation is used here; align_corners is set to False for typical usage.
        x = torch.unsqueeze(chroma,dim=0)
        return torch.squeeze(F.interpolate(x, size=target_size, mode='bicubic', align_corners=False),dim=0)
    
    def train_ycbcr(self, img, num_iters):
        """Fit neural net to image in YCbCr with channel-weighted loss.

        Args:
            img (torch.Tensor): Image in RGB (C,H,W) on device.
            num_iters (int): Number of iterations to train for.
        """
        # Convert image from RGB -> YCbCr
        target = kornia.color.rgb_to_ycbcr(img)

        # Convert the target image into (N, 3) coordinate/feature pairs
        coordinates, features = to_coordinates_and_features(target)
        coordinates, features = coordinates.to(device), features.to(device)

        with tqdm.trange(num_iters, ncols=100) as t:
            for i in t:
                # Forward pass
                self.optimizer.zero_grad()
                predicted = self.representation(coordinates)  # shape: (N, 3)

                # Split the predicted + target features by channel
                predicted_y  = predicted[:, 0]
                predicted_cb = predicted[:, 1]
                predicted_cr = predicted[:, 2]

                target_y  = features[:, 0]
                target_cb = features[:, 1]
                target_cr = features[:, 2]

                # Compute per-channel MSE losses
                loss_y  = F.mse_loss(predicted_y,  target_y)
                loss_cb = F.mse_loss(predicted_cb, target_cb)
                loss_cr = F.mse_loss(predicted_cr, target_cr)

                # Weights for each channel: heavier on Y, lighter on Cb/Cr
                w_y  = 1.0
                w_cb = 0.8
                w_cr = 0.8

                # Weighted sum of channel losses
                loss = w_y * loss_y + w_cb * loss_cb + w_cr * loss_cr

                # Backprop + step
                loss.backward()
                self.optimizer.step()

                # Convert predicted back to (C,H,W) for PSNR calculation
                img_recon_ycbcr = predicted.reshape(target.shape[1], target.shape[2], 3).permute(2, 0, 1)
                img_recon = kornia.color.ycbcr_to_rgb(img_recon_ycbcr)
                psnr = get_clamped_psnr(img_recon, img)

                # Logging
                log_dict = {
                    'loss': loss.item(),
                    'psnr': psnr,
                    'best_psnr': self.best_vals['psnr']
                }
                t.set_postfix(**log_dict)
                for key in ['loss', 'psnr']:
                    self.logs[key].append(log_dict[key])

                # Track best values
                if loss.item() < self.best_vals['loss']:
                    self.best_vals['loss'] = loss.item()
                if psnr > self.best_vals['psnr']:
                    self.best_vals['psnr'] = psnr
                    # If model achieves best PSNR seen during training, update weights
                    if i > int(num_iters / 2.):
                        for k, v in self.representation.state_dict().items():
                            self.best_model[k].copy_(v)
    
    def build_image(self,y_hat,cbcr_hat,img,chroma_sub_2):
        cbcr_hat = cbcr_hat.reshape(chroma_sub_2.shape[1], chroma_sub_2.shape[2], 2).permute(2, 0, 1)  # shape: (N, 2)
        upscaled_chroma = self.chroma_upsample(cbcr_hat,(img.shape[1], img.shape[2]))
        y_hat = y_hat.reshape(img.shape[1], img.shape[2], 1).permute(2, 0, 1)  # shape: (N, 1)
        luma_recon = torch.cat([y_hat,upscaled_chroma],dim=0)
        img_recon = kornia.color.ycbcr_to_rgb(luma_recon)
        return img_recon


    def get_coords_from_image(self,img):
        target = kornia.color.rgb_to_ycbcr(img)
        # print(f"target shape{target.shape}")
        target_y  = torch.unsqueeze(target[0,:,:],dim=0)
        # print(f"target_y  size{target_y.shape}")
        chroma = target[1:3, :, :]
        # print(f"chroma  size{chroma.shape}")
        chroma_sub_2 = torch.nn.AvgPool2d(kernel_size=2, stride=2)(chroma)
        # print(f"chroma sub 2 size{chroma_sub_2.shape}")
        
        coordinatesY, featuresY = to_coordinates_and_features(target)
        coordinatesUV, featuresUV = to_coordinates_and_features(chroma_sub_2)

        coordinatesY, featuresY = coordinatesY.to(device), featuresY.to(device)
        coordinatesUV, featuresUV = coordinatesUV.to(device), featuresUV.to(device)
        return coordinatesY, coordinatesUV, chroma_sub_2,featuresY,featuresUV,chroma,target_y

    def train_ycbcr_with_exploit(self, img, num_iters):
            """Fit neural net to image in YCbCr with channel-weighted loss.

            Args:
                img (torch.Tensor): Image in RGB (C,H,W) on device.
                num_iters (int): Number of iterations to train for.
            """
            
            coordinatesY, coordinatesUV, chroma_sub_2,featuresY,featuresUV,chroma,target_y = self.get_coords_from_image(img)
            ssim_module = StructuralSimilarityIndexMeasure(data_range=1.0)


            if 'grad_norm' not in self.logs:
                self.logs['grad_norm'] = []
            self.logs['ssim'] = []
            self.logs['iter'] = []
            target_grad_norm = 0.3
            with tqdm.trange(num_iters, ncols=100) as t:
                for i in t:
                    self.logs['iter'].append(i)
                    self.optimizer.zero_grad()
                    y_hat,cbcr_hat = self.representation(coordinatesY,coordinatesUV)
                    cbcr_hat = cbcr_hat.reshape(chroma_sub_2.shape[1], chroma_sub_2.shape[2], 2).permute(2, 0, 1)  # shape: (N, 2)
                    upscaled_chroma = self.chroma_upsample(cbcr_hat,(img.shape[1], img.shape[2]))
                    y_hat = y_hat.reshape(img.shape[1], img.shape[2], 1).permute(2, 0, 1)
                  
                    luma_recon = torch.cat([y_hat,upscaled_chroma],dim=0)
                    img_recon = kornia.color.ycbcr_to_rgb(luma_recon)
                   
                    # loss = self.loss_func(y_hat,target_y) +self.loss_func(upscaled_chroma,chroma) 
                    # loss.backward()
                    # self.optimizer.step()
                    
                    reconstruction_loss = self.loss_func(y_hat,target_y) +self.loss_func(upscaled_chroma,chroma)
                    # reconstruction_loss.backward(retain_graph=True)

                    # total_grad_norm = 0.0
                    # for param in self.representation.parameters():
                    #     if param.grad is not None:
                    #         total_grad_norm += param.grad.data.norm(2).item() ** 2
                    # total_grad_norm = total_grad_norm ** 0.5

                    # grad_loss = (total_grad_norm - target_grad_norm) ** 2
                    loss = reconstruction_loss 
                    loss.backward()
                    self.optimizer.step()

                    psnr = get_clamped_psnr(img_recon, img)
                    with torch.no_grad():
                        ssim = ssim_module(torch.unsqueeze(img_recon,dim=0), torch.unsqueeze(img,dim=0)).item()



                    # Logging
                    log_dict = {
                        'loss': loss.item(),
                        'psnr': psnr,
                        'ssim':ssim,
                        # 'grad_norm': total_grad_norm,
                        'best_psnr': self.best_vals['psnr']
                    }

                    t.set_postfix(**log_dict)
                    for key in ['loss', 'psnr','ssim']:
                        self.logs[key].append(log_dict[key])

                    # Track best values
                    if loss.item() < self.best_vals['loss']:
                        self.best_vals['loss'] = loss.item()
                    if psnr > self.best_vals['psnr']:
                        self.best_vals['psnr'] = psnr
                        # If model achieves best PSNR seen during training, update weights
                        if i > int(num_iters / 2.):
                            for k, v in self.representation.state_dict().items():
                                self.best_model[k].copy_(v)
            # grad_norms = self.logs['grad_norm']  # List of gradient norms
            # psnrs = self.logs['psnr']            # List of PSNR values

            # plt.figure(figsize=(8, 6))
            # # You can create a scatter plot, a line plot, or both. Here’s an example using a scatter plot:
            # plt.scatter(grad_norms, psnrs, alpha=0.6, marker='o')
            # plt.xlabel('Gradient Norm')
            # plt.ylabel('PSNR (dB)')
            # plt.title('PSNR vs. Gradient Norm During Training')
            # plt.grid(True)
            # plt.savefig("luma_exploit/grad_norm_vs_PSNR") 
            # iterations = self.logs.get('iter', list(range(len(self.logs['psnr']))))

            # plt.figure(figsize=(12, 5))

            # # PSNR vs. Iteration
            # plt.subplot(1, 2, 1)
            # plt.plot(iterations, self.logs['psnr'], label='PSNR', color='tab:blue')
            # plt.xlabel('Iteration')
            # plt.ylabel('PSNR (dB)')
            # plt.title('PSNR over Iterations')
            # plt.grid(True)
            # plt.legend()

            # # Gradient Norm vs. Iteration
            # plt.subplot(1, 2, 2)
            # plt.plot(iterations, self.logs['grad_norm'], label='Gradient Norm', color='tab:orange')
            # plt.xlabel('Iteration')
            # plt.ylabel('Gradient Norm')
            # plt.title('Gradient Norm over Iterations')
            # plt.grid(True)
            # plt.legend()

            # plt.tight_layout()
            # plt.savefig("luma_exploit/norm_and_psnr_in_time.png") 
    def train_in_fft_and_rgb(self, img,coordinates, features, num_iters):
        """Fit neural net to image.

        Args:
            coordinates (torch.Tensor): Tensor of coordinates.
                Shape (num_points, coordinate_dim).
            features (torch.Tensor): Tensor of features. Shape (num_points, feature_dim).
            num_iters (int): Number of iterations to train for.
        """
        img_fft = fft.fft2(img, norm='ortho')
        def fft_loss(img_fft, recon_fft):
            # Compute the loss directly in frequency domain (complex difference)
            return F.mse_loss(torch.abs(img_fft), torch.abs(recon_fft))
        with tqdm.trange(num_iters, ncols=100) as t:
            for i in t:
                # Update model
                self.optimizer.zero_grad()
                predicted = self.representation(coordinates)
                img_recon = predicted.reshape(img.shape[1], img.shape[2], 3).permute(2, 0, 1)
                recon_fft = fft.fft2(img_recon, norm='ortho')
                loss = self.loss_func(predicted, features) + 0.08*fft_loss(img_fft,recon_fft)
                loss.backward()
                self.optimizer.step()

                # Calculate psnr
                psnr = get_clamped_psnr(predicted, features)

                # Print results and update logs
                log_dict = {'loss': loss.item(),
                            'psnr': psnr,
                            'best_psnr': self.best_vals['psnr']}
                t.set_postfix(**log_dict)
                for key in ['loss', 'psnr']:
                    self.logs[key].append(log_dict[key])

                # Update best values
                if loss.item() < self.best_vals['loss']:
                    self.best_vals['loss'] = loss.item()
                if psnr > self.best_vals['psnr']:
                    self.best_vals['psnr'] = psnr
                    # If model achieves best PSNR seen during training, update
                    # model
                    if i > int(num_iters / 2.):
                        for k, v in self.representation.state_dict().items():
                            self.best_model[k].copy_(v)

    def train(self, coordinates, features, num_iters):
        with tqdm.trange(num_iters, ncols=100) as t:
                for i in t:
                    # Update model
                    self.optimizer.zero_grad()
                    predicted = self.representation(coordinates)
                    loss = self.loss_func(predicted, features)
                    loss.backward()
                    self.optimizer.step()

                    # Calculate psnr
                    psnr = get_clamped_psnr(predicted, features)

                    # Print results and update logs
                    log_dict = {'loss': loss.item(),
                                'psnr': psnr,
                                'best_psnr': self.best_vals['psnr']}
                    t.set_postfix(**log_dict)
                    for key in ['loss', 'psnr']:
                        self.logs[key].append(log_dict[key])

                    # Update best values
                    if loss.item() < self.best_vals['loss']:
                        self.best_vals['loss'] = loss.item()
                    if psnr > self.best_vals['psnr']:
                        self.best_vals['psnr'] = psnr
                        # If model achieves best PSNR seen during training, update
                        # model
                        if i > int(num_iters / 2.):
                            for k, v in self.representation.state_dict().items():
                                self.best_model[k].copy_(v)

    def train_with_scale(self, coordinates, features, total_iterations,img):
        """Fit neural net to image.

        Args:
            coordinates (torch.Tensor): Tensor of coordinates.
                Shape (num_points, coordinate_dim).
            features (torch.Tensor): Tensor of features. Shape (num_points, feature_dim).
            num_iters (int): Number of iterations to train for.
        """
        training_scales = [0.25,0.5,0.75]
        def task_builder(x,s):
            img_4d = torch.unsqueeze(x,dim=0)
            lr_img = F.interpolate(input=img_4d,scale_factor=s,mode='bicubic',antialias=True)
            lr_img = torch.squeeze(lr_img)
            normed_scale = normalize_log2_scale(s)
            coords_LR, rgb_LR = to_coordinates_and_features(lr_img)  # Downsampled LR view
            normed_scale_as_feature = normed_scale.expand(coords_LR.shape[0],1)
            normed_scale_as_feature = normed_scale_as_feature.to(device)
            coords_LR = torch.cat([coords_LR, normed_scale_as_feature], dim=-1)
            return coords_LR, rgb_LR
        
        for scale in training_scales:
            self.best_vals = {'psnr': 0.0, 'loss': 1e8}
            coords, rgb = task_builder(img,scale)
            num_iters = total_iterations // len(training_scales)

            with tqdm.trange(num_iters,desc=f"training for scale: {scale}", ncols=100) as t:
                for i in t:
                    # Update model
                    self.optimizer.zero_grad()
                    predicted = self.representation(coordinates)
                    loss = self.loss_func(predicted, features)
                    loss.backward()
                    self.optimizer.step()

                    # Calculate psnr
                    psnr = get_clamped_psnr(predicted, features)

                    # Print results and update logs
                    log_dict = {'loss': loss.item(),
                                'psnr': psnr,
                                'best_psnr': self.best_vals['psnr']}
                    t.set_postfix(**log_dict)
                    for key in ['loss', 'psnr']:
                        self.logs[key].append(log_dict[key])

                    # Update best values
                    if loss.item() < self.best_vals['loss']:
                        self.best_vals['loss'] = loss.item()
                    if psnr > self.best_vals['psnr']:
                        self.best_vals['psnr'] = psnr
                        # If model achieves best PSNR seen during training, update
                        # model
                        # if i > int(num_iters / 2.):
                        for k, v in self.representation.state_dict().items():
                            self.best_model[k].copy_(v)

    def train_in_dct(self,img, coordinates, features, num_iters):
        target = dct.dct_2d(img) 
        # print(f"target shape {target.shape}")
        # return
        K = 70
        if 'grad_norm' not in self.logs:
            self.logs['grad_norm'] = []

        with tqdm.trange(num_iters, ncols=100) as t:
            for i in t:
                # Update model
                self.optimizer.zero_grad()
                predicted = self.representation(coordinates)
                img_recon = predicted.reshape(img.shape[1], img.shape[2], 3).permute(2, 0, 1)
                img_recon = dct.dct_2d(img_recon) 
                coeff_rec  = img_recon[ :, :K, :K]
                coeff_tgt  = target[   :, :K, :K]
                recon_loss = self.loss_func(coeff_rec,coeff_tgt) + 0.1 * F.l1_loss(coeff_rec,coeff_tgt)
                recon_loss.backward(retain_graph=True)
                total_grad_norm = 0.0
                for param in self.representation.parameters():
                    if param.grad is not None:
                        total_grad_norm += param.grad.data.norm(2).item() ** 2
                total_grad_norm = total_grad_norm ** 0.5


                loss = recon_loss
                loss.backward()
                self.optimizer.step()

                # Calculate psnr
                psnr = get_clamped_psnr(predicted, features)

                # Print results and update logs
                log_dict = {'loss': loss.item(),
                            'psnr': psnr,
                            'grad_norm':total_grad_norm,
                            'best_psnr': self.best_vals['psnr']}
                t.set_postfix(**log_dict)
                for key in ['loss', 'psnr','grad_norm']:
                    self.logs[key].append(log_dict[key])

                # Update best values
                if loss.item() < self.best_vals['loss']:
                    self.best_vals['loss'] = loss.item()
                if psnr > self.best_vals['psnr']:
                    self.best_vals['psnr'] = psnr
                    # If model achieves best PSNR seen during training, update
                    # model
                    if i > int(num_iters / 2.):
                        for k, v in self.representation.state_dict().items():
                            self.best_model[k].copy_(v)
            grad_norms = self.logs['grad_norm']  # List of gradient norms
            psnrs = self.logs['psnr']            # List of PSNR values
  
            plt.figure(figsize=(8, 6))
            # You can create a scatter plot, a line plot, or both. Here’s an example using a scatter plot:
            plt.scatter(grad_norms, psnrs, alpha=0.6, marker='o')
            plt.xlabel('Gradient Norm')
            plt.ylabel('PSNR (dB)')
            plt.title('PSNR vs. Gradient Norm During Training')
            plt.grid(True)
            plt.savefig("dct2_latent/latent_loss__grad_norm_vs_PSNR") 
            iterations = self.logs.get('iter', list(range(len(self.logs['psnr']))))

            plt.figure(figsize=(12, 5))

            # PSNR vs. Iteration
            plt.subplot(1, 2, 1)
            plt.plot(iterations, self.logs['psnr'], label='PSNR', color='tab:blue')
            plt.xlabel('Iteration')
            plt.ylabel('PSNR (dB)')
            plt.title('PSNR over Iterations')
            plt.grid(True)
            plt.legend()

            # Gradient Norm vs. Iteration
            plt.subplot(1, 2, 2)
            plt.plot(iterations, self.logs['grad_norm'], label='Gradient Norm', color='tab:orange')
            plt.xlabel('Iteration')
            plt.ylabel('Gradient Norm')
            plt.title('Gradient Norm over Iterations')
            plt.grid(True)
            plt.legend()

            plt.tight_layout()
            plt.savefig("dct2_latent/latent_loss_norm_and_psnr_in_time.png") 
    
    def degrade_image(self,image, t, max_sigma=2.0):
        """
        Degrade the image based on t.
        t: a scalar between 0 and 1. 0 gives very blurry image, 1 gives original.
        """
        sigma = (1 - t) * max_sigma  # More blur when t is small.
        # For simplicity, we'll use torchvision's GaussianBlur.
        # Kernel size is chosen as a function of sigma.
        if sigma > 0:
            # Kernel size must be odd.
            kernel_size = int(2 * round(3 * sigma) + 1)
            blur = T.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
            degraded = blur(image)
        else:
            degraded = image
        return degraded
    def time_dependent_train(self, img, num_iters):
        
        time_steps = [1, 2,3]
        degraded_images = []
        for ts in time_steps:
             degraded_images.append(self.degrade_image(img,ts/len(time_steps)))
             
        with tqdm.trange(num_iters, ncols=100) as t:
            # coords, features = to_coordinates_and_features(img)
            # coords, features = coords.to(device), features.to(device)

            for i in t:
                self.optimizer.zero_grad()
                final_prediction = torch.zeros_like(img.reshape(img.shape[0], -1).T,requires_grad=False)
                for ts in time_steps:
                    coords, features = to_coordinates_and_features(degraded_images[ts-1])
                    coords, features = coords.to(device), features.to(device)

                    predicted = self.representation(coords,float(ts)/len(time_steps)) 
                    final_prediction += predicted 

                loss = self.loss_func(final_prediction,features)
                loss.backward()  
                self.optimizer.step()
                    # Calculate psnr
                psnr = get_clamped_psnr(final_prediction, features)

                    # Print results and update logs
                log_dict = {'loss': loss.item(),
                                'psnr': psnr,
                                'best_psnr': self.best_vals['psnr']}
                t.set_postfix(**log_dict)
                for key in ['loss', 'psnr']:
                        self.logs[key].append(log_dict[key])

                    # Update best values
                if loss.item() < self.best_vals['loss']:
                        self.best_vals['loss'] = loss.item()
                if psnr > self.best_vals['psnr']:
                        self.best_vals['psnr'] = psnr
                        # If model achieves best PSNR seen during training, update
                        # model
                        if i > int(num_iters / 2.):
                            for k, v in self.representation.state_dict().items():
                                self.best_model[k].copy_(v)
    
    def coarse_to_fine_grained_training(self,img,num_iters,num_iters_coarse,num_iters_intermediate):
        
         pool_coarse = torch.nn.AvgPool2d(kernel_size=8, stride=8)
         pool_coarse_max = torch.nn.MaxPool2d(kernel_size=8, stride=8)

         pool_intermediate = torch.nn.AvgPool2d(kernel_size=4, stride=4)
         pool_intermediate_max  = torch.nn.MaxPool2d(kernel_size=4, stride=4)


         img_coarse = pool_coarse(img)
         img_coarse_max  = pool_coarse_max(img)

         img_intermediate = pool_intermediate(img)

         img_intermediate_max = pool_intermediate_max(img)

         img_fine = img 
         num_iters_fine = num_iters - (num_iters_coarse + num_iters_intermediate)
         f = lambda r1,r2, alpha: alpha*r1 + (1-alpha)*r2 

        #  num_iters_coarse_avg = num_iters_coarse // 2 
         
         with tqdm.trange(num_iters, ncols=100) as t:
            for i in t:
                self.optimizer.zero_grad()
                if i < num_iters_coarse:
                         target = img_coarse
                elif i < num_iters_coarse + num_iters_intermediate:
                         target = img_intermediate
                         if abs(i - num_iters_coarse) < 2 :
                             self.representation.addLayer(3)
                else: 
                         if abs(i - num_iters_coarse + num_iters_intermediate) < 2:
                             self.representation.addLayer(4)
                         target = img_fine
                coordinates, features = to_coordinates_and_features(img=target)
                coordinates, features = coordinates.to(device), features.to(device)
                predicted = self.representation(coordinates)
                img_recon = predicted.reshape(target.shape[1], target.shape[2], 3).permute(2, 0, 1)
                loss = self.loss_func(img_recon, target)
                loss.backward()
                self.optimizer.step()

                if i > num_iters_intermediate+num_iters_coarse:
                    psnr = get_clamped_psnr(predicted, features)

                    # Print results and update logs
                    log_dict = {'loss': loss.item(),
                                    'psnr': psnr,
                                    'best_psnr': self.best_vals['psnr']}
                    t.set_postfix(**log_dict)
                    for key in ['loss', 'psnr']:
                        self.logs[key].append(log_dict[key])

                        # Update best values
                        if loss.item() < self.best_vals['loss']:
                            self.best_vals['loss'] = loss.item()
                        if psnr > self.best_vals['psnr']:
                            self.best_vals['psnr'] = psnr
                            # If model achieves best PSNR seen during training, update
                            # model
                            if i > int(num_iters_intermediate+num_iters_coarse):
                                for k, v in self.representation.state_dict().items():
                                    self.best_model[k].copy_(v)
       
    def frequency_space(self,img,num_iters,k):
         frequency_domain = apply_dct2(img_tensor=img)
         test_recon = torch.zeros_like(img)
         test_recon = test_recon.to(device)
         target = frequency_domain[:,:k,:k]
         target = target.to(device)
         print("mean value ",torch.mean(target))

         test_recon[:,:k,:k] = target
         test_img_recon = apply_idct2(test_recon)
         psnr = get_clamped_psnr(test_img_recon, img)
         print(f"staring psnr{psnr}")

         coordinates,features = to_coordinates_and_features(img=target)
         coordinates,features = coordinates.to(device),features.to(device)
         
         with tqdm.trange(num_iters, ncols=100) as t:
              for i in t:
                self.optimizer.zero_grad()
                predicted = self.representation(coordinates)
                # predicted=predicted.reshape(target.shape[1], target.shape[2], 3).permute(2, 0, 1)
                loss = self.loss_func(predicted,features)
                loss.backward()
                self.optimizer.step()

                predicted=predicted.reshape(target.shape[1], target.shape[2], 3).permute(2, 0, 1)
                recon_frequency_space = torch.zeros_like(img)
                recon_frequency_space = recon_frequency_space.to(device)

                recon_frequency_space[:,:k,:k] = predicted

                img_recon = apply_idct2(recon_frequency_space)
                psnr = get_clamped_psnr(img_recon, img)
                log_dict = {'loss': loss.item(),
                            'psnr': psnr,
                            'best_psnr': self.best_vals['psnr']}
                t.set_postfix(**log_dict)
                for key in ['loss', 'psnr']:
                    self.logs[key].append(log_dict[key])

                # Update best values
                if loss.item() < self.best_vals['loss']:
                    self.best_vals['loss'] = loss.item()
                if psnr > self.best_vals['psnr']:
                    self.best_vals['psnr'] = psnr
                    # If model achieves best PSNR seen during training, update
                    # model
                    if i > int(num_iters / 2.):
                        for key, v in self.representation.state_dict().items():
                            self.best_model[key].copy_(v)



    def dct2_risidual(self,img,num_iters,k,coordinates):
         frequency_domain = apply_dct2(img_tensor=img)
         mask = torch.zeros_like(img)
         mask[:k,:k] = 1 
         complement_mask = -1*mask + torch.ones_like(img) 
         frequency_domain, mask,complement_mask = frequency_domain.to(device), mask.to(device),complement_mask.to(device)
         base_frequencies = frequency_domain*mask
         unmasked_frequencies = frequency_domain*complement_mask
         with tqdm.trange(num_iters, ncols=100) as t:
              for i in t: 
                self.optimizer.zero_grad()
                predicted = self.representation(coordinates)
                risidual_recon = predicted.reshape(img.shape[1], img.shape[2], 3).permute(2, 0, 1)
                
                dct_recon = apply_idct2(base_frequencies)
                target = apply_idct2(unmasked_frequencies)

                img_recon= risidual_recon+dct_recon
                loss = self.loss_func(risidual_recon, target)
                # loss =  torch.nn.L1Loss()(risidual_recon, target)*100
                loss.backward()
                self.optimizer.step()

                # Calculate psnr
                psnr = get_clamped_psnr(img_recon, img)

                # Print results and update logs
                log_dict = {'loss': loss.item(),
                            'psnr': psnr,
                            'best_psnr': self.best_vals['psnr']}
                t.set_postfix(**log_dict)
                for key in ['loss', 'psnr']:
                    self.logs[key].append(log_dict[key])

                # Update best values
                if loss.item() < self.best_vals['loss']:
                    self.best_vals['loss'] = loss.item()
                if psnr > self.best_vals['psnr']:
                    self.best_vals['psnr'] = psnr
                    # If model achieves best PSNR seen during training, update
                    # model
                    if i > int(num_iters / 2.):
                        for k, v in self.representation.state_dict().items():
                            self.best_model[k].copy_(v)
    
    def coarse_to_fine_grainded_in_frequency_space(self,img,num_iters,num_iters_coarse,num_iters_intermediate):
         coordinates,features = torch.tensor(0),torch.tensor(0)
         def get_coords_and_features(img,k):
            frequency_domain = apply_dct2(img_tensor=img)
            mask = torch.zeros_like(img)
            mask[:,:k,:k] = 1 
            base_frequencies = frequency_domain*mask
            target = apply_idct2(base_frequencies)
            target = target.to(device)
            coordinates,features = to_coordinates_and_features(img=target)
            return coordinates,features
         

         with tqdm.trange(num_iters, ncols=100) as t:
              for i in t: 
                self.optimizer.zero_grad()

                if i < num_iters_coarse: 
                    coordinates,features= get_coords_and_features(img,30)
                    coordinates,features = coordinates.to(device),features.to(device)
                if i < num_iters_coarse + num_iters_intermediate: 
                    coordinates,features= get_coords_and_features(img,60)
                    coordinates,features = coordinates.to(device),features.to(device)
                else: 
                    coordinates,features = to_coordinates_and_features(img=img)
                    coordinates,features = coordinates.to(device),features.to(device)

                predicted = self.representation(coordinates)
                img_recon = predicted.reshape(img.shape[1], img.shape[2], 3).permute(2, 0, 1)
                loss = self.loss_func(img_recon, img)
                loss.backward()
                self.optimizer.step()
                if i > num_iters_intermediate+num_iters_coarse:
                    psnr = get_clamped_psnr(predicted, features)

                    # Print results and update logs
                    log_dict = {'loss': loss.item(),
                                    'psnr': psnr,
                                    'best_psnr': self.best_vals['psnr']}
                    t.set_postfix(**log_dict)
                    for key in ['loss', 'psnr']:
                        self.logs[key].append(log_dict[key])

                        # Update best values
                        if loss.item() < self.best_vals['loss']:
                            self.best_vals['loss'] = loss.item()
                        if psnr > self.best_vals['psnr']:
                            self.best_vals['psnr'] = psnr
                            # If model achieves best PSNR seen during training, update
                            # model
                            if i > int(num_iters_intermediate+num_iters_coarse):
                                for k, v in self.representation.state_dict().items():
                                    self.best_model[k].copy_(v)
    def train_with_laplacian_pyramid(self,img, coordinates, features, num_iters):
        """Fit neural net to image.

        Args:
            coordinates (torch.Tensor): Tensor of coordinates.
                Shape (num_points, coordinate_dim).
            features (torch.Tensor): Tensor of features. Shape (num_points, feature_dim).
            num_iters (int): Number of iterations to train for.
        """
        if 'grad_norm' not in self.logs:
            self.logs['grad_norm'] = []
        
        gt_lap, gt_gauss = LaplacianPyramid()(torch.unsqueeze(img,dim=0))
        init_weights = [0.15, 0.10, 0.25, 0.50]  # [L0, L1, L2, G3]
        def get_weights(epoch, max_epoch):
            # linearly blend from init to uniform
            alpha = epoch / max_epoch
            uniform = [1/4]*4
            return [(1-alpha)*w0 + alpha*wu for w0, wu in zip(init_weights, uniform)]

        def lossfn(gt_pyr,out_pyr,iter,num_iters):
            weight = get_weights(iter,num_iters)
            loss =0 
            for lvl in range(len(gt_pyr)):
                loss += weight[lvl] * F.mse_loss(out_pyr[lvl], gt_pyr[lvl])
            return loss


        with tqdm.trange(num_iters, ncols=100) as t:
            for i in t:
                # Update model
                self.optimizer.zero_grad()
                predicted = self.representation(coordinates)
                img_recon = predicted.reshape(img.shape[1], img.shape[2], 3).permute(2, 0, 1)
                pred_lap, pred_gauss = LaplacianPyramid()(torch.unsqueeze(img_recon,dim=0))

                reconstruction_loss = lossfn(gt_lap,pred_lap,i,num_iters)
                reconstruction_loss.backward(retain_graph=True)
                total_grad_norm = 0.0
                for param in self.representation.parameters():
                    if param.grad is not None:
                        total_grad_norm += param.grad.data.norm(2).item() ** 2
                total_grad_norm = total_grad_norm ** 0.5
                self.logs['grad_norm'].append(total_grad_norm)
                loss = reconstruction_loss
                loss.backward()
                self.optimizer.step()

                # Calculate psnr
                psnr = get_clamped_psnr(predicted, features)

                # Print results and update logs
                log_dict = {'loss': loss.item(),
                            'psnr': psnr,
                            'best_psnr': self.best_vals['psnr']}
                t.set_postfix(**log_dict)
                for key in ['loss', 'psnr']:
                    self.logs[key].append(log_dict[key])

                # Update best values
                if loss.item() < self.best_vals['loss']:
                    self.best_vals['loss'] = loss.item()
                if psnr > self.best_vals['psnr']:
                    self.best_vals['psnr'] = psnr
                    # If model achieves best PSNR seen during training, update
                    # model
                    if i > int(num_iters / 2.):
                        for k, v in self.representation.state_dict().items():
                            self.best_model[k].copy_(v)

            grad_norms = self.logs['grad_norm']  # List of gradient norms
            psnrs = self.logs['psnr']            # List of PSNR values

            plt.figure(figsize=(8, 6))
            # You can create a scatter plot, a line plot, or both. Here’s an example using a scatter plot:
            plt.scatter(grad_norms, psnrs, alpha=0.6, marker='o')
            plt.xlabel('Gradient Norm')
            plt.ylabel('PSNR (dB)')
            plt.title('PSNR vs. Gradient Norm During Training')
            plt.grid(True)
            plt.savefig("laplacian_pyramid/grad_norm_vs_PSNR") 
            iterations = self.logs.get('iter', list(range(len(self.logs['psnr']))))

            plt.figure(figsize=(12, 5))

            # PSNR vs. Iteration
            plt.subplot(1, 2, 1)
            plt.plot(iterations, self.logs['psnr'], label='PSNR', color='tab:blue')
            plt.xlabel('Iteration')
            plt.ylabel('PSNR (dB)')
            plt.title('PSNR over Iterations')
            plt.grid(True)
            plt.legend()

            # Gradient Norm vs. Iteration
            plt.subplot(1, 2, 2)
            plt.plot(iterations, self.logs['grad_norm'], label='Gradient Norm', color='tab:orange')
            plt.xlabel('Iteration')
            plt.ylabel('Gradient Norm')
            plt.title('Gradient Norm over Iterations')
            plt.grid(True)
            plt.legend()

            plt.tight_layout()
            plt.savefig("laplacian_pyramid/norm_and_psnr_in_time.png") 

    
    def train_with_correction_net(self, coordinates, features, num_iters,img,correction_cycles):
        """Fit neural net to image.

        Args:
            coordinates (torch.Tensor): Tensor of coordinates.
                Shape (num_points, coordinate_dim).
            features (torch.Tensor): Tensor of features. Shape (num_points, feature_dim).
            num_iters (int): Number of iterations to train for.
        """
        ssim_module = StructuralSimilarityIndexMeasure(data_range=1.0)
        pool_intermediate = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        half = pool_intermediate(img)
        coords_half , features_half = to_coordinates_and_features(img=half)
        coords_half , features_half = coords_half.to(device) , features_half.to(device)
        self.base_model_logs['ssim'] = []
        with tqdm.trange(num_iters, ncols=100) as t:
            for i in t:
                # Update model
                self.base_learner_optimizer.zero_grad()
               
                predicted = self.representation.base_learner(coords_half)
        
                reconstruction_loss = self.loss_func(predicted, features_half)
                loss = reconstruction_loss
                loss.backward()
                self.base_learner_optimizer.step()

                # Calculate psnr
                psnr = get_clamped_psnr(predicted, features_half)
                
                #calculate ssim
                gt = features_half.reshape(half.shape[1], half.shape[2], 3).permute(2, 0, 1)
                recon = predicted.reshape(half.shape[1], half.shape[2], 3).permute(2, 0, 1)
                with torch.no_grad():
                    ssim = ssim_module(torch.unsqueeze(recon,dim=0),torch.unsqueeze(gt,dim=0)).item()


                # Print results and update logs
                log_dict = {'loss': loss.item(),
                            'psnr': psnr,
                            'ssim': ssim,
                            'best_psnr': self.base_model_best_vals['psnr']}
                t.set_postfix(**log_dict)
                for key in ['loss', 'psnr','ssim']:
                    self.base_model_logs[key].append(log_dict[key])

                # Update best values
                if loss.item() < self.base_model_best_vals['loss']:
                    self.base_model_best_vals['loss'] = loss.item()
                if psnr > self.base_model_best_vals['psnr']:
                    self.base_model_best_vals['psnr'] = psnr
                    # If model achieves best PSNR seen during training, update
                    # model
                    if i > int(num_iters / 2.):
                        for k, v in self.representation.base_learner.state_dict().items():
                            self.best_base_learner[k].copy_(v)
            


        # set the best base learner
        self.representation.base_learner.load_state_dict(self.best_base_learner)
        print(f"best base learner PSNR achieved:{self.base_model_best_vals['psnr']}")

        self.logs['ssim'] = []

        coordinates, features = to_coordinates_and_features(img=img)
        coordinates, features = coordinates.to(device), features.to(device)

        with tqdm.trange(correction_cycles, ncols=100) as t:
            for i in t:
                self.correction_optimizer.zero_grad()
                predicted = self.representation(coordinates,img)
                reconstruction_loss = torch.nn.L1Loss()(predicted, img)
            

                loss = reconstruction_loss
                loss.backward()
                self.correction_optimizer.step()

                # Calculate psnr
                psnr = get_clamped_psnr(predicted, img)
                
                #calculate ssim
                # gt = features.reshape(img.shape[1], img.shape[2], 3).permute(2, 0, 1)
                # recon = predicted.reshape(img.shape[1], img.shape[2], 3).permute(2, 0, 1)
                with torch.no_grad():
                    ssim = ssim_module(torch.unsqueeze(predicted,dim=0),torch.unsqueeze(img,dim=0)).item()


                # Print results and update logs
                log_dict = {'loss': loss.item(),
                            'psnr': psnr,
                            'ssim': ssim,
                            'best_psnr': self.best_vals['psnr']}
                t.set_postfix(**log_dict)
                for key in ['loss', 'psnr','ssim']:
                    self.logs[key].append(log_dict[key])

                # Update best values
                if loss.item() < self.best_vals['loss']:
                    self.best_vals['loss'] = loss.item()
                if psnr > self.best_vals['psnr']:
                    self.best_vals['psnr'] = psnr
                    # If model achieves best PSNR seen during training, update
                    # model
                    if i > int(correction_cycles / 2.):
                        for k, v in self.representation.state_dict().items():
                            self.best_model[k].copy_(v)

    def train_with_maml(self,img,num_iters_downsampled,num_iters_normal):
         pool_intermediate = torch.nn.AvgPool2d(kernel_size=8, stride=8)
         img_intermediate = pool_intermediate(img)
         coordinates, features = to_coordinates_and_features(img=img_intermediate)
         coordinates, features = coordinates.to(device), features.to(device)

         coordinates_og, features_og = to_coordinates_and_features(img=img)
         coordinates_og, features_og = coordinates_og.to(device), features_og.to(device)
         with tqdm.trange(num_iters_downsampled, ncols=100) as t:
             for i in t:
                 self.optimizer.zero_grad()
                 predicted = self.representation(coordinates)
                 loss = self.loss_func(predicted, features)
                 loss.backward()
                 self.optimizer.step()
                 psnr = get_clamped_psnr(predicted, features)
                # Print results and update logs
                 log_dict = {'loss': loss.item(),
                             'psnr': psnr,
                             'best_psnr': self.best_vals['psnr']}
                 t.set_postfix(**log_dict)
                 for key in ['loss', 'psnr']:
                     self.logs[key].append(log_dict[key])
 
                 # Update best values
                 if loss.item() < self.best_vals['loss']:
                     self.best_vals['loss'] = loss.item()
                 if psnr > self.best_vals['psnr']:
                     self.best_vals['psnr'] = psnr
                     # If model achieves best PSNR seen during training, update
                     # model
                     if i > int(num_iters_downsampled / 2.):
                         for k, v in self.representation.state_dict().items():
                             self.best_model[k].copy_(v)
         inner_optimizer = torch.optim.Adam(self.representation.parameters(), lr=0.00001)        
         with higher.innerloop_ctx(self.representation, inner_optimizer, copy_initial_weights=True) as (fmodel, diffopt):
            for i in range(30):
             support_preds = fmodel(coordinates_og)
             support_loss = self.loss_func(support_preds, features_og)
             print(f"inner loop step: {i} step loss: {support_loss.item()}\n")
             diffopt.step(support_loss)

         query_preds = fmodel(coordinates_og)
         query_loss = self.loss_func(query_preds, features_og)
         print(f"meta loss: {query_loss.item()}")
         
         # Compute meta-gradient and update original model
         self.optimizer.zero_grad()
         query_loss.backward()
         self.optimizer.step()
         
         # === Phase 3: Final finetuning ===
         with tqdm.trange(num_iters_normal, ncols=100) as t:
             for i in t:
                 self.optimizer.zero_grad()
                 pred = self.representation(coordinates_og)
                 loss = self.loss_func(pred, features_og)
                 loss.backward()
                 self.optimizer.step()
         
                 psnr = get_clamped_psnr(pred, features_og)
                 log_dict = {'loss': loss.item(),
                             'psnr': psnr,
                             'best_psnr': self.best_vals['psnr']}
                 t.set_postfix(**log_dict)
                 for key in ['loss', 'psnr']:
                     self.logs[key].append(log_dict[key])
         
                 if loss.item() < self.best_vals['loss']:
                     self.best_vals['loss'] = loss.item()
                 if psnr > self.best_vals['psnr']:
                     self.best_vals['psnr'] = psnr
                     for k, v in self.representation.state_dict().items():
                         self.best_model[k].copy_(v)


from torch.nn.utils.stateless import functional_call
from collections import OrderedDict
def maml_inner_loop(model, coords_LR, rgb_LR, loss_fn, lr_inner, steps):
    # Initialize fast weights as a copy of model's current parameters
    fast_weights = OrderedDict((name, param.clone()) for name, param in model.named_parameters())

    for _ in range(steps):
        preds = functional_call(model, fast_weights, (coords_LR,))
        loss = loss_fn(preds, rgb_LR)

        # Compute gradients w.r.t. fast weights
        grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)

        # Update fast weights
        for (name, param), grad in zip(fast_weights.items(), grads):
            fast_weights[name] = param - lr_inner * grad

    return fast_weights

def normalize_log2_scale(scale, min_scale=0.25, max_scale=2.0):
    """
    Normalize log2(scale) to [-1, 1] given the range of scales.
    """
    log_scale = torch.log2(torch.tensor(scale))
    log_min = np.log2(min_scale)
    log_max = np.log2(max_scale)
    norm_log_scale = 2 * (log_scale - log_min) / (log_max - log_min) - 1
    return norm_log_scale

def maml_train(model, tasks, loss_fn, lr_inner, steps_inner, meta_lr, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=meta_lr)
    
    tasks_with_scale = []
    # scales = [0.5,0.75,1]
    scales = [1]
    for img in tasks:
        for scale in scales:
            tasks_with_scale.append((img,scale)) 

     
    
    best_model = OrderedDict((k, v.detach().clone()) for k, v in model.state_dict().items())
    best_psnr =0.0 

    def task_builder(x,s):
        img_4d = torch.unsqueeze(x,dim=0)
        lr_img = F.interpolate(input=img_4d,scale_factor=s,mode='bicubic',antialias=True)
        lr_img = torch.squeeze(lr_img)
        normed_scale = normalize_log2_scale(s)
        coords_LR, rgb_LR = to_coordinates_and_features(lr_img)  # Downsampled LR view
        normed_scale_as_feature = normed_scale.expand(coords_LR.shape[0],1)
        normed_scale_as_feature = normed_scale_as_feature.to(device)
        coords_LR = torch.cat([coords_LR, normed_scale_as_feature], dim=-1)
        return coords_LR, rgb_LR

    for epoch in range(epochs):
        meta_loss = 0.0
        psnrs = []

        for task_img,task_scale in tasks_with_scale:
            coords_LR, rgb_LR = task_builder(task_img,task_scale)
            coords_LR, rgb_LR = coords_LR.to(device), rgb_LR.to(device)

            # Inner loop adaptation on LR
            fast_weights = maml_inner_loop(model, coords_LR, rgb_LR, loss_fn, lr_inner, steps_inner)

            # Outer loop evaluation on HR
            preds = functional_call(model, fast_weights, (coords_LR,))
            loss = loss_fn(preds, rgb_LR)
            meta_loss += loss

            psnr = get_clamped_psnr(preds, rgb_LR)
            psnrs.append(psnr)

        # Average meta-loss and update base model
        meta_loss /= len(tasks)
        optimizer.zero_grad()
        meta_loss.backward()
        optimizer.step()

        meta_psnr = sum(psnrs)/len(psnrs)
        print(f"Epoch {epoch+1} | Meta Loss: {meta_loss.item():.4f} | PSNR: {meta_psnr:.2f} | BEST PSNR: {best_psnr:.2f}")
        if meta_psnr > best_psnr:
            best_psnr = meta_psnr
            for k, v in model.state_dict().items():
                best_model[k].copy_(v)

    return best_model

def train_from_starting_point(model,img,epochs,lr):
    pool2 = torch.nn.AvgPool2d(kernel_size=2, stride=2)
    lr_img = pool2(img)
    coords_LR, rgb_LR = to_coordinates_and_features(lr_img)
    lf = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        pred = model(coords_LR)
        loss = lf(pred,rgb_LR)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        psnr = get_clamped_psnr(pred, rgb_LR)
        print(f"Epoch {epoch+1} | Meta Loss: {loss.item():.4f} | LOW_RES_PSNR: {psnr:.2f}")

def maml_adaptation(model, img_hr,scale, loss_fn, lr_inner, inner_steps):
    img_4d = torch.unsqueeze(img_hr,dim=0)
    lr_img = F.interpolate(input=img_4d,scale_factor=scale,mode='bicubic',antialias=True)
    lr_img = torch.squeeze(lr_img)
    coords_LR, rgb_LR = to_coordinates_and_features(lr_img)
    scale = normalize_log2_scale(scale)
    scale = scale.expand(coords_LR.shape[0],1)
    coords_LR =  torch.cat([coords_LR,scale],dim=-1)
    

    fast_weights = OrderedDict((name, param.clone()) for name, param in model.named_parameters())
    for _ in range(inner_steps):
        preds_LR = functional_call(model, fast_weights, (coords_LR,))
        loss_inner = loss_fn(preds_LR, rgb_LR)
        grads = torch.autograd.grad(loss_inner, fast_weights.values(), create_graph=True)
        fast_weights = OrderedDict((
            name,
            (param - lr_inner * g)
              .detach()
              .requires_grad_(True)
        ) for ((name, param), g) in zip(fast_weights.items(), grads))
        torch.cuda.empty_cache()

    preds_HR = functional_call(model, fast_weights, (coords_LR,))
    return fast_weights, preds_HR













                   
    
                            

         
         

                         




              
              
              





