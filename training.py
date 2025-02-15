import torch
import tqdm
from collections import OrderedDict
from util import get_clamped_psnr
from util import to_coordinates_and_features_and_time


class Trainer():
    def __init__(self, representation, lr=1e-3, print_freq=1):
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

    def train(self, coordinates, features, num_iters):
        """Fit neural net to image.

        Args:
            coordinates (torch.Tensor): Tensor of coordinates.
                Shape (num_points, coordinate_dim).
            features (torch.Tensor): Tensor of features. Shape (num_points, feature_dim).
            num_iters (int): Number of iterations to train for.
        """
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
    

def time_dependent_train(self, img, num_iters):
    total_steps = 3
    time_steps = [1, 2, 3]
    time_step_weights = [0.2, 0.3, 0.5]

    with tqdm.trange(num_iters, ncols=100) as t:
        for i in t:
            self.optimizer.zero_grad()
            total_loss = 0
            skips = [torch.zeros_like(img.reshape(img.shape[0], -1).T)]  # Initialize skips
            final_prediction = torch.zeros_like(img.reshape(img.shape[0], -1).T)

            for ts, weight in zip(time_steps, time_step_weights):
                coords, features = to_coordinates_and_features_and_time(img, ts, total_steps)
                predicted = self.representation(coords) + skips[ts - 1]
                skips.append(predicted.detach())  # Detach to prevent gradient flow through skips
                loss = self.loss_func(predicted, features) * weight
                total_loss += loss
                if ts == time_steps[len(time_steps) - 1]:
                    final_prediction = skips[len(skips) - 1]

            total_loss.backward()  # Single backward pass
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





