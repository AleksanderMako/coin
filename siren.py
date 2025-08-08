# Based on https://github.com/lucidrains/siren-pytorch
import torch
from torch import nn
from math import sqrt
import torch.nn.functional as F


class Sine(nn.Module):
    """Sine activation with scaling.

    Args:
        w0 (float): Omega_0 parameter from SIREN paper.
    """
    def __init__(self, w0=1.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class SirenLayer(nn.Module):
    """Implements a single SIREN layer.

    Args:
        dim_in (int): Dimension of input.
        dim_out (int): Dimension of output.
        w0 (float):
        c (float): c value from SIREN paper used for weight initialization.
        is_first (bool): Whether this is first layer of model.
        use_bias (bool):
        activation (torch.nn.Module): Activation function. If None, defaults to
            Sine activation.
    """
    def __init__(self, dim_in, dim_out, w0=30., c=6., is_first=False,
                 use_bias=True, activation=None):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        self.linear = nn.Linear(dim_in, dim_out, bias=use_bias)

        # Initialize layers following SIREN paper
        w_std = (1 / (2)) if self.is_first else (sqrt(c / dim_in) / w0)
        nn.init.uniform_(self.linear.weight, -w_std, w_std)
        if use_bias:
            nn.init.uniform_(self.linear.bias, -w_std, w_std)

        self.activation = Sine(w0) if activation is None else activation

    def forward(self, x):
        out = self.linear(x)
        out = self.activation(out)
        return out


class Siren(nn.Module):
    """SIREN model.

    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        dim_out (int): Dimension of output.
        num_layers (int): Number of layers.
        w0 (float): Omega 0 from SIREN paper.
        w0_initial (float): Omega 0 for first layer.
        use_bias (bool):
        final_activation (torch.nn.Module): Activation function.
    """
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, w0=30.,
                 w0_initial=30., use_bias=True, final_activation=None):
        super().__init__()
        layers = []
        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden
            self.dim_hidden = dim_hidden
            self.w0 = w0

            layers.append(SirenLayer(
                dim_in=layer_dim_in,
                dim_out=dim_hidden,
                w0=layer_w0,
                use_bias=False,
                is_first=is_first
            ))

        self.net = nn.Sequential(*layers)

        final_activation = nn.Identity() if final_activation is None else final_activation
        self.last_layer = SirenLayer(dim_in=dim_hidden, dim_out=dim_out, w0=w0,
                                use_bias=use_bias, activation=final_activation)
    def forward(self, x):
        x = self.net(x)
        return self.last_layer(x)

class FactoredSiren(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, w0=30.,
                 w0_initial=30., use_bias=True, final_activation=None):
        super().__init__()
        half_d_in = int(dim_hidden//2)
        quarter_d_in  = int(half_d_in //2 )
        half_n_layers = int(num_layers//2)
        quarter_n_layers = int(half_n_layers//2)

        self.green = Siren(dim_in,half_d_in+4,1,10,w0,w0_initial,use_bias,final_activation)
        self.red_green = Siren(dim_in+1,quarter_d_in+7,1,4,w0,w0_initial,use_bias,final_activation)
        self.blue_green = Siren(dim_in+1,quarter_d_in+7,1,4,w0,w0_initial,use_bias,final_activation)
    def forward(self,x):
        g = self.green(x)
        x = torch.cat([x,g],dim=-1)
        r_g = self.red_green(x)
        b_g = self.blue_green(x)
        r = g + r_g 
        b = g + b_g
        together =  torch.cat([r,g,b],dim= -1)
        # print("factored output: ",together.shape)
        # print("factored green: ",g.shape)
        # print("factored r: ",r_g.shape)
        # print("factored b: ",b_g.shape)
        return together
#python main_ycbcr_fast.py -ni 5000 -lss 28 -nl 10 -iid 3 -ld luma_exploit/ni5000_lss28_nl10_iid3
#python main.py -ni 5000 -lss 28 -nl 10 -iid 3 -ld luma_exploit/normal_procedure/ni5000_lss28_nl10_iid3
# python main.py -ni 5000 -lss 28 -nl 10 -iid 3 -ld laplacian_pyramid/ni5000_lss28_nl10_iid3
# python main.py -ni 5000 -lss 28 -nl 10 -iid 14 -ld dct_latent/ni5000_lss28_nl10_iid14
# python main.py -ni 15000 -lss 28 -nl 10 -iid 14 -ld coarse_to_fine_grained/ni15000_lss40_nl10_iid14
# python main.py -ni 1500 -lss 28 -nl 10 -iid 14 -ld maml_training/ni15000_lss28_nl10_iid14
#python main_maml.py -ni 1500 -lss 28 -nl 10 -iid 14 -ld maml_training/ni15000_lss28_nl10_iid14
class CoordNet(nn.Module):
    def __init__(self, in_features=2, hidden_dim=12, hidden_layers=3, out_features=2, positional_freq=2):
        super(CoordNet, self).__init__()

        self.positional_freq = positional_freq

        # Positional Encoding
        self.input_dim = in_features * positional_freq * 2

        layers = []
        layers.append(nn.Linear(self.input_dim, hidden_dim))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Linear(hidden_dim, out_features))  # Cb and Cr output

        self.network = nn.Sequential(*layers)

    def positional_encoding(self, coords):
        encodings = [coords]
        for i in range(self.positional_freq):
            freq = 2 ** i * torch.pi
            encodings.append(torch.sin(freq * coords))
            encodings.append(torch.cos(freq * coords))
        return torch.cat(encodings[1:], dim=-1)

    def forward(self, coords):
        coords_enc = self.positional_encoding(coords)
        output = self.network(coords_enc)
        return output

class CorrectorNet(nn.Module):
    def __init__(self, in_features=3, hidden_dim=12, hidden_layers=3, out_features=3):
        super(CorrectorNet, self).__init__()
        layers = []
        layers.append(nn.Linear(in_features, hidden_dim))
        layers.append(Sine(30.0))

        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(Sine(30.0))

        layers.append(nn.Linear(hidden_dim, out_features))

        self.network = nn.Sequential(*layers)
    def forward(self, x):
        output = self.network(x)
        return output

class CorrectorCNN(nn.Module):
    def __init__(self, num_layers) -> None:
        super(CorrectorCNN,self).__init__()
        layers = []
        layers.append(nn.Conv2d(3,10,3,1,padding='same'))
        layers.append(Sine(30.0))
        for _ in range( 3):
            layers.append(nn.Conv2d(10,10,3,1,padding='same'))
            layers.append(Sine(30.0))
        
        layers.append(nn.Conv2d(10,3,3,1,padding='same'))
        
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        output = self.network(x)
        return output



class SirenWithCorrectorNet(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, w0=30.,
                 w0_initial=30., use_bias=True, final_activation=None):
        super(SirenWithCorrectorNet,self).__init__()
        self.base_learner = Siren(dim_in, dim_hidden, dim_out, num_layers, w0=30.,
                 w0_initial=30., use_bias=True, final_activation=None)
        
        # self.corrector = Siren(dim_in, 14, dim_out, num_layers, w0=30.,
        #          w0_initial=30., use_bias=True, final_activation=None)
        # self.corrector = CorrectorNet(in_features=3, hidden_dim=19, hidden_layers=5, out_features=3)
        self.corrector = CorrectorCNN(10)
    
    def forward(self,coords,img):
        estimate = self.base_learner(coords)
        estimate = estimate.reshape(img.shape[1], img.shape[2], 3).permute(2, 0, 1)
        correction = self.corrector(estimate)
        return  estimate + correction



class YCbCrSiren(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, w0=30.,
                 w0_initial=30., use_bias=True, final_activation=None):
        super().__init__()
        self.luma = CoordNet(in_features=2, hidden_dim=20, hidden_layers=10, out_features=1, positional_freq=10)
        self.chroma = CoordNet(in_features=2, hidden_dim=12, hidden_layers=10, out_features=2, positional_freq=10)

    @torch.jit.export
    def forward(self, lumaX, cbcrX):
        y_fut = torch.jit.fork(self.luma, lumaX)
        cbcr = self.chroma(cbcrX)
        y = torch.jit.wait(y_fut)
        return y, cbcr
