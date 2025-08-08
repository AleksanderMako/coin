import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import weights_collector

# -----------------------------------------------------------------------------
# Assumption:
# -----------
# You already have a list (or array) of PyTorch models in memory, all with
# the same architecture (i.e., same shapes for each linear layer). Let's call it:
#
#     models = [model_1, model_2, ..., model_N]
#
# where each `model_i` is an nn.Module containing only Linear layers (or at least
# that is what we want to extract).
# -----------------------------------------------------------------------------

# 1. Helper function to combine weight + bias for a single nn.Linear layer
#    as a 2D tensor by appending bias as the last column.
def combine_weight_bias_as_2d(linear_layer: nn.Linear):
    """
    Takes a Linear layer whose weight is of shape (out_features, in_features)
    and bias is of shape (out_features,).
    Concatenates the bias as the last column, resulting in:
      => (out_features, in_features + 1)
    """
    W = linear_layer.weight              # shape: (out_features, in_features)
    b = linear_layer.bias.unsqueeze(1)   # shape: (out_features, 1)
    return torch.cat([W, b], dim=1)      # shape: (out_features, in_features + 1)


# 2. Extract layer-by-layer from a single model, returning a list of 2D tensors
def extract_model_layer_tensors(model: nn.Module):
    """
    Iterates over *all submodules* of the given model. For each `nn.Linear`,
    returns the 2D weight+bias tensor. Returns a list of these.
    """
    layer_tensors = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Combine weight + bias into a single 2D tensor
            layer_tensors.append(combine_weight_bias_as_2d(module))
    return layer_tensors



# 3. Build the time-series dataset from the list of models
def build_time_series_data(models):
    """
    Given a list of models, each with the same architecture, extract their
    linear parameters in 2D form (weight+bias) for each layer.
    Then stack them into a final 4D tensor of shape:
       (num_models, num_layers, out_features, in_features + 1)
    """
    all_models_data = []
    for m in models:
        layer_tensor_list = extract_model_layer_tensors(m)
        # Stack across layers => shape: (num_layers, out_features, in_features+1)
        layer_stack = torch.stack(layer_tensor_list, dim=0)
        all_models_data.append(layer_stack)
    
    # Now stack across models => shape: (num_models, num_layers, out_features, in_features+1)
    time_series_data = torch.stack(all_models_data, dim=0)
    return time_series_data


# 4. (Optional) Define a Dataset to wrap the resulting time-series data
class ModelParamsTimeSeriesDataset(Dataset):
    """
    A simple Dataset that treats each model's parameters as one data point.
    """
    def __init__(self, data_tensor):
        """
        data_tensor shape: (num_models, num_layers, out_features, in_features+1)
        """
        self.data = data_tensor
    
    def __len__(self):
        return self.data.shape[0]  # number of models
    
    def __getitem__(self, idx):
        # Return the parameters for model 'idx'
        # shape: (num_layers, out_features, in_features+1)
        return self.data[idx]


# -----------------------------------------------------------------------------
# 5. Example usage:
# -----------------------------------------------------------------------------
#
# Suppose you already have your models array:
#     models = [model_1, model_2, ..., model_N]
#
# We build the time-series tensor:
#     time_series_data = build_time_series_data(models)
#     print("time_series_data shape:", time_series_data.shape)
#
# Then, if you want to use it as a PyTorch dataset:
#     dataset = ModelParamsTimeSeriesDataset(time_series_data)
#     dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
#
#     for batch in dataloader:
#         print("Batch shape:", batch.shape)
#         # shape: (batch_size, num_layers, out_features, in_features+1)
#         # do something with batch...
# -----------------------------------------------------------------------------

models = weights_collector.load_models("nl_3_lss_40_iid_fulldataset_ni_2000")
nets = weights_collector.get_models(models)

time_series_data = build_time_series_data(nets)
dataset = ModelParamsTimeSeriesDataset(time_series_data)

print(dataset[1].shape)