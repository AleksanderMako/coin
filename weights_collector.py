import os
import torch
import numpy as np
from siren import Siren
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_models(directory="cifar_10_full_dataset"):
    # Check if directory exists
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory '{directory}' not found")
    
    models_dict = {}
    
    # Iterate through 0 to 4000 (inclusive)
    for i in range(4001):
        filename = f"best_model_{i}.pt"
        file_path = os.path.join(directory, filename)
        
        # Check if file exists before loading
        if os.path.isfile(file_path):
            try:
                # Load model with map_location for device compatibility
                model = torch.load(file_path, map_location=torch.device('cpu'))
                models_dict[i] = model
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")
        else:
            print(f"Warning: File {filename} not found, skipping")
    
    return models_dict

# Usage example:
models = load_models()
# Access model with index 42: models[42]
print(models[1])


# def collect_parameters(model):
#     params = []
#     for param in model.parameters():
#         params.append(param.detach().cpu().numpy().flatten())
#     return np.concatenate(params) if params else np.array([])

# Example usage:
# all_weights_biases = collect_parameters(your_model)
# python main_cifar.py -ld cifar_10_full_dataset -fd -nl 5 -lss 20 -ni 250 weights
def get_models():
    ms = []
    for i in range(101): 
        func_rep = Siren(
                dim_in=2,
                dim_hidden=20,
                dim_out=3,
                num_layers=6,
                final_activation=torch.nn.Identity(),
                w0_initial=30.0,
                w0=30.0
            ).to(device)
        func_rep.load_state_dict(models[i])
        ms.append(func_rep)
    return ms 
 
def collect_layer_weights(models):
    """
    Given a list of models, collect the weights for each named parameter (layer).
    Returns a dictionary:
    
        {
            'layer_name': [
                numpy_array_of_weights_for_model_1,
                numpy_array_of_weights_for_model_2,
                ...,
            ],
            ...
        }
    """
    all_layers_weights = {}

    for model_idx, model in enumerate(models):
        # Iterate through all named parameters (including bias)
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Convert the parameter tensor to a CPU numpy array
                weights_np = param.detach().cpu().numpy().ravel()

                # Accumulate this layer's weights across models
                if name not in all_layers_weights:
                    all_layers_weights[name] = []
                all_layers_weights[name].append(weights_np)
    
    return all_layers_weights

def plot_weights(all_layers_weights, save_path='weights_distribution.png'):
    """
    Given the dictionary of layer weights across models (output of collect_layer_weights),
    make a separate subplot for each layer and plot a boxplot of the distribution
    of weights for all models in that layer.
    
    Finally, save the plot as a PNG in the current directory.
    """

    # Create a subplot for each layer
    n_layers = len(all_layers_weights)
    fig, axes = plt.subplots(n_layers, 1, figsize=(10, 4*n_layers), squeeze=False)

    # Convert axes to a 1D list if it's a 2D array
    axes = axes.flatten()

    for idx, (layer_name, weight_arrays_list) in enumerate(all_layers_weights.items()):
        # weight_arrays_list is a list of arrays (one array per model),
        # each containing the weights for that layer.
        
        # We can combine them into a list of 1D arrays for boxplot:
        axes[idx].boxplot(weight_arrays_list, showfliers=False)  # `showfliers=False` to hide outliers if you want
        axes[idx].set_title(f"Layer: {layer_name}", fontsize=12)
        axes[idx].set_xlabel("Model index")
        axes[idx].set_ylabel("Weight values")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


ms = get_models()
all_layers_weights = collect_layer_weights(ms)
plot_weights(all_layers_weights, save_path="weights_distribution.png")

