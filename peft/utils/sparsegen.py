import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F

class UniversalMLP(nn.Module):
    def __init__(self, proj_sizes=[1024, 2048, 3072], hidden=64, output_activation="logsigmoid"):
        super().__init__()
        self.proj_layers = nn.ModuleDict({
            str(in_dim): nn.Sequential(
                nn.Linear(in_dim, hidden, bias=True),
                nn.ReLU(),
                nn.Linear(hidden, 1, bias=True),
                nn.LogSigmoid()
            )
            for in_dim in proj_sizes
        })

    def forward(self, x):
        in_dim = x.size(-1)
        out = self.proj_layers[str(in_dim)](x)
        return out


def initialize_sparsegen_weights(mlp_or_module, init_strategy="kaiming"):
    """
    Unified initialization function for sparsegen MLP weights.
    
    Args:
        mlp_or_module: Either an nn.Sequential MLP, Sparsegen_lin module, or GlobalSparsegen module
        init_strategy: Initialization strategy - "kaiming", "zeros", "sparse", "dense"
        layer_idx: Layer index for layer-dependent bias (used with local sparsegen)
        total_layers: Total number of layers (used with local sparsegen)
        verbose: Whether to print initialization info
        
    Supported strategies:
        - "kaiming": Standard Kaiming uniform initialization
        - "zeros": Zero initialization for weights
        - "sparse": Initialize for sparse routing (λ≈1, high threshold)
        - "dense": Initialize for dense routing (λ≈0, low threshold)
    """
    for i, layer in enumerate(mlp_or_module.modules()):
        if isinstance(layer, nn.Linear):
            _initialize_linear_layer(layer, init_strategy)
                

def _initialize_linear_layer(layer, init_strategy):
    """Initialize a single linear layer based on the strategy."""
    
    if init_strategy == "kaiming":
        nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
            
    elif init_strategy == "zeros":
        nn.init.zeros_(layer.weight)
            
    elif init_strategy == "sparse":
        # Initialize for sparse routing (λ≈1, high threshold)
        # LogSigmoid(output) ≈ 0, so output should be large positive
        nn.init.uniform_(layer.weight, a=0.0, b=0.5)
            
    elif init_strategy == "dense":
        # Initialize for dense routing (λ≈0, low threshold)
        # LogSigmoid(output) ≈ -1 or more negative, so output should be large negative
        nn.init.uniform_(layer.weight, a=-0.5, b=0.0)
            
    else:
        nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
    
    # Initialize bias
    if layer.bias is not None:
        if init_strategy == "zeros":
            # For zeros strategy, use layer-dependent bias
            # base_bias = -2.5
            # layer_bias = layer_idx / max(total_layers, 1)  # Avoid division by zero
            base_bias = -2.0
            layer_bias = 0
            nn.init.constant_(layer.bias, base_bias + layer_bias)
        else:
            # For other strategies, use moderate bias for sparsity
            nn.init.constant_(layer.bias, -2.5)


class GlobalSparsegen(torch.nn.Module):
    """
    A global sparsegen function that can be shared across all attention layers and Q,K,V projections.
    This reduces the number of parameters compared to having individual sparsegen functions.
    Pre-creates MLPs for known input sizes to avoid dynamic creation during forward pass.
    """
    def __init__(self, input_sizes, hidden_size, eps=1e-2, 
                 init_strategy="kaiming", mlp_activation="logsigmoid"):
        super(GlobalSparsegen, self).__init__()
        
        self.base_feature_size = input_sizes  # Store base size for reference
        self.eps = eps
        self.init_strategy = init_strategy
        self.mlp = UniversalMLP(
            proj_sizes=input_sizes, hidden=hidden_size, output_activation=mlp_activation)
        initialize_sparsegen_weights(
            self.mlp, 
            init_strategy=self.init_strategy, 
        )

    def forward(self, z, x):
        """
        Forward pass for global sparsegen.
        
        Args:
            z: [bs, seqlen, lora_num] - routing logits
            x: [bs, seqlen, features] - input features
            
        Returns:
            prob: [bs, seqlen, lora_num] - sparsegen probabilities
        """
        if z.dim() == 3:
            bs, seqlen, dim = z.size()
            flat_in = z.view(bs * seqlen, dim)
            flat_x = x.view(bs * seqlen, -1)
            prob, lam = self._compute_prob(flat_in, flat_x)
            prob = prob.view(bs, seqlen, dim)
            lam = lam.view(bs, seqlen)
        elif z.dim() == 2:
            prob, lam = self._compute_prob(z, x)
        else:
            raise ValueError(f"Expected 2D or 3D z; got {z.dim()}D.")
        return prob, lam

    def _compute_prob(self, z, x):
        """Core sparsegen logic for 2D tensors."""
        bs, dim = z.size()
        device = z.device
        # compute in float32 for numerical stability, cast back at the end
        orig_dtype = z.dtype
        z = z.float()
        x = x.float()

        # Sort and find k(z)
        z_sorted = torch.sort(z, descending=True)[0]
        z_cumsum = torch.cumsum(z_sorted, dim=1)
        k = Variable(torch.arange(1, dim + 1).unsqueeze(0).repeat(bs, 1)).to(device)
        lam = self.mlp(x).to(device) + (1 - self.eps)
        z_check = torch.gt(1 - lam + k * z_sorted, z_cumsum)
        k_z = torch.sum(z_check.float(), 1).clamp_min(1)
        # tau(z)
        tausum = torch.sum(z_check.float() * z_sorted, -1)  
        tau_z = (tausum - 1 + lam.squeeze(1)) / k_z
        tau_z = tau_z.view(bs, 1).repeat(1, dim)
        prob = z.sub(tau_z).clamp(min=0)

        lam_rep = lam.repeat(1, dim)
        denom = torch.clamp(1 - lam_rep, min=self.eps)
        prob /= denom
        
        if torch.isnan(prob).any() or torch.isinf(prob).any():
            print("prob is {}, stopping training".format(prob))
            exit()
            
        return prob.to(orig_dtype), lam.to(orig_dtype)
