import torch
import math
from torch import nn
import torch.nn.functional as F
from typing import List, Optional, Union, Dict, Any
from dataclasses import dataclass
from transformers.utils import ModelOutput

@dataclass
class LossInput(ModelOutput):
    input: Optional[torch.FloatTensor] = None
    target: Optional[torch.FloatTensor] = None
    all_loraout_dict: Optional[List[Dict[str, torch.Tensor]]] = None
    attention_mask: Optional[torch.BoolTensor] = None
    kwargs: Optional[Dict[str, Any]] = None


class BaseLoss(nn.Module):
    """
    Base class for all loss functions that accept LossInput as input.
    Provides a standardized interface and common utilities.
    """
    
    def __init__(self, name: Optional[str] = None, **kwargs):
        super().__init__()
        self.name = name or self.__class__.__name__
        self.mode = kwargs.get('mode', None) 
        
    def forward(self, loss_input: LossInput) -> torch.Tensor:
        """
        Forward pass that must be implemented by subclasses.
        
        Args:
            loss_input: LossInput object containing all necessary tensors
            
        Returns:
            torch.Tensor: Computed loss value
        """
        raise NotImplementedError("Forward method must be implemented in subclasses.")
    
    def validate_input(self, loss_input: LossInput) -> None:
        """Validate input before processing. Override in subclasses if needed."""
        if loss_input.input is None or loss_input.target is None:
            raise ValueError(f"{self.name} requires both input and target tensors")


class CompositeLoss(nn.Module):
    """
    Composite loss that combines multiple loss functions with weights.
    Enables dynamic addition and removal of loss components.
    """
    
    def __init__(self, losses: Optional[List[BaseLoss]] = None, 
                 weights: Optional[List[float]] = None, **kwargs):
        super().__init__()
        self.losses = nn.ModuleList(losses or [])
        self.weights = weights or [1.0] * len(self.losses)
        
    def add_loss(self, loss: BaseLoss, weight: float = 1.0) -> None:
        self.losses.append(loss)
        self.weights.append(weight)
        
    def remove_loss(self, index: int) -> None:
        if 0 <= index < len(self.losses):
            del self.losses[index]
            del self.weights[index]
            
    def forward(self, input, target, all_loraout_dict, attn_mask) -> torch.Tensor:
        if not self.losses:
            raise ValueError("No loss functions configured")
        loss_input = LossInput(
            input=input,
            target=target,
            all_loraout_dict=all_loraout_dict,
            attention_mask=attn_mask
        )
        losses = []
        self.individual_losses = {}
        for loss_fn, weight in zip(self.losses, self.weights):
            loss_value = loss_fn(loss_input)
            if not math.isfinite(loss_value.detach()):
                print(f"Loss is {loss_value}, stopping training")
                print(f"Loss function: {loss_fn.name}")
                print(f"Weight: {weight}")
                print(f"Loss value: {loss_value}")
                print(f"Input shape: {input.shape if input is not None else None}")
                print(f"Target shape: {target.shape if target is not None else None}")
                raise Exception(f"Loss from {loss_fn.name} is not finite")
            weighted_loss = weight * loss_value
            losses.append(weighted_loss)
            self.individual_losses[loss_fn.name] = loss_value.detach()
        total_loss = torch.stack(losses).sum()
        return total_loss


class LoadBalancingLoss(BaseLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def validate_input(self, loss_input):
        all_loraout_dict = loss_input.all_loraout_dict
        if not all_loraout_dict:
            raise ValueError(
                f"{self.name} requires all_loraout_dict in LossInput")

    def forward(self, loss_input: LossInput) -> torch.Tensor:
        """
        Forward pass for load balancing loss.
        
        Args:
            loss_input: LossInput object containing routing_weight and attention_mask
            
        Returns:
            torch.Tensor: Computed load balancing loss
        """
        self.validate_input(loss_input)
        attn_mask = loss_input.attention_mask
        all_loraout_dict = loss_input.all_loraout_dict
        total_losses = []
        for i, loraout_dict in enumerate(all_loraout_dict):
            for key, loraout in loraout_dict.items():
                routing_weight = loraout.routing_weight
                if routing_weight is not None:
                    loss = self._forward(attn_mask, routing_weight)
                    total_losses.append(loss)
        total_loss = torch.stack(total_losses).mean() if total_losses else loss_input.input.new_tensor(0.0)
        return total_loss

    def _forward(self, attn_mask, routing_weight) -> torch.Tensor:
        num_experts = routing_weight.shape[-1]
        mask = attn_mask.to(routing_weight.dtype)
        num_token = mask.sum()
        masked_weight = routing_weight * mask.unsqueeze(-1)
        count = torch.sign(routing_weight * mask.unsqueeze(-1))
        activate_count = count.sum()
        assert torch.any(routing_weight != 0), "All routing weights are zero"
        assert activate_count > 0, "No experts activated"
        # count.sum() for dynamic routing for each token
        freq = torch.sum(count.view(-1, num_experts), dim=0) / activate_count
        prop = torch.sum(masked_weight .view(-1, num_experts), dim=0) / num_token
        loss = torch.sum(prop * freq) * num_experts
        return loss


class Router_z_loss(BaseLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def validate_input(self, loss_input: LossInput) -> None:
        all_loraout_dict = loss_input.all_loraout_dict
        if not all_loraout_dict:
            raise ValueError(
                f"{self.name} requires all_loraout_dict in LossInput")
    
    def forward(self, loss_input: LossInput) -> torch.Tensor:
        self.validate_input(loss_input)
        all_loraout_dict = loss_input.all_loraout_dict
        loss = 0
        total_count = 0
        for i, loraout_dict in enumerate(all_loraout_dict):
            for key, loraout in loraout_dict.items():
                gate_score = loraout.gate_score
                if gate_score is not None:
                    router_z_loss = torch.logsumexp(gate_score, dim = -1)
                    router_z_loss = torch.square(router_z_loss)            
                    router_z_loss = router_z_loss.mean()
                    loss += router_z_loss
                    total_count += 1

        return loss / max(total_count, 1)


class CrossEntropyLoss(BaseLoss):
    def __init__(self, ignore_index=-100, weight=None, 
                 reduction='mean', label_smoothing=0.0, **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            weight=weight,
            reduction=reduction,
            label_smoothing=label_smoothing
        )
    
    def forward(self, loss_input: LossInput) -> torch.Tensor:
        self.validate_input(loss_input)
        return self.loss_fn(loss_input.input, loss_input.target)


class MSELoss(BaseLoss):
    def __init__(self, reduction='mean', **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = nn.MSELoss(reduction=reduction)
    
    def forward(self, loss_input: LossInput) -> torch.Tensor:
        self.validate_input(loss_input)
        return self.loss_fn(loss_input.input, loss_input.target)


class RegularizationLoss(BaseLoss):
    """
    Base class for regularization lambda losses.
    """
    def __init__(self, lora_modules: Optional[List[nn.Module]] = None, **kwargs):
        super().__init__(**kwargs)

    def validate_input(self, loss_input: LossInput) -> None:
        all_loraout_dict = loss_input.all_loraout_dict
        if not all_loraout_dict:
            raise ValueError(f"{self.name} requires all_loraout_dict in LossInput")
    
    def forward(self, loss_input: LossInput) -> torch.Tensor:
        """
        Forward pass for regularization loss.
        
        Args:
            loss_input: LossInput object containing all necessary tensors
            
        Returns:
            torch.Tensor: Computed regularization loss
        """
        self.validate_input(loss_input)
        total_loss = 0.0
        total_count = 0

        for i, loraout_dict in enumerate(loss_input.all_loraout_dict):
            for key, loraout in loraout_dict.items():
                lam = loraout.lam
                gate_score = loraout.gate_score
                if gate_score is not None and lam is not None:
                    lower, upper = lambda_interval_k(gate_score, k=2)
                    total_loss += torch.relu(lower - lam).mean()
                total_count += 1
        return total_loss / max(total_count, 1)


def lambda_interval_k(gate_score: torch.Tensor, k: int = 2):
    u = gate_score.detach()
    u_sorted, _ = torch.sort(u, descending=True, dim=-1)
    U_k = u_sorted[:,:,:k].sum(dim=-1)  # (bs, seq_len)
    lower = 1.0 - (U_k - k * u_sorted[:,:,k])     # inclusive
    upper = 1.0 - (U_k - k * u_sorted[:,:,k-1])   # exclusive
    return lower, upper


class LamSparseLoss(BaseLoss):
    def __init__(self, lora_modules: Optional[List[nn.Module]] = None, **kwargs):
        super().__init__(**kwargs)

    def validate_input(self, loss_input: LossInput) -> None:
        all_loraout_dict = loss_input.all_loraout_dict
        if not all_loraout_dict:
            raise ValueError(f"{self.name} requires all_loraout_dict in LossInput")
    
    def forward(self, loss_input: LossInput) -> torch.Tensor:
        self.validate_input(loss_input)
        total_losses = []
        for i, loraout_dict in enumerate(loss_input.all_loraout_dict):
            for key, loraout in loraout_dict.items():
                lam = loraout.lam
                if lam is not None:
                    loss = (1 - lam).mean()
                    total_losses.append(loss)
        total_loss = torch.stack(total_losses).mean()
        return total_loss