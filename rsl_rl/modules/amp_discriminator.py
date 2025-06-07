# Some code in this file is adapted from:
# https://github.com/Alescontrela/AMP_for_hardware/blob/main/rsl_rl/rsl_rl/algorithms/amp_discriminator.py
# Copyright belongs to the original authors. For research and educational use only.

import torch
import torch.nn as nn
from torch import autograd
from rsl_rl.utils import resolve_nn_activation

class AMPDiscriminator(nn.Module):
    def __init__(self, 
            input_dim, 
            hidden_dims = [256, 256, 256],
            activation="relu",
            amp_reward_scale=1.0, 
            task_reward_lerp=0.0
        ):
        """AMP Discriminator for Adversarial Motion Planning (AMP).

        Args:
            input_dim (int): The dimensionality of the input.
            hidden_dims (list, optional): The dimensionality of the hidden layers. Defaults to [256, 256, 256].
            activation (str, optional): The activation function to use. Defaults to "identity".
            amp_reward_scale (float, optional): The scaling factor for the AMP reward. Defaults to 1.0.
            task_reward_lerp (float, optional): The linear interpolation factor for the task reward. Defaults to 0.0.
        """
        super().__init__()
        self.input_dim = input_dim
        activation = resolve_nn_activation(activation)
        assert amp_reward_scale >= 0, "AMP reward scale must be non-negative."
        assert 0 <= task_reward_lerp <= 1, "Task reward lerp factor must be in [0, 1]."
        self.amp_reward_scale = amp_reward_scale
        self.task_reward_lerp = task_reward_lerp
        
        # build the discriminator network
        amp_layers = []
        curr_in_dim = input_dim
        for hidden_dim in hidden_dims:
            amp_layers.append(nn.Linear(curr_in_dim, hidden_dim))
            amp_layers.append(activation)
            curr_in_dim = hidden_dim
        self.trunk = nn.Sequential(*amp_layers)
        self.amp_linear = nn.Linear(hidden_dims[-1], 1)
        
        print(f"AMP Discriminator MLP: {self.trunk}")
        print(f"AMP Discriminator Output Layer: {self.amp_linear}")


    def forward(self, x):
        """Forward pass for the AMP Discriminator.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        h = self.trunk(x)
        d = self.amp_linear(h)
        return d

    def compute_grad_pen(self, expert_data: torch.Tensor, scale=10):
        """Compute the gradient penalty for the AMP Discriminator.

        Args:
            expert_data (torch.Tensor): Expert data tensor, with two steps concatenated.
            scale (int, optional): Weight for the gradient penalty. Defaults to 10.

        Returns:
            torch.Tensor: Scaled gradient penalty.
        """

        disc = self.forward(expert_data)
        ones = torch.ones_like(disc, device=expert_data.device)
        grad = autograd.grad(
            outputs=disc, inputs=expert_data,
            grad_outputs=ones, create_graph=True,
            retain_graph=True, only_inputs=True)[0]

        # Enforce that the grad norm approaches 0.
        grad_penalty = scale * (grad.norm(2, dim=1) - 0).pow(2).mean()
        return grad_penalty

    def predict_amp_reward(self, data: torch.Tensor, task_reward: torch.Tensor):
        """ Predict the AMP reward based on the discriminator output and task reward.

        Args:
            data (torch.Tensor): Input data tensor.
            task_reward (torch.Tensor): Task reward tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The predicted AMP reward and discriminator output.
        """
        with torch.no_grad():
            self.eval()
            disc = self.forward(data)
            amp_reward = self.amp_reward_scale * torch.clamp(1 - (1/4) * torch.square(disc - 1), min=0)
            reward = self._lerp_reward(amp_reward, task_reward.unsqueeze(-1))
            self.train()
        return reward.squeeze(), disc

    def _lerp_reward(self, amp_rew: torch.Tensor, task_rew: torch.Tensor):
        """Linearly interpolate between the AMP reward and the task reward."""
        rew = (1.0 - self.task_reward_lerp) * amp_rew + self.task_reward_lerp * task_rew
        return rew

