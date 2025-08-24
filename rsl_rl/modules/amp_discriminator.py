# Some code in this file is adapted from:
# https://github.com/Alescontrela/AMP_for_hardware/blob/main/rsl_rl/rsl_rl/algorithms/amp_discriminator.py
# Copyright belongs to the original authors. For research and educational use only.

import torch
import torch.nn as nn
from torch import autograd
from rsl_rl.utils import resolve_nn_activation
from rsl_rl.modules.normalizer import EmpiricalNormalization

class AMPDiscriminator(nn.Module):
    def __init__(self, 
            num_amp_obs: int,
            num_amp_steps: int,
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
        self.input_dim = num_amp_obs * num_amp_steps
        self.num_amp_obs = num_amp_obs
        self.num_amp_steps = num_amp_steps
        activation = resolve_nn_activation(activation)
        assert amp_reward_scale >= 0, "AMP reward scale must be non-negative."
        assert 0 <= task_reward_lerp <= 1, "Task reward lerp factor must be in [0, 1]."
        self.amp_reward_scale = amp_reward_scale
        self.task_reward_lerp = task_reward_lerp
        
        # build the discriminator network
        amp_layers = []
        curr_in_dim = self.input_dim
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
        
        expert_data_copy = expert_data.clone().detach().requires_grad_(True)

        disc = self.forward(expert_data_copy)
        ones = torch.ones_like(disc, device=expert_data_copy.device)
        grad = autograd.grad(
            outputs=disc, inputs=expert_data_copy,
            grad_outputs=ones, create_graph=True,
            retain_graph=True, only_inputs=True)[0]

        # Enforce that the grad norm approaches 0.
        grad_penalty = scale * (grad.norm(2, dim=1) - 0).pow(2).mean()
        return grad_penalty

    def predict_amp_reward(self, data: torch.Tensor, dt:float, task_reward: torch.Tensor, amp_normalizer: EmpiricalNormalization):
        """ Predict the AMP reward based on the discriminator output and task reward.

        Args:
            data (torch.Tensor): Input data tensor.
            dt (float): Time step duration in seconds.
            task_reward (torch.Tensor): Task reward tensor.
        """
        
        if self.num_amp_obs * self.num_amp_steps != data.shape[1]:
            raise ValueError(f"[AMPDiscriminator] Input data dimension 1 {data.shape[1]} does not match expected dimension {self.num_amp_obs * self.num_amp_steps}.")
        
        with torch.no_grad():
            self.eval()
            
            # Normalize the input data
            data_copy = data.clone().detach()
            for i in range(self.num_amp_steps):
                data_copy[:, i*self.num_amp_obs:(i+1)*self.num_amp_obs] = amp_normalizer.forward_no_update(data[:, i*self.num_amp_obs:(i+1)*self.num_amp_obs])

            # Compute the discriminator output
            disc = self.forward(data_copy)
            amp_reward = dt * self.amp_reward_scale * torch.clamp(1 - (1/4) * torch.square(disc - 1), min=0)
            reward = self._lerp_reward(amp_reward, task_reward.unsqueeze(-1))
            self.train()

        return reward.squeeze(), disc.squeeze(), amp_reward.squeeze()

    def _lerp_reward(self, amp_rew: torch.Tensor, task_rew: torch.Tensor):
        """Linearly interpolate between the AMP reward and the task reward."""
        rew = (1.0 - self.task_reward_lerp) * amp_rew + self.task_reward_lerp * task_rew
        return rew

