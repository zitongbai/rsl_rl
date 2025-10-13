from __future__ import annotations

import torch
import torch.nn as nn
from torch import autograd
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.utils import resolve_nn_activation
from rsl_rl.networks import EmpiricalNormalization


class AMPDiscriminator(nn.Module):
    def __init__(self, 
            num_amp_obs: int,
            num_amp_steps: int,
            obs_groups: dict,
            device="cpu",
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
        self.obs_groups = obs_groups
        activation = resolve_nn_activation(activation)
        assert amp_reward_scale >= 0, "AMP reward scale must be non-negative."
        assert 0 <= task_reward_lerp <= 1, "Task reward lerp factor must be in [0, 1]."
        self.amp_reward_scale = amp_reward_scale
        self.task_reward_lerp = task_reward_lerp
        
        # AMP observation normalizer
        self.amp_obs_normalizer = EmpiricalNormalization(shape=self.num_amp_obs, until=1e8).to(device)
        
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

    def forward(self, x)-> torch.Tensor:
        """Forward pass for the AMP Discriminator.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        h = self.trunk(x)
        d = self.amp_linear(h)
        return d
    
    def update_normalization(self, x: torch.Tensor):
        """Update the AMP observation normalizer with new data.

        Args:
            x (torch.Tensor): Input tensor.
        """
        if len(x.shape) != 2:
            raise ValueError(f"[AMPDiscriminator] Input data for update_normalization must be 2D, but got {len(x.shape)}D.")
        if self.num_amp_obs != x.shape[1]:
            raise ValueError(f"[AMPDiscriminator] Input data dimension 1 {x.shape[1]} does not match expected dimension {self.num_amp_obs}. Only use one step of AMP observation to update the normalizer.")
        self.amp_obs_normalizer.update(x)
        
    def get_amp_obs(self, obs: TensorDict, flatten_history_dim: bool = False):
        """Extract and concatenate AMP observations from the observation dictionary.

        Args:
            obs (TensorDict): Observation dictionary.
            flatten_history_dim (bool, optional): Whether to flatten the history dimension. Defaults to False.
        Returns:
            torch.Tensor: Concatenated AMP observation tensor. If flatten_history_dim is True, 
                        the shape is [num_envs, history_length * total_amp_obs_dim], 
                        otherwise [num_envs, history_length, total_amp_obs_dim].
                        Most recent entry at the end and oldest entry at the beginning.
        """
        amp_obs_list = []
        for obs_group in self.obs_groups["amp"]:
            obs_tensor = obs[obs_group] # [num_envs, history_length, obs_dim]
            assert len(obs_tensor.shape) == 3, "The AMP module only supports 1D observations with history"
            num_envs, history_length, obs_dim = obs_tensor.shape
            assert history_length == self.num_amp_steps, f"The AMP observation history length {history_length} does not match the expected number of steps {self.num_amp_steps}."
            amp_obs_list.append(obs_tensor)
        amp_obs = torch.cat(amp_obs_list, dim=-1) # [num_envs, history_length, total_amp_obs_dim]
        if flatten_history_dim:
            amp_obs = amp_obs.view(num_envs, -1) # [num_envs, history_length * total_amp_obs_dim]
        return amp_obs

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

    def predict_amp_reward(self, amp_obs: torch.Tensor, dt: float):
        """Predict the AMP reward based on the AMP observations.

        Args:
            amp_obs (torch.Tensor): Input AMP observation tensor with shape [num_envs, history_length, amp_obs_dim].
            dt (float): Time step duration in seconds.

        Returns:
            torch.Tensor: AMP reward tensor with shape [num_envs].
            torch.Tensor: Discriminator output tensor with shape [num_envs].
        """
        
        if len(amp_obs.shape) != 3: 
            raise ValueError(f"[AMPDiscriminator] Input data for predict_amp_reward must be 3D (num_envs, history_length, amp_obs_dim), but got {len(amp_obs.shape)}D.")
        if self.num_amp_obs != amp_obs.shape[2]:
            raise ValueError(f"[AMPDiscriminator] Input data dimension 2 {amp_obs.shape[2]} does not match expected dimension {self.num_amp_obs}.")
        if self.num_amp_steps != amp_obs.shape[1]:
            raise ValueError(f"[AMPDiscriminator] Input data dimension 1 {amp_obs.shape[1]} does not match expected dimension {self.num_amp_steps}.")
        
        with torch.no_grad():
            self.eval()
            
            # Normalize the input data
            # amp_obs shape: [num_envs, history_length, amp_obs_dim]
            amp_obs_normalized = torch.zeros_like(amp_obs)
            for i in range(self.num_amp_steps):
                amp_obs_normalized[:, i, :] = self.amp_obs_normalizer.forward(amp_obs[:, i, :])
                
            # flatten the history dimension and compute the discriminator output
            amp_obs_normalized_flat = amp_obs_normalized.view(amp_obs.shape[0], -1) # [num_envs, history_length * amp_obs_dim]
            disc = self.forward(amp_obs_normalized_flat) # [num_envs, 1]
            amp_reward = dt * self.amp_reward_scale * torch.clamp(1 - (1/4) * torch.square(disc - 1), min=0) # [num_envs, 1]
            
            self.train()
            
        return amp_reward.squeeze(), disc.squeeze()

    # def predict_amp_reward(self, data: torch.Tensor, dt:float, task_reward: torch.Tensor, amp_normalizer: EmpiricalNormalization):
    #     """ Predict the AMP reward based on the discriminator output and task reward.

    #     Args:
    #         data (torch.Tensor): Input data tensor.
    #         dt (float): Time step duration in seconds.
    #         task_reward (torch.Tensor): Task reward tensor.
    #     """
        
    #     if self.num_amp_obs * self.num_amp_steps != data.shape[1]:
    #         raise ValueError(f"[AMPDiscriminator] Input data dimension 1 {data.shape[1]} does not match expected dimension {self.num_amp_obs * self.num_amp_steps}.")
        
    #     with torch.no_grad():
    #         self.eval()
            
    #         # Normalize the input data
    #         data_copy = data.clone().detach()
    #         for i in range(self.num_amp_steps):
    #             data_copy[:, i*self.num_amp_obs:(i+1)*self.num_amp_obs] = amp_normalizer.forward_no_update(data[:, i*self.num_amp_obs:(i+1)*self.num_amp_obs])

    #         # Compute the discriminator output
    #         disc = self.forward(data_copy)
    #         amp_reward = dt * self.amp_reward_scale * torch.clamp(1 - (1/4) * torch.square(disc - 1), min=0)
    #         reward = self._lerp_reward(amp_reward, task_reward.unsqueeze(-1))
    #         self.train()

    #     return reward.squeeze(), disc.squeeze(), amp_reward.squeeze()

    def lerp_reward(self, amp_rew: torch.Tensor, task_rew: torch.Tensor):
        """Linearly interpolate between the AMP reward and the task reward."""
        rew = (1.0 - self.task_reward_lerp) * amp_rew + self.task_reward_lerp * task_rew
        return rew


def resolve_amp_config(alg_cfg, obs: TensorDict, obs_groups: dict, env: VecEnv):
    if "amp_cfg" in alg_cfg and alg_cfg["amp_cfg"] is not None:
        # Try to get the motion data loader from the ENV
        # TODO: support multiple motion datasets
        motion_dataloader_term = alg_cfg["amp_cfg"].get("motion_dataset", None)
        if motion_dataloader_term is None:
            raise ValueError("AMP configuration requires a motion dataset.")
        alg_cfg["amp_cfg"]["_motion_loader"] = env.unwrapped.motion_data_manager.get_term(motion_dataloader_term)
        # get example AMP observation to infer dimensions
        num_amp_obs = 0
        for obs_group in obs_groups["amp"]:
            obs_tensor = obs[obs_group] # [num_envs, history_length, obs_dim]
            assert len(obs_tensor.shape) == 3, "The AMP module only supports 1D observations with history"
            num_amp_obs += obs_tensor.shape[2]
        num_amp_steps = obs_tensor.shape[1]
        
        # this is used by the AMP discriminator to handle the input dimension
        alg_cfg["amp_cfg"]["num_amp_steps"] = num_amp_steps
        alg_cfg["amp_cfg"]["num_amp_obs"] = num_amp_obs
        # step_dt would be used in computing the AMP reward
        alg_cfg["amp_cfg"]["step_dt"] = env.env.unwrapped.step_dt
        
        # AMP normalizer
        # TODO
        
        return alg_cfg
    else:
        raise ValueError