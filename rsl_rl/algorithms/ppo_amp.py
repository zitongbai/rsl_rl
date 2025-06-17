from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from itertools import chain

from rsl_rl.modules import ActorCritic
from rsl_rl.modules.rnd import RandomNetworkDistillation
from rsl_rl.storage import RolloutStorage, CircularBuffer
from rsl_rl.utils import string_to_callable

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.modules.amp_discriminator import AMPDiscriminator
from rsl_rl.modules.normalizer import EmpiricalNormalization


class PPOAmp(PPO):
    
    policy: ActorCritic
    """The actor critic module."""
    
    def __init__(
        self,
        policy,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        normalize_advantage_per_mini_batch=False,
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # AMP parameters
        amp_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
    ):
        super().__init__(
            policy=policy,
            num_learning_epochs=num_learning_epochs,
            num_mini_batches=num_mini_batches,
            clip_param=clip_param,
            gamma=gamma,
            lam=lam,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            use_clipped_value_loss=use_clipped_value_loss,
            schedule=schedule,
            desired_kl=desired_kl,
            device=device,
            normalize_advantage_per_mini_batch=normalize_advantage_per_mini_batch,
            rnd_cfg=rnd_cfg,
            symmetry_cfg=symmetry_cfg,
            multi_gpu_cfg=multi_gpu_cfg,
        )
        
        # Store AMP configuration
        self.amp_cfg = amp_cfg
        
        # AMP Discriminator
        if self.amp_cfg is not None:
            self.amp_discriminator = AMPDiscriminator(
                input_dim=self.amp_cfg["num_amp_obs"],
                **self.amp_cfg["amp_discriminator"]
            ).to(self.device)
            
            # optimizer for policy and discriminator
            params = [
                {
                    "name": "amp_trunk", 
                    "params": self.amp_discriminator.trunk.parameters(),
                    "weight_decay": self.amp_cfg["amp_trunk_weight_decay"],  # L2 regularization for the discriminator trunk
                },
                {
                    "name": "amp_linear",
                    "params": self.amp_discriminator.amp_linear.parameters(),
                    "weight_decay": self.amp_cfg["amp_linear_weight_decay"],  # L2 regularization for the discriminator linear layer
                }
            ]
            # use a separate optimizer for the AMP discriminator
            self.amp_optimizer = optim.Adam(
                params,
                lr=self.amp_cfg["amp_learning_rate"],
            )
            self.amp_max_grad_norm = self.amp_cfg["amp_max_grad_norm"]

            # Storage
            self.amp_replay_buffer: CircularBuffer = None   # type: ignore
            
            # Motion loader
            self.motion_loader = self.amp_cfg["_motion_loader"]
            
            # AMP normalizer (amp obs and motion data share the same normalizer)
            self.amp_normalizer: EmpiricalNormalization = self.amp_cfg["_amp_normalizer"]

        else:
            self.amp_discriminator = None

        
    def init_storage(
        self, training_type, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, actions_shape
    ):
        super().init_storage(
            training_type=training_type,
            num_envs=num_envs,
            num_transitions_per_env=num_transitions_per_env,
            actor_obs_shape=actor_obs_shape,
            critic_obs_shape=critic_obs_shape,
            actions_shape=actions_shape
        )
        
        # Initialize the AMP replay buffer
        if self.amp_cfg is not None:
            self.amp_replay_buffer = CircularBuffer(
                max_len=self.amp_cfg["replay_buffer_size"],
                batch_size=num_envs,
                device=self.device
            )
    
    def process_env_step(self, rewards, dones, infos):
        self.amp_replay_buffer.append(infos["amp_obs_processed"])
        super().process_env_step(rewards, dones, infos)


    def update(self):  # noqa: C901
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        # -- RND loss
        if self.rnd:
            mean_rnd_loss = 0
        else:
            mean_rnd_loss = None
        # -- Symmetry loss
        if self.symmetry:
            mean_symmetry_loss = 0
        else:
            mean_symmetry_loss = None
        if self.amp_discriminator:
            mean_amp_loss = 0
            mean_grad_penalty_loss = 0
            mean_policy_disc = 0
            mean_expert_disc = 0

        # generator for mini batches
        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        amp_replay_generator = self.amp_replay_buffer.mini_batch_generator(
            fetch_length=self.storage.num_transitions_per_env,
            num_mini_batches=self.num_mini_batches,
            num_epochs=self.num_learning_epochs
        )
        
        motion_loader_generator = self.motion_loader.mini_batch_generator(
            num_transitions_per_env=self.storage.num_transitions_per_env,
            num_mini_batches=self.num_mini_batches,
            num_epochs=self.num_learning_epochs
        )

        for sample, amp_replay_batch, motion_data_batch in zip(
            generator, amp_replay_generator, motion_loader_generator
        ):
            (
                obs_batch,
                critic_obs_batch,
                actions_batch,
                target_values_batch,
                advantages_batch,
                returns_batch,
                old_actions_log_prob_batch,
                old_mu_batch,
                old_sigma_batch,
                hid_states_batch,
                masks_batch,
                rnd_state_batch,
            ) = sample
            
            # number of augmentations per sample
            # we start with 1 and increase it if we use symmetry augmentation
            num_aug = 1
            # original batch size
            original_batch_size = obs_batch.shape[0]

            # check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            # Perform symmetric augmentation
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                # augmentation using symmetry
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                # returned shape: [batch_size * num_aug, ...]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch, actions=actions_batch, env=self.symmetry["_env"], obs_type="policy"
                )
                critic_obs_batch, _ = data_augmentation_func(
                    obs=critic_obs_batch, actions=None, env=self.symmetry["_env"], obs_type="critic"
                )
                # compute number of augmentations per sample
                num_aug = int(obs_batch.shape[0] / original_batch_size)
                # repeat the rest of the batch
                # -- actor
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                # -- critic
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)
            
            # Recompute actions log prob and entropy for current batch of transitions
            # Note: we need to do this because we updated the policy with the new parameters
            # -- actor
            self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            # -- critic
            value_batch = self.policy.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            # -- entropy
            # we only keep the entropy of the first augmentation (the original one)
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]
            
            # KL
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # Update the learning rate
                    # Perform this adaptation only on the main process
                    # TODO: Is this needed? If KL-divergence is the "same" across all GPUs,
                    #       then the learning rate should be the same across all GPUs.
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # Update the learning rate for all GPUs
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    # Update the learning rate for all parameter groups
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate
            
            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # Discriminator loss
            if self.amp_discriminator:
                with torch.no_grad():
                    num_amp_obs_1_step = int(amp_replay_batch.shape[1] // 2)
                    
                    self.amp_normalizer.eval()
                    # only forward, no update
                    amp_replay_batch[:, num_amp_obs_1_step:] = self.amp_normalizer(amp_replay_batch[:, num_amp_obs_1_step:])
                    motion_data_batch[:, num_amp_obs_1_step:] = self.amp_normalizer(motion_data_batch[:, num_amp_obs_1_step:])
                    
                    self.amp_normalizer.train()
                    # forward and update the AMP normalizer
                    amp_replay_batch[:, :num_amp_obs_1_step] = self.amp_normalizer(amp_replay_batch[:, :num_amp_obs_1_step])
                    motion_data_batch[:, :num_amp_obs_1_step] = self.amp_normalizer(motion_data_batch[:, :num_amp_obs_1_step])
                
                policy_disc = self.amp_discriminator(amp_replay_batch)
                expert_disc = self.amp_discriminator(motion_data_batch)
                policy_loss = torch.nn.MSELoss()(
                    policy_disc, -1 * torch.ones_like(policy_disc, device=self.device)
                )
                expert_loss = torch.nn.MSELoss()(
                    expert_disc, torch.ones_like(expert_disc, device=self.device)
                )
                disc_loss = 0.5 * (policy_loss + expert_loss)
                grad_penalty_loss = self.amp_discriminator.compute_grad_pen(
                    motion_data_batch, scale=self.amp_cfg["grad_penalty_scale"]
                )
                
                amp_loss = disc_loss + grad_penalty_loss
                
            # Symmetry loss
            if self.symmetry:
                # obtain the symmetric actions
                # if we did augmentation before then we don't need to augment again
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(
                        obs=obs_batch, actions=None, env=self.symmetry["_env"], obs_type="policy"
                    )
                    # compute number of augmentations per sample
                    num_aug = int(obs_batch.shape[0] / original_batch_size)

                # actions predicted by the actor for symmetrically-augmented observations
                mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())

                # compute the symmetrically augmented actions
                # note: we are assuming the first augmentation is the original one.
                #   We do not use the action_batch from earlier since that action was sampled from the distribution.
                #   However, the symmetry loss is computed using the mean of the distribution.
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"], obs_type="policy"
                )

                # compute the loss (we skip the first augmentation as it is the original one)
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                )
                # add the loss to the total loss
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # Random Network Distillation loss
            if self.rnd:
                # predict the embedding and the target
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                # compute the loss as the mean squared error
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)
                
            # Compute the gradients
            # -- For PPO
            self.optimizer.zero_grad()
            
            if self.amp_discriminator:
                self.amp_optimizer.zero_grad()
                loss.backward(retain_graph=True) # retain graph for AMP loss
                amp_loss.backward()
            else:
                loss.backward()
                
            # -- For RND
            if self.rnd:
                self.rnd_optimizer.zero_grad()  # type: ignore
                rnd_loss.backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients
            # -- For PPO
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            # -- For AMP Discriminator
            if self.amp_discriminator:
                nn.utils.clip_grad_norm_(self.amp_discriminator.parameters(), self.amp_max_grad_norm)
                self.amp_optimizer.step()
            
            # -- For RND
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            # -- RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # -- Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()
            # -- AMP losses
            if mean_amp_loss is not None:
                mean_amp_loss += amp_loss.item()
                mean_grad_penalty_loss += grad_penalty_loss.item()
                mean_policy_disc += policy_disc.mean().item()
                mean_expert_disc += expert_disc.mean().item()
            
        # -- For PPO
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        # -- For RND
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        # -- For Symmetry
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        # -- For AMP
        if mean_amp_loss is not None:
            mean_amp_loss /= num_updates
            mean_grad_penalty_loss /= num_updates
            mean_policy_disc /= num_updates
            mean_expert_disc /= num_updates    
        # -- Clear the storage
        self.storage.clear()

        # construct the loss dictionary
        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss
        if self.amp_discriminator:
            loss_dict["amp_loss"] = mean_amp_loss
            loss_dict["grad_penalty_loss"] = mean_grad_penalty_loss
            loss_dict["policy_disc"] = mean_policy_disc
            loss_dict["expert_disc"] = mean_expert_disc
            
        return loss_dict