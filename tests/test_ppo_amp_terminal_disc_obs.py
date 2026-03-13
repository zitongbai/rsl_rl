from __future__ import annotations

import torch

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.algorithms.ppo_amp import PPOAMP


def test_process_env_step_uses_terminal_obs_for_done_envs(monkeypatch) -> None:
    class _FakeAmpDiscriminator:
        def __init__(self):
            self.predicted_disc_obs = None

        def get_disc_obs(self, obs, flatten_history_dim: bool = False):
            assert flatten_history_dim is False
            return obs["disc"].clone()

        def get_disc_demo_obs(self, obs, flatten_history_dim: bool = False):
            assert flatten_history_dim is False
            return obs["disc_demo"].clone()

        def predict_style_reward(self, disc_obs: torch.Tensor, dt: float):
            self.predicted_disc_obs = disc_obs.clone()
            return disc_obs[:, 0, 0].clone(), disc_obs[:, 0, 0].clone()

        def lerp_reward(self, task_reward: torch.Tensor, style_reward: torch.Tensor) -> torch.Tensor:
            return style_reward

    class _Buffer:
        def __init__(self):
            self.last = None

        def append(self, value):
            self.last = value.clone()

    captured = {}

    def _fake_super_process_env_step(self, obs, rewards, dones, extras):
        captured["obs"] = obs
        captured["rewards"] = rewards.clone()
        captured["dones"] = dones.clone()
        captured["extras"] = extras

    monkeypatch.setattr(PPO, "process_env_step", _fake_super_process_env_step)

    alg = object.__new__(PPOAMP)
    alg.amp_discriminator = _FakeAmpDiscriminator()
    alg.disc_obs_buffer = _Buffer()
    alg.disc_demo_obs_buffer = _Buffer()
    alg.amp_cfg = {"step_dt": 0.02}

    obs = {
        "disc": torch.tensor([[[1.0]], [[2.0]]]),
        "disc_demo": torch.tensor([[[10.0]], [[20.0]]]),
    }
    rewards = torch.tensor([0.5, 0.25])
    dones = torch.tensor([True, False])
    extras = {
        "terminal_obs": {
            "disc": torch.tensor([[[11.0]], [[22.0]]]),
            "policy": torch.tensor([[[111.0]], [[222.0]]]),
        }
    }

    alg.process_env_step(obs, rewards, dones, extras)

    expected_disc_obs = torch.tensor([[[11.0]], [[2.0]]])
    assert torch.equal(alg.amp_discriminator.predicted_disc_obs, expected_disc_obs)
    assert torch.equal(alg.style_rewards, torch.tensor([11.0, 2.0]))
    assert torch.equal(alg.disc_obs_buffer.last, expected_disc_obs)
    assert torch.equal(alg.disc_demo_obs_buffer.last, obs["disc_demo"])
    assert torch.equal(captured["rewards"], torch.tensor([11.0, 2.0]))
