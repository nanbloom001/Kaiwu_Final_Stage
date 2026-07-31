#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Recurrent visual actor with an asymmetric privileged critic."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from agent_ppo.model.actor_critic import ActorCritic
from agent_ppo.model.vision_encoder import VisionEncoder


class VisualActorCritic(ActorCritic):
    """Depth/proprio actor and raw privileged-state critic for visual PPO.

    The Camera policy tensor keeps the platform layout
    ``[proprio | height_scan | depth]``. The actor deliberately skips the
    height-scan slice; it consumes only proprio and depth. The critic receives
    the independent 316-D critic observation supplied by Isaac Lab.
    """

    is_recurrent = True

    def __init__(
        self,
        num_proprio: int,
        num_scan: int,
        depth_shape: Tuple[int, int, int],
        latent_dim: int,
        cnn_output_dim: int,
        lstm_hidden_size: int,
        lstm_num_layers: int,
        num_critic_obs: int,
        num_actions: int,
        actor_hidden_dims=(512, 256, 128),
        critic_hidden_dims=(512, 256, 128),
        activation: str = "elu",
        init_noise_std=0.25,
    ):
        super().__init__(
            num_obs=int(num_proprio) + int(latent_dim),
            num_critic_obs=int(num_critic_obs),
            num_actions=int(num_actions),
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
        )
        self.num_proprio = int(num_proprio)
        self.num_scan = int(num_scan)
        self.depth_shape = tuple(int(value) for value in depth_shape)
        self.depth_size = 1
        for value in self.depth_shape:
            self.depth_size *= value
        self.actor_observation_dim = self.num_proprio + self.num_scan + self.depth_size
        self.compact_actor_observation_dim = self.num_proprio + int(cnn_output_dim)
        self.vision_encoder = VisionEncoder(
            image_shape=self.depth_shape,
            proprio_dim=self.num_proprio,
            cnn_output_dim=int(cnn_output_dim),
            rnn_hidden_dim=int(lstm_hidden_size),
            rnn_num_layers=int(lstm_num_layers),
            rnn_output_dim=int(latent_dim),
            use_lstm=True,
        )
        self._last_latent = None
        self._last_cnn_features = None

    def _split_actor_observation(self, obs: torch.Tensor):
        if obs.shape[-1] != self.actor_observation_dim:
            raise ValueError(
                "Visual actor observation mismatch: "
                f"expected {self.actor_observation_dim}, got {obs.shape[-1]}"
            )
        proprio = obs[..., : self.num_proprio]
        depth_start = self.num_proprio + self.num_scan
        depth = obs[..., depth_start:].reshape(*obs.shape[:-1], *self.depth_shape)
        return proprio, depth

    def _split_actor_input(self, obs: torch.Tensor):
        if obs.shape[-1] == self.compact_actor_observation_dim:
            return obs[..., : self.num_proprio], obs[..., self.num_proprio :], True
        proprio, depth = self._split_actor_observation(obs)
        return proprio, depth, False

    def _set_encoder_hidden(self, hidden_states):
        if hidden_states is None or hidden_states[0] is None:
            self.vision_encoder.reset_hidden_state()
            return
        self.vision_encoder.set_hidden_state(
            tuple(state.contiguous() for state in hidden_states)
        )

    def _encode_sequence(
        self,
        obs: torch.Tensor,
        hidden_states=None,
        masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        proprio, encoded, compact = self._split_actor_input(obs)
        time_steps, batch_size = obs.shape[:2]
        rollout_hidden = self.vision_encoder.get_hidden_state()
        self._set_encoder_hidden(hidden_states)
        if self.vision_encoder.get_hidden_state() is None:
            self.vision_encoder.reset_hidden_state(batch_size, obs.device)

        latents = []
        try:
            for step in range(time_steps):
                # observations[step] precedes dones[step], so use the prior
                # transition's mask for recurrent reset.
                step_mask = None
                if masks is not None and step > 0:
                    step_mask = masks[step - 1].reshape(batch_size)
                latents.append(
                    self.vision_encoder(
                        encoded[step], proprio[step], masks=step_mask, detach_hidden=False
                    ) if not compact else self.vision_encoder.forward_from_cnn_features(
                        encoded[step], proprio[step], masks=step_mask, detach_hidden=False
                    )
                )
            return torch.stack(latents, dim=0)
        finally:
            if rollout_hidden is None:
                self.vision_encoder.reset_hidden_state()
            else:
                self.vision_encoder.set_hidden_state(rollout_hidden)

    def _actor_features(self, obs, hidden_states=None, masks=None):
        proprio, encoded, compact = self._split_actor_input(obs)
        if obs.dim() == 3:
            latent = self._encode_sequence(obs, hidden_states=hidden_states, masks=masks)
        elif obs.dim() == 2:
            if compact:
                latent = self.vision_encoder.forward_from_cnn_features(
                    encoded, proprio, masks=masks, detach_hidden=True
                )
            else:
                cnn_features = self.vision_encoder.cnn(encoded)
                self._last_cnn_features = cnn_features
                latent = self.vision_encoder.forward_from_cnn_features(
                    cnn_features, proprio, masks=masks, detach_hidden=True
                )
        else:
            raise ValueError(
                f"Visual actor expects [B,D] or [T,B,D], got {tuple(obs.shape)}"
            )
        self._last_latent = latent
        return torch.cat((proprio, latent), dim=-1)

    @property
    def last_latent(self):
        if self._last_latent is None:
            raise RuntimeError("Visual latent is unavailable before an actor forward pass")
        return self._last_latent

    @property
    def last_cnn_features(self):
        if self._last_cnn_features is None:
            raise RuntimeError("CNN features are unavailable before a full actor forward pass")
        return self._last_cnn_features

    def update_distribution(self, obs, hidden_states=None, masks=None):
        mean = self.actor(
            self._actor_features(obs, hidden_states=hidden_states, masks=masks)
        )
        self._set_distribution(mean)

    def _set_distribution(self, mean):
        if self.noise_std_type == "scalar":
            std = self.std.clamp(min=1e-6).expand_as(mean)
        else:
            std = torch.exp(self.log_std).expand_as(mean)
        self.distribution = torch.distributions.Normal(mean, std)

    def update_distribution_from_proprio_depth(
        self, proprio, depth, hidden_states=None, masks=None
    ):
        """P3 collection fast path without allocating a 57901-D adapter tensor."""
        if proprio.ndim != 2 or proprio.shape[-1] != self.num_proprio:
            raise ValueError("P3 proprio input must be [B,num_proprio]")
        if depth.shape[0] != proprio.shape[0] or tuple(depth.shape[1:]) != self.depth_shape:
            raise ValueError(
                f"P3 depth input must be [B,{self.depth_shape}], got {tuple(depth.shape)}"
            )
        if hidden_states is not None:
            self._set_encoder_hidden(hidden_states)
        if self.vision_encoder.get_hidden_state() is None:
            self.vision_encoder.reset_hidden_state(proprio.shape[0], proprio.device)
        cnn_features = self.vision_encoder.cnn(depth)
        self._last_cnn_features = cnn_features
        latent = self.vision_encoder.forward_from_cnn_features(
            cnn_features,
            proprio,
            masks=masks,
            detach_hidden=True,
        )
        self._last_latent = latent
        self._set_distribution(self.actor(torch.cat((proprio, latent), dim=-1)))

    def act_from_proprio_depth(self, proprio, depth, hidden_states=None, masks=None):
        self.update_distribution_from_proprio_depth(
            proprio, depth, hidden_states=hidden_states, masks=masks
        )
        return self.distribution.sample()

    def act(self, obs, hidden_states=None, masks=None):
        self.update_distribution(obs, hidden_states=hidden_states, masks=masks)
        return self.distribution.sample()

    def act_inference(self, obs, hidden_states=None, masks=None):
        return self.actor(
            self._actor_features(obs, hidden_states=hidden_states, masks=masks)
        )

    def _init_hidden_states(self, batch_size, device, dtype=None):
        del dtype
        self.vision_encoder.reset_hidden_state(batch_size, device)

    def get_hidden_states(self):
        return self.vision_encoder.get_hidden_state()

    def set_hidden_states(self, hidden_states):
        self._set_encoder_hidden(hidden_states)

    def reset(self, dones=None):
        if dones is None:
            self.vision_encoder.reset_hidden_state()
            return
        done_ids = torch.nonzero(dones.reshape(-1).bool(), as_tuple=False).flatten()
        if done_ids.numel() > 0:
            self.vision_encoder.reset_hidden_state_for_envs(done_ids)
