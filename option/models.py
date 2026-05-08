from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ATDecOptionModel(nn.Module):
    """
    Shared option-execution model.

    Execution path:
        actor(obs) depends only on local observation + public tape.
    Training-only heads:
        privileged_value, belief, sender, receiver.
    """

    def __init__(
        self,
        obs_dim: int,
        privileged_dim: int,
        action_dim: int,
        belief_dim: int,
        hidden_dim: int = 128,
        urgency_classes: int = 4,
        coordination_dim: int = 32,
    ) -> None:
        super().__init__()
        self.obs_encoder = MLP(obs_dim, hidden_dim, hidden_dim)
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

        self.privileged_encoder = MLP(privileged_dim, hidden_dim, hidden_dim)
        self.teacher_value_head = nn.Linear(hidden_dim, 1)

        self.belief_head = nn.Linear(hidden_dim, belief_dim)
        self.robustness_head = nn.Linear(hidden_dim, 1)
        self.sender_head = nn.Linear(hidden_dim, urgency_classes)
        self.receiver_head = nn.Linear(hidden_dim, urgency_classes)
        self.sender_projection = nn.Linear(hidden_dim, coordination_dim)
        self.receiver_projection = nn.Linear(hidden_dim, coordination_dim)
        self.action_embedding = nn.Embedding(action_dim, coordination_dim)

    def encode_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return self.obs_encoder(obs)

    def actor(self, obs: torch.Tensor) -> torch.Tensor:
        hidden = self.encode_obs(obs)
        return self.actor_head(hidden)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        hidden = self.encode_obs(obs)
        return self.value_head(hidden)

    def privileged_value(self, privileged: torch.Tensor) -> torch.Tensor:
        hidden = self.privileged_encoder(privileged)
        return self.teacher_value_head(hidden)

    def belief(self, obs: torch.Tensor) -> torch.Tensor:
        hidden = self.encode_obs(obs)
        return self.belief_head(hidden)

    def robustness(self, obs: torch.Tensor) -> torch.Tensor:
        hidden = self.encode_obs(obs)
        return self.robustness_head(hidden)

    def coordination(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encode_obs(obs)
        return self.sender_head(hidden), self.receiver_head(hidden)

    def coordination_latents(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encode_obs(obs)
        return self.sender_projection(hidden), self.receiver_projection(hidden)

    def embed_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return self.action_embedding(actions.long())
