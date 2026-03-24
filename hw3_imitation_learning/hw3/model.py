"""Model definitions for SO-100 imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        """Compute training loss for a batch."""
        raise NotImplementedError

    @abc.abstractmethod
    def sample_actions(self, state: torch.Tensor) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""
        raise NotImplementedError


# TODO: Students implement ObstaclePolicy here.
class ObstaclePolicy(BasePolicy):
    """Predicts action chunks with an MSE loss.

    A simple MLP that maps a state vector to a flat action chunk
    (chunk_size * action_dim) and reshapes to (B, chunk_size, action_dim).
    """

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        self.smoothness_weight = 0.1
        self.action_deadband = 1e-3
        hidden_dim = 512
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, chunk_size * action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return predicted action chunk of shape (B, chunk_size, action_dim)."""
        batch_size = state.shape[0]
        flat_actions = self.mlp(state)
        return flat_actions.view(batch_size, self.chunk_size, self.action_dim)

    def compute_loss(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        pred = self(state)
        mse = torch.nn.functional.mse_loss(pred, action_chunk)

        # Temporal smoothness regularization over chunk dimension to reduce jitter.
        if pred.shape[1] > 1:
            delta = pred[:, 1:, :] - pred[:, :-1, :]
            smooth = (delta ** 2).mean()
            return mse + self.smoothness_weight * smooth

        return mse

    def sample_actions(self, state: torch.Tensor) -> torch.Tensor:
        out = self(state)
        # Suppress tiny oscillatory commands at inference time.
        return torch.where(out.abs() < self.action_deadband, torch.zeros_like(out), out)


# TODO: Students implement MultiTaskPolicy here.
# class MultiTaskPolicy(BasePolicy):
#     """Goal-conditioned policy for the multicube scene."""

#     def __init__(self, state_dim, action_dim, chunk_size, goal_idx_start=9, goal_idx_end=12):
#         super().__init__(state_dim, action_dim, chunk_size)
#         self.goal_idx_start = goal_idx_start
#         self.goal_idx_end = goal_idx_end
#         self.smoothness_weight = 0.1
#         self.action_deadband = 1e-3
#         hidden_dim = 512
        
#         # shared encoder
#         self.encoder = nn.Sequential(
#             nn.Linear(state_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.ReLU(),
#         )
        
#         # separate head per color (red=0, green=1, blue=2)
#         self.heads = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(hidden_dim, hidden_dim),
#                 nn.LayerNorm(hidden_dim),
#                 nn.ReLU(),
#                 nn.Linear(hidden_dim, chunk_size * action_dim),
#             )
#             for _ in range(3)
#         ])

#     def forward(self, state):
#         goal = state[:, self.goal_idx_start:self.goal_idx_end]  # now reads 9:12
#         color_idx = goal.argmax(dim=-1)
#         features = self.encoder(state)
#         out = torch.zeros(state.shape[0], self.chunk_size * self.action_dim,
#                         device=state.device)
#         for i in range(3):
#             mask = (color_idx == i)
#             if mask.any():
#                 out[mask] = self.heads[i](features[mask])
#         return out.view(state.shape[0], self.chunk_size, self.action_dim)


#     def compute_loss(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
#         pred = self(state)
#         mse = torch.nn.functional.mse_loss(pred, action_chunk)

#         if pred.shape[1] > 1:
#             delta = pred[:, 1:, :] - pred[:, :-1, :]
#             smooth = (delta ** 2).mean()
#             return mse + self.smoothness_weight * smooth

#         return mse

#     def sample_actions(self, state: torch.Tensor) -> torch.Tensor:
#         out = self(state)
#         return torch.where(out.abs() < self.action_deadband, torch.zeros_like(out), out)

#     # def forward(self, state: torch.Tensor) -> torch.Tensor:
#     #     """Return predicted action chunk of shape (B, chunk_size, action_dim)."""
#     #     batch_size = state.shape[0]
#     #     flat_actions = self.mlp(state)
#     #     return flat_actions.view(batch_size, self.chunk_size, self.action_dim)

class MultiTaskPolicy(BasePolicy):
    def __init__(self, state_dim, action_dim, chunk_size, goal_idx_start=9, goal_idx_end=12):
        super().__init__(state_dim, action_dim, chunk_size)
        self.goal_idx_start = goal_idx_start
        self.goal_idx_end = goal_idx_end
        hidden_dim = 512
        self.smoothness_weight = 0.1
        self.action_deadband = 1e-3

        # encode goal separately and project to hidden dim
        self.goal_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
        )

        # encode state separately
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # combined trunk takes concatenated state+goal features
        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, chunk_size * action_dim),
        )

    def forward(self, state):
        goal = state[:, self.goal_idx_start:self.goal_idx_end]
        goal_features = self.goal_encoder(goal)
        state_features = self.state_encoder(state)
        combined = torch.cat([state_features, goal_features], dim=-1)
        out = self.trunk(combined)
        return out.view(state.shape[0], self.chunk_size, self.action_dim)

    def compute_loss(self, state, action_chunk):
        pred = self(state)
        mse = torch.nn.functional.mse_loss(pred, action_chunk)
        if pred.shape[1] > 1:
            delta = pred[:, 1:, :] - pred[:, :-1, :]
            smooth = (delta ** 2).mean()
            return mse + self.smoothness_weight * smooth
        return mse

    def sample_actions(self, state):
        out = self(state)
        return torch.where(out.abs() < self.action_deadband, 
                          torch.zeros_like(out), out)
    
PolicyType: TypeAlias = Literal["obstacle", "multitask"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    d_model: int | None = None,
    depth: int | None = None,
) -> BasePolicy:
    if policy_type == "obstacle":
        return ObstaclePolicy(
            action_dim=action_dim,
            state_dim=state_dim,
            chunk_size=chunk_size,
            # TODO: Build with your chosen specifications
        )
    if policy_type == "multitask":
        return MultiTaskPolicy(
            action_dim=action_dim,
            state_dim=state_dim,
            chunk_size=chunk_size,
            # TODO: Build with your chosen specifications
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
