"""Heuristic bidding scorer used when explicit bidding training is disabled.

This module keeps the `predict_env(cards)` API expected by the main
environment while living directly in the maintained env package.
"""

from __future__ import annotations

import os

import torch
from torch import nn


class _BidHeuristicNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(60, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, 512)
        self.fc6 = nn.Linear(512, 1)
        self.dropout5 = nn.Dropout(0.5)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout1 = nn.Dropout(0.1)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        out = self.fc1(feature)
        out = torch.relu(self.dropout1(self.fc2(out)))
        out = torch.relu(self.dropout3(self.fc3(out)))
        out = torch.relu(self.dropout5(self.fc4(out)))
        out = torch.relu(self.dropout5(self.fc5(out)))
        out = self.fc6(out)
        return out


def _env_to_onehot(cards: list[int]) -> torch.Tensor:
    env_to_idx = {
        3: 0,
        4: 1,
        5: 2,
        6: 3,
        7: 4,
        8: 5,
        9: 6,
        10: 7,
        11: 8,
        12: 9,
        13: 10,
        14: 11,
        17: 12,
        20: 13,
        30: 14,
    }
    mapped = [env_to_idx[card] for card in cards]
    onehot = torch.zeros((4, 15), dtype=torch.float32)
    for index in range(15):
        onehot[: mapped.count(index), index] = 1
    return onehot


def _default_weights_path() -> str:
    module_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(module_dir, "bid_weights.pkl")


_MODEL = _BidHeuristicNet()
_MODEL.eval()
_BID_WEIGHTS_PATH = os.environ.get("AT_DEC_POS_BID_WEIGHTS", _default_weights_path())
if os.path.exists(_BID_WEIGHTS_PATH):
    state = torch.load(_BID_WEIGHTS_PATH, map_location="cpu")
    _MODEL.load_state_dict(state)


def predict_env(cards: list[int]) -> float:
    feature = torch.flatten(_env_to_onehot(cards))
    score = _MODEL(feature)
    return float(score[0].item() * 100)
