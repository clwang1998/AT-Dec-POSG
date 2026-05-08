from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OptionBenchmarkPolicy:
    name: str

    def act(self, env, agent_idx: int, observation: np.ndarray) -> int:
        raise NotImplementedError


def _choose_action_from_score(score: float) -> int:
    if score <= 0.05:
        return 1
    if score <= 0.20:
        return 2
    if score <= 0.38:
        return 3
    if score <= 0.58:
        return 4
    if score <= 0.78:
        return 5
    return 6


class TWAPPolicy(OptionBenchmarkPolicy):
    def __init__(self) -> None:
        super().__init__(name="TWAP")

    def act(self, env, agent_idx: int, observation: np.ndarray) -> int:
        local = env.decode_local_observation(observation)
        if abs(local["remaining_ratio"]) < 0.03:
            return 0
        time_left = max(local["time_ratio"], 1.0 / env.config.horizon)
        urgency = min(1.0, abs(local["remaining_ratio"]) / time_left * 0.5)
        return _choose_action_from_score(urgency)


class VWAPPolicy(OptionBenchmarkPolicy):
    def __init__(self) -> None:
        super().__init__(name="VWAP")

    def act(self, env, agent_idx: int, observation: np.ndarray) -> int:
        local = env.decode_local_observation(observation)
        if abs(local["remaining_ratio"]) < 0.03:
            return 0
        activity = np.clip(local["public_activity"] * 3.5, 0.0, 1.0)
        spread_drag = 0.20 * max(local["spread_ratio"] - 1.0, 0.0)
        score = np.clip(0.20 + 0.55 * activity - spread_drag, 0.0, 1.0)
        if local["time_ratio"] < 0.25:
            score = min(1.0, score + 0.20)
        return _choose_action_from_score(score)


class POVPolicy(OptionBenchmarkPolicy):
    def __init__(self) -> None:
        super().__init__(name="POV")

    def act(self, env, agent_idx: int, observation: np.ndarray) -> int:
        local = env.decode_local_observation(observation)
        if abs(local["remaining_ratio"]) < 0.03:
            return 0
        score = np.clip(0.10 + 0.80 * local["public_activity"], 0.0, 1.0)
        if local["spread_ratio"] > 1.4:
            score *= 0.7
        return _choose_action_from_score(score)


class AlmgrenChrissPolicy(OptionBenchmarkPolicy):
    def __init__(self) -> None:
        super().__init__(name="Almgren-Chriss")

    def act(self, env, agent_idx: int, observation: np.ndarray) -> int:
        local = env.decode_local_observation(observation)
        if abs(local["remaining_ratio"]) < 0.03:
            return 0
        time_left = max(local["time_ratio"], 1.0 / env.config.horizon)
        urgency = np.clip(abs(local["remaining_ratio"]) / time_left * 0.45, 0.0, 1.0)
        risk_push = np.clip(local["iv"] / max(env.config.base_iv, 1e-6) - 1.0, 0.0, 1.0)
        spread_cost = np.clip(local["spread_ratio"] - 1.0, 0.0, 1.0)
        score = np.clip(urgency + 0.25 * risk_push - 0.35 * spread_cost, 0.0, 1.0)
        return _choose_action_from_score(score)
