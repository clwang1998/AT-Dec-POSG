from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import OptionExecutionConfig, OptionMarketScenario


class MultiOptionExecutionEnv:
    """
    Synthetic multi-option execution game with AT-Dec-POSG information structure.

    Team side:
        One execution agent per option contract.
    Adversarial side:
        Aggregate liquidity response encoded through market state dynamics.
    Public channel:
        The shared trade tape, i.e. visible fills/aggressiveness from all contracts.
    """

    def __init__(
        self,
        config: OptionExecutionConfig,
        scenario: Optional[OptionMarketScenario] = None,
    ) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.default_scenario = scenario or OptionMarketScenario()
        self.scenario = self.default_scenario
        self.public_history: deque[np.ndarray] = deque(
            maxlen=self.config.public_tape_window
        )
        self.reset()

    def reset(
        self,
        seed: Optional[int] = None,
        scenario: Optional[OptionMarketScenario] = None,
    ) -> List[np.ndarray]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.scenario = scenario or self.default_scenario
        self.t = 0
        self.initial_targets = self._sample_targets()
        self.remaining = self.initial_targets.copy()
        noise = self.rng.normal(0.0, 0.35, size=self.config.num_contracts)
        self.reference_prices = (
            self.config.base_mid_price
            + np.linspace(-2.0, 2.0, self.config.num_contracts)
            + noise
        )
        self.mid_prices = self.reference_prices.copy()
        self.spreads = np.full(self.config.num_contracts, self.config.base_spread)
        self.ivs = np.full(self.config.num_contracts, self.config.base_iv) + self.rng.normal(
            0.0, 0.015, size=self.config.num_contracts
        )
        self.hidden_liquidity = self.rng.uniform(0.35, 0.95, size=self.config.num_contracts)
        self.pressure = self.rng.uniform(0.05, 0.25, size=self.config.num_contracts)
        self.last_local_flow = np.zeros(self.config.num_contracts, dtype=np.float32)
        self.total_filled = np.zeros(self.config.num_contracts, dtype=np.float32)
        self.cumulative_shortfall = 0.0
        self.cumulative_inventory_penalty = 0.0
        self.cumulative_risk_penalty = 0.0
        self.cumulative_reward = 0.0
        self.public_history.clear()
        for _ in range(self.config.public_tape_window):
            self.public_history.append(np.zeros(self.config.num_contracts * 3, dtype=np.float32))
        return self._build_observations()

    def step(
        self,
        actions: Sequence[int],
    ) -> Tuple[List[np.ndarray], float, bool, Dict[str, object]]:
        if len(actions) != self.config.num_contracts:
            raise ValueError(
                f"expected {self.config.num_contracts} actions, got {len(actions)}"
            )

        public_entry = np.zeros((self.config.num_contracts, 3), dtype=np.float32)
        step_shortfall = 0.0
        for idx, action_idx in enumerate(actions):
            template = self.config.templates[int(action_idx)]
            fill_qty, fill_rate, exec_shortfall = self._execute_template(idx, template)
            public_entry[idx, 0] = fill_qty / max(abs(self.initial_targets[idx]), 1.0)
            public_entry[idx, 1] = template.aggressiveness
            public_entry[idx, 2] = fill_rate
            step_shortfall += exec_shortfall

        self.public_history.append(public_entry.reshape(-1))
        self._advance_market(public_entry)

        inventory_penalty = self.config.inventory_penalty * float(
            np.mean(np.abs(self.remaining) / np.maximum(np.abs(self.initial_targets), 1.0))
        )
        risk_penalty = self.config.risk_penalty * float(
            np.mean(np.abs(self.remaining) * self.ivs) / max(self.config.target_scale, 1.0)
        )
        self.cumulative_shortfall += step_shortfall
        self.cumulative_inventory_penalty += inventory_penalty
        self.cumulative_risk_penalty += risk_penalty

        self.t += 1
        done = self.t >= self.config.horizon or bool(np.all(np.abs(self.remaining) <= 0.05))
        final_penalty = 0.0
        if done:
            final_penalty = self.config.non_completion_penalty * float(
                np.sum(np.abs(self.remaining))
                / max(np.sum(np.abs(self.initial_targets)), 1.0)
            )

        reward = -(step_shortfall + inventory_penalty + risk_penalty + final_penalty)
        self.cumulative_reward += reward
        observations = self._build_observations()
        info: Dict[str, object] = {
            "team_shortfall": step_shortfall,
            "inventory_penalty": inventory_penalty,
            "risk_penalty": risk_penalty,
            "final_penalty": final_penalty,
            "privileged_state": self.get_privileged_state(),
            "belief_target": self.get_belief_target(),
            "urgency_targets": self.get_urgency_targets(),
            "scenario_name": self.scenario.name,
            "scenario_role": self.scenario.role,
            "coordination_stress": self.get_coordination_stress(),
        }
        if done:
            info["episode_metrics"] = self.get_metrics()
        return observations, reward, done, info

    def get_privileged_state(self) -> np.ndarray:
        per_contract = np.stack(
            [
                (self.mid_prices - self.reference_prices) / np.maximum(self.reference_prices, 1.0),
                self.spreads / max(self.config.base_spread, 1e-6),
                self.ivs,
                self.hidden_liquidity,
                self.pressure / 2.0,
                self.remaining / np.maximum(np.abs(self.initial_targets), 1.0),
            ],
            axis=1,
        ).reshape(-1)
        return np.concatenate(
            [
                per_contract.astype(np.float32),
                np.array([self.time_ratio], dtype=np.float32),
                self._public_tape_features(),
            ]
        )

    def get_belief_target(self) -> np.ndarray:
        return np.concatenate(
            [self.hidden_liquidity, np.clip(self.pressure / 2.0, 0.0, 1.0)]
        ).astype(np.float32)

    def get_urgency_targets(self) -> np.ndarray:
        time_left = max(self.config.horizon - self.t, 1)
        urgency = np.abs(self.remaining) / np.maximum(np.abs(self.initial_targets), 1.0)
        urgency = urgency / time_left * self.config.horizon
        return np.clip(urgency, 0.0, 1.0).astype(np.float32)

    def get_coordination_stress(self) -> float:
        liquidity_stress = float(np.mean(1.0 - self.hidden_liquidity))
        pressure_stress = float(np.mean(np.clip(self.pressure / 2.0, 0.0, 1.0)))
        return float(np.clip(0.5 * liquidity_stress + 0.5 * pressure_stress, 0.0, 1.0))

    def get_metrics(self) -> Dict[str, float]:
        total_abs_target = max(float(np.sum(np.abs(self.initial_targets))), 1.0)
        total_abs_filled = max(float(np.sum(np.abs(self.total_filled))), 1.0)
        completion_rate = 1.0 - float(np.sum(np.abs(self.remaining))) / total_abs_target
        slippage = self.cumulative_shortfall / total_abs_filled
        volatility_norm = 1.0 + float(np.std(self.mid_prices / np.maximum(self.reference_prices, 1.0)))
        risk_adjusted_pnl = -(
            self.cumulative_shortfall + self.cumulative_risk_penalty
        ) / volatility_norm
        return {
            "implementation_shortfall": float(self.cumulative_shortfall),
            "slippage": float(slippage),
            "completion_rate": float(np.clip(completion_rate, 0.0, 1.0)),
            "risk_adjusted_pnl": float(risk_adjusted_pnl),
            "inventory_penalty": float(self.cumulative_inventory_penalty),
            "reward": float(self.cumulative_reward),
        }

    def decode_local_observation(self, observation: np.ndarray) -> Dict[str, float]:
        local = observation[: self.config.local_feature_dim]
        public = observation[self.config.local_feature_dim :]
        public_flow = public.reshape(self.config.public_tape_window, self.config.num_contracts, 3)
        return {
            "mid_return": float(local[0]),
            "spread_ratio": float(local[1]),
            "iv": float(local[2]),
            "remaining_ratio": float(local[3]),
            "completion_ratio": float(local[4]),
            "time_ratio": float(local[5]),
            "local_flow": float(local[6]),
            "public_activity": float(np.mean(np.abs(public_flow[:, :, 0]))),
            "public_aggressiveness": float(np.mean(public_flow[:, :, 1])),
        }

    @property
    def time_ratio(self) -> float:
        return max(self.config.horizon - self.t, 0) / max(self.config.horizon, 1)

    @property
    def public_tape_slice(self) -> slice:
        return slice(self.config.local_feature_dim, self.config.obs_dim)

    def public_tape_view(self, observation: np.ndarray) -> np.ndarray:
        return observation[self.public_tape_slice].reshape(
            self.config.public_tape_window,
            self.config.num_contracts,
            3,
        )

    def corrupt_teammate_public_trace(
        self,
        observation: np.ndarray,
        agent_idx: int,
        fill_value: float = 0.0,
    ) -> np.ndarray:
        """
        Corrupt only teammate-originating public-trace features.

        This mirrors the paper protocol: local state stays intact, the agent's own
        public trace stays visible, and only teammate-attributed public tape is removed.
        """

        corrupted = np.array(observation, copy=True)
        public = self.public_tape_view(corrupted)
        teammate_mask = np.ones(self.config.num_contracts, dtype=bool)
        teammate_mask[agent_idx] = False
        public[:, teammate_mask, :] = fill_value
        return corrupted.astype(np.float32)

    def apply_public_trace_intervention(
        self,
        observations: Sequence[np.ndarray],
        fill_value: float = 0.0,
    ) -> List[np.ndarray]:
        return [
            self.corrupt_teammate_public_trace(observation, agent_idx, fill_value=fill_value)
            for agent_idx, observation in enumerate(observations)
        ]

    def _sample_targets(self) -> np.ndarray:
        magnitudes = self.rng.integers(
            low=max(4, int(self.config.target_scale * 0.5)),
            high=max(6, int(self.config.target_scale * 1.5)),
            size=self.config.num_contracts,
        ).astype(np.float32)
        directions = self.rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=self.config.num_contracts)
        return magnitudes * directions

    def _execute_template(
        self,
        idx: int,
        template,
    ) -> Tuple[float, float, float]:
        remaining = self.remaining[idx]
        if abs(remaining) <= 0.05 or template.size_fraction <= 0.0:
            self.last_local_flow[idx] = 0.0
            return 0.0, 0.0, 0.0

        direction = np.sign(remaining)
        desired_abs = min(
            abs(remaining),
            max(1.0, abs(self.initial_targets[idx]) * template.size_fraction),
        )
        liquidity_term = 0.25 * self.hidden_liquidity[idx]
        spread_term = 0.18 * self.spreads[idx] / max(self.config.base_spread, 1e-6)
        fill_noise = self.rng.normal(0.0, 0.05)
        fill_rate = np.clip(
            0.08 + 0.70 * template.aggressiveness + liquidity_term - spread_term + fill_noise,
            0.0,
            1.0,
        )
        fill_qty = float(direction * desired_abs * fill_rate)
        impact = (
            self.config.impact_coeff
            * self.scenario.impact_multiplier
            * abs(fill_qty)
            / max(abs(self.initial_targets[idx]), 1.0)
        )
        execution_offset = direction * (
            0.5 * self.spreads[idx] * (0.1 + template.aggressiveness)
            + impact
            + 0.05 * self.pressure[idx]
        )
        execution_price = self.mid_prices[idx] + execution_offset
        shortfall = float(fill_qty * (execution_price - self.reference_prices[idx]))
        self.remaining[idx] -= fill_qty
        self.total_filled[idx] += fill_qty
        self.last_local_flow[idx] = fill_qty / max(abs(self.initial_targets[idx]), 1.0)
        return fill_qty, float(fill_rate), shortfall

    def _advance_market(self, public_entry: np.ndarray) -> None:
        team_activity = float(np.mean(np.abs(public_entry[:, 0])))
        signed_pressure = float(np.mean(public_entry[:, 0]))
        vol_noise = self.rng.normal(0.0, 0.08, size=self.config.num_contracts)
        self.mid_prices = self.mid_prices + (
            self.scenario.volatility_multiplier * vol_noise
            + 0.12 * signed_pressure
        )
        spread_noise = self.rng.normal(0.0, 0.015, size=self.config.num_contracts)
        self.spreads = np.clip(
            self.config.base_spread * self.scenario.spread_multiplier
            + 0.18 * team_activity
            + 0.10 * (1.0 - self.hidden_liquidity)
            + spread_noise,
            0.08,
            2.50,
        )
        liquidity_noise = self.rng.normal(0.0, 0.05, size=self.config.num_contracts)
        self.hidden_liquidity = np.clip(
            0.80 * self.hidden_liquidity
            + 0.20 * self.rng.uniform(0.25, 0.95, size=self.config.num_contracts)
            - self.scenario.liquidity_decay * team_activity
            + liquidity_noise,
            0.05,
            1.00,
        )
        pressure_noise = self.rng.normal(0.0, 0.04, size=self.config.num_contracts)
        self.pressure = np.clip(
            0.72 * self.pressure
            + 0.50 * team_activity
            + self.scenario.pressure_bias
            + pressure_noise,
            0.0,
            2.0,
        )
        iv_noise = self.rng.normal(0.0, 0.01, size=self.config.num_contracts)
        self.ivs = np.clip(
            self.ivs + 0.04 * team_activity + iv_noise,
            0.05,
            1.10,
        )

    def _public_tape_features(self) -> np.ndarray:
        return np.concatenate(list(self.public_history)).astype(np.float32)

    def _build_observations(self) -> List[np.ndarray]:
        public = self._public_tape_features()
        observations: List[np.ndarray] = []
        for idx in range(self.config.num_contracts):
            local = np.array(
                [
                    (self.mid_prices[idx] - self.reference_prices[idx])
                    / max(self.reference_prices[idx], 1.0),
                    self.spreads[idx] / max(self.config.base_spread, 1e-6),
                    self.ivs[idx],
                    self.remaining[idx] / max(abs(self.initial_targets[idx]), 1.0),
                    1.0
                    - abs(self.remaining[idx]) / max(abs(self.initial_targets[idx]), 1.0),
                    self.time_ratio,
                    self.last_local_flow[idx],
                ],
                dtype=np.float32,
            )
            observations.append(np.concatenate([local, public]).astype(np.float32))
        return observations
