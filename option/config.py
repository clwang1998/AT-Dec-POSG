from dataclasses import dataclass, field
from typing import List, Sequence, Tuple


@dataclass(frozen=True)
class ExecutionTemplate:
    """Discrete execution template exposed to each option agent."""

    name: str
    aggressiveness: float
    size_fraction: float


def default_templates() -> Tuple[ExecutionTemplate, ...]:
    return (
        ExecutionTemplate("hold", 0.0, 0.0),
        ExecutionTemplate("passive_small", 0.15, 0.25),
        ExecutionTemplate("passive_large", 0.15, 0.50),
        ExecutionTemplate("neutral_small", 0.50, 0.25),
        ExecutionTemplate("neutral_large", 0.50, 0.50),
        ExecutionTemplate("aggressive_small", 0.95, 0.25),
        ExecutionTemplate("aggressive_large", 0.95, 0.50),
    )


@dataclass(frozen=True)
class OptionMarketScenario:
    """Market-side regime sampled by the league for training or evaluation."""

    name: str = "main"
    role: str = "main"
    spread_multiplier: float = 1.0
    impact_multiplier: float = 1.0
    volatility_multiplier: float = 1.0
    pressure_bias: float = 0.0
    liquidity_decay: float = 0.08


def default_scenarios() -> List[OptionMarketScenario]:
    return [
        OptionMarketScenario(name="main_balanced", role="main"),
        OptionMarketScenario(
            name="snapshot_wide_spread",
            role="snapshot",
            spread_multiplier=1.20,
            impact_multiplier=1.05,
            volatility_multiplier=1.10,
            liquidity_decay=0.10,
        ),
        OptionMarketScenario(
            name="snapshot_fast_tape",
            role="snapshot",
            spread_multiplier=0.95,
            impact_multiplier=1.10,
            volatility_multiplier=1.25,
            pressure_bias=0.05,
        ),
        OptionMarketScenario(
            name="exploiter_stressed_liquidity",
            role="exploiter",
            spread_multiplier=1.35,
            impact_multiplier=1.35,
            volatility_multiplier=1.20,
            pressure_bias=0.15,
            liquidity_decay=0.14,
        ),
    ]


@dataclass
class OptionExecutionConfig:
    """Synthetic replay-style multi-option execution task configuration."""

    num_contracts: int = 3
    horizon: int = 32
    public_tape_window: int = 4
    base_mid_price: float = 100.0
    base_spread: float = 0.6
    base_iv: float = 0.22
    target_scale: float = 12.0
    impact_coeff: float = 0.12
    inventory_penalty: float = 0.06
    risk_penalty: float = 0.04
    non_completion_penalty: float = 1.50
    seed: int = 7
    templates: Tuple[ExecutionTemplate, ...] = field(default_factory=default_templates)

    @property
    def local_feature_dim(self) -> int:
        return 7

    @property
    def public_tape_dim(self) -> int:
        return self.public_tape_window * self.num_contracts * 3

    @property
    def obs_dim(self) -> int:
        return self.local_feature_dim + self.public_tape_dim

    @property
    def belief_dim(self) -> int:
        # Hidden liquidity and market-maker pressure for each contract.
        return self.num_contracts * 2

    @property
    def privileged_dim(self) -> int:
        # Mid, spread, iv, hidden liquidity, pressure, remaining ratio per contract
        # plus shared time-to-go and the public tape summary.
        return self.num_contracts * 6 + 1 + self.public_tape_dim

    @property
    def action_dim(self) -> int:
        return len(self.templates)


@dataclass
class OptionTrainingConfig:
    """Training knobs for the baseline PPO executor and the full AT-Dec solver."""

    episodes: int = 250
    eval_episodes: int = 30
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.20
    learning_rate: float = 3e-4
    entropy_coef: float = 0.01
    value_coef: float = 0.50
    hidden_dim: int = 128
    update_epochs: int = 4
    use_module_a: bool = True
    use_module_b: bool = True
    use_module_c: bool = True
    module_a_weight: float = 0.25
    module_b_weight: float = 0.10
    module_c_weight: float = 0.10
    module_b_coord_dim: int = 32
    module_b_mi_weight: float = 0.05
    module_b_temperature: float = 0.20
    use_boltzmann_eval: bool = False
    eval_boltzmann_temperature: float = 0.35
    use_multitask_loss_balancer: bool = True
    multitask_anchor_ppo: bool = True
    multitask_aux_scale: float = 0.25
    multitask_aux_warmup_ratio: float = 0.0
    multitask_log_var_init: float = 0.0
    multitask_log_var_min: float = -5.0
    multitask_log_var_max: float = 5.0
    completion_regularizer_weight: float = 0.0
    trace_dropout_prob: float = 0.0
    trace_consistency_weight: float = 0.0
    minimax_alpha: float = 0.12
    minimax_gamma: float = 1.0
    minimax_value_floor: float = 0.0
    league_main_prob: float = 0.5
    league_snapshot_prob: float = 0.3
    league_exploiter_prob: float = 0.2
    log_interval: int = 25

    def role_probs(self) -> Sequence[float]:
        total = (
            self.league_main_prob
            + self.league_snapshot_prob
            + self.league_exploiter_prob
        )
        if total <= 0:
            return (1.0, 0.0, 0.0)
        return (
            self.league_main_prob / total,
            self.league_snapshot_prob / total,
            self.league_exploiter_prob / total,
        )
