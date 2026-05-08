from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Sequence, Type

import numpy as np

from .baselines import (
    AlmgrenChrissPolicy,
    OptionBenchmarkPolicy,
    POVPolicy,
    TWAPPolicy,
    VWAPPolicy,
)
from .config import OptionExecutionConfig, OptionTrainingConfig
from .env import MultiOptionExecutionEnv
from .solver import ATDecOptionSolver, IndependentPPOSolver, OptionPPOSolver

HIGHER_IS_BETTER = {
    "implementation_shortfall": False,
    "slippage": False,
    "completion_rate": True,
    "risk_adjusted_pnl": True,
    "inventory_penalty": False,
    "reward": True,
    "sender_accuracy": True,
    "receiver_accuracy": True,
    "action_public_mi_lb": True,
    "intent_public_mi_lb": True,
}
PAIRWISE_KEYS = (
    "implementation_shortfall",
    "completion_rate",
    "risk_adjusted_pnl",
    "reward",
)
SOLVER_KEY_TO_CLASS: Dict[str, Type[OptionPPOSolver]] = {
    "ppo": IndependentPPOSolver,
    "full": ATDecOptionSolver,
}


def evaluate_policy_episodes(
    policy: OptionBenchmarkPolicy,
    env_config: OptionExecutionConfig,
    episodes: int = 30,
    seed: int = 7,
    corrupt_public_trace: bool = False,
) -> List[Dict[str, float]]:
    env = MultiOptionExecutionEnv(env_config)
    metrics: List[Dict[str, float]] = []
    for episode_idx in range(episodes):
        observations = env.reset(seed=seed + episode_idx)
        done = False
        while not done:
            policy_observations = (
                env.apply_public_trace_intervention(observations)
                if corrupt_public_trace
                else observations
            )
            actions = [
                int(policy.act(env, agent_idx, obs))
                for agent_idx, obs in enumerate(policy_observations)
            ]
            observations, _, done, info = env.step(actions)
        metrics.append(info["episode_metrics"])
    return metrics


def evaluate_policy(
    policy: OptionBenchmarkPolicy,
    env_config: OptionExecutionConfig,
    episodes: int = 30,
    seed: int = 7,
    corrupt_public_trace: bool = False,
) -> Dict[str, float]:
    metrics = evaluate_policy_episodes(
        policy,
        env_config,
        episodes=episodes,
        seed=seed,
        corrupt_public_trace=corrupt_public_trace,
    )
    return {
        key: float(np.mean([metric[key] for metric in metrics]))
        for key in metrics[0]
    }


def _seed_schedule(base_seed: int, num_seeds: int) -> List[int]:
    return [base_seed + 1000 * idx for idx in range(max(num_seeds, 1))]


def _stable_offset(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    return sum((idx + 1) * ord(char) for idx, char in enumerate(text)) % 10007


def _bootstrap_mean_ci(
    values: Sequence[float],
    confidence_level: float,
    bootstrap_samples: int,
    seed: int,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return (float("nan"), float("nan"))
    if array.size == 1:
        return (float(array[0]), float(array[0]))
    rng = np.random.default_rng(seed)
    draws = rng.choice(array, size=(bootstrap_samples, array.size), replace=True).mean(axis=1)
    alpha = max(0.0, min(1.0, 1.0 - confidence_level)) / 2.0
    low, high = np.quantile(draws, [alpha, 1.0 - alpha])
    return float(low), float(high)


def _paired_randomization_pvalue(
    method_values: Sequence[float],
    baseline_values: Sequence[float],
    *,
    higher_is_better: bool,
    seed: int,
    samples: int = 5000,
) -> tuple[float, float]:
    method = np.asarray(method_values, dtype=np.float64)
    baseline = np.asarray(baseline_values, dtype=np.float64)
    if method.shape != baseline.shape:
        raise ValueError("paired significance test requires equal-length inputs")
    if method.size == 0:
        return (float("nan"), float("nan"))
    signed_delta = method - baseline if higher_is_better else baseline - method
    observed = float(np.mean(signed_delta))
    if signed_delta.size == 1 or np.allclose(signed_delta, 0.0):
        return (observed, 1.0)
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=(samples, signed_delta.size))
    null_distribution = np.mean(signs * signed_delta[None, :], axis=1)
    pvalue = float((np.sum(np.abs(null_distribution) >= abs(observed)) + 1) / (samples + 1))
    return observed, pvalue


def _summarize_method(
    method: str,
    seed_metrics: Sequence[Dict[str, float]],
    *,
    confidence_level: float,
    bootstrap_samples: int,
    seed: int,
    corrupted_seed_metrics: Optional[Sequence[Dict[str, float]]] = None,
    baseline_metrics: Optional[Sequence[Dict[str, float]]] = None,
    baseline_name: Optional[str] = None,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "method": method,
        "num_seeds": len(seed_metrics),
    }
    metric_keys = seed_metrics[0].keys()
    for metric_key in metric_keys:
        samples = [record[metric_key] for record in seed_metrics]
        summary[metric_key] = float(np.mean(samples))
        summary[f"{metric_key}_std"] = float(np.std(samples, ddof=0))
        low, high = _bootstrap_mean_ci(
            samples,
            confidence_level=confidence_level,
            bootstrap_samples=bootstrap_samples,
            seed=seed + _stable_offset(method, metric_key),
        )
        summary[f"{metric_key}_ci_low"] = low
        summary[f"{metric_key}_ci_high"] = high

    if corrupted_seed_metrics is not None:
        reward_drop = [
            normal["reward"] - corrupted["reward"]
            for normal, corrupted in zip(seed_metrics, corrupted_seed_metrics)
        ]
        pnl_drop = [
            normal["risk_adjusted_pnl"] - corrupted["risk_adjusted_pnl"]
            for normal, corrupted in zip(seed_metrics, corrupted_seed_metrics)
        ]
        shortfall_increase = [
            corrupted["implementation_shortfall"] - normal["implementation_shortfall"]
            for normal, corrupted in zip(seed_metrics, corrupted_seed_metrics)
        ]
        for metric_key, samples in (
            ("public_trace_reward_drop", reward_drop),
            ("public_trace_pnl_drop", pnl_drop),
            ("public_trace_shortfall_increase", shortfall_increase),
        ):
            summary[metric_key] = float(np.mean(samples))
            low, high = _bootstrap_mean_ci(
                samples,
                confidence_level=confidence_level,
                bootstrap_samples=bootstrap_samples,
                seed=seed + _stable_offset(method, metric_key, "corruption"),
            )
            summary[f"{metric_key}_ci_low"] = low
            summary[f"{metric_key}_ci_high"] = high

    if baseline_metrics is not None and baseline_name is not None and len(seed_metrics) == len(baseline_metrics):
        summary["comparison_baseline"] = baseline_name
        for metric_key in PAIRWISE_KEYS:
            delta, pvalue = _paired_randomization_pvalue(
                [record[metric_key] for record in seed_metrics],
                [record[metric_key] for record in baseline_metrics],
                higher_is_better=HIGHER_IS_BETTER[metric_key],
                seed=seed + _stable_offset(method, baseline_name, metric_key, "pairwise"),
            )
            summary[f"{metric_key}_delta_vs_baseline"] = float(delta)
            summary[f"{metric_key}_p_vs_baseline"] = float(pvalue)

    return summary


def _evaluate_solver_seed(
    solver_cls: Type[OptionPPOSolver],
    env_config: OptionExecutionConfig,
    training_config: OptionTrainingConfig,
    *,
    device: Optional[str] = None,
    seed: int,
    include_public_intervention: bool,
) -> tuple[Dict[str, float], Optional[Dict[str, float]]]:
    solver = solver_cls(deepcopy(env_config), deepcopy(training_config), device=device, seed=seed)
    solver.train(training_config.episodes)
    normal = solver.evaluate(training_config.eval_episodes)
    corrupted = None
    if include_public_intervention:
        corrupted = solver.evaluate(
            training_config.eval_episodes,
            corrupt_public_trace=True,
        )
    return normal, corrupted


def resolve_solver_class(solver: str) -> Type[OptionPPOSolver]:
    try:
        return SOLVER_KEY_TO_CLASS[solver]
    except KeyError as exc:
        supported = ", ".join(sorted(SOLVER_KEY_TO_CLASS))
        raise ValueError(f"unsupported solver '{solver}'. Expected one of: {supported}") from exc


def run_solver_research_eval(
    solver: str,
    env_config: Optional[OptionExecutionConfig] = None,
    training_config: Optional[OptionTrainingConfig] = None,
    device: Optional[str] = None,
    seed: int = 7,
    num_seeds: int = 3,
    confidence_level: float = 0.95,
    bootstrap_samples: int = 2000,
    include_public_intervention: bool = True,
) -> Dict[str, Any]:
    """Run one solver under a fixed research budget and aggregate across seeds."""

    env_conf = deepcopy(env_config or OptionExecutionConfig(seed=seed))
    train_conf = deepcopy(training_config or OptionTrainingConfig())
    seeds = _seed_schedule(seed, num_seeds)
    solver_cls = resolve_solver_class(solver)
    method_name = solver_cls(deepcopy(env_conf), deepcopy(train_conf), device=device, seed=seed).name

    seed_records: List[Dict[str, float]] = []
    seed_corrupted_records: List[Dict[str, float]] = []
    for run_seed in seeds:
        normal, corrupted = _evaluate_solver_seed(
            solver_cls,
            deepcopy(env_conf),
            deepcopy(train_conf),
            device=device,
            seed=run_seed,
            include_public_intervention=include_public_intervention,
        )
        seed_records.append(normal)
        if corrupted is not None:
            seed_corrupted_records.append(corrupted)

    summary = _summarize_method(
        method_name,
        seed_records,
        confidence_level=confidence_level,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        corrupted_seed_metrics=seed_corrupted_records if include_public_intervention else None,
    )
    return {
        "solver": solver,
        "method": method_name,
        "seed_schedule": seeds,
        "summary": summary,
        "seed_metrics": seed_records,
        "corrupted_seed_metrics": seed_corrupted_records if include_public_intervention else [],
    }


def run_benchmark(
    env_config: Optional[OptionExecutionConfig] = None,
    training_config: Optional[OptionTrainingConfig] = None,
    device: Optional[str] = None,
    seed: int = 7,
    num_seeds: int = 5,
    confidence_level: float = 0.95,
    bootstrap_samples: int = 2000,
    include_public_intervention: bool = True,
    comparison_baseline: str = "Independent PPO",
) -> List[Dict[str, Any]]:
    env_conf = deepcopy(env_config or OptionExecutionConfig(seed=seed))
    train_conf = deepcopy(training_config or OptionTrainingConfig())
    seeds = _seed_schedule(seed, num_seeds)

    method_metrics: Dict[str, List[Dict[str, float]]] = {}
    corrupted_method_metrics: Dict[str, List[Dict[str, float]]] = {}

    heuristic_factories: Sequence[Callable[[], OptionBenchmarkPolicy]] = (
        TWAPPolicy,
        VWAPPolicy,
        POVPolicy,
        AlmgrenChrissPolicy,
    )
    for factory in heuristic_factories:
        method_name = factory().name
        seed_records: List[Dict[str, float]] = []
        seed_corrupted_records: List[Dict[str, float]] = []
        for run_seed in seeds:
            seed_records.append(
                evaluate_policy(
                    factory(),
                    deepcopy(env_conf),
                    episodes=train_conf.eval_episodes,
                    seed=run_seed,
                )
            )
            if include_public_intervention:
                seed_corrupted_records.append(
                    evaluate_policy(
                        factory(),
                        deepcopy(env_conf),
                        episodes=train_conf.eval_episodes,
                        seed=run_seed,
                        corrupt_public_trace=True,
                    )
                )
        method_metrics[method_name] = seed_records
        if include_public_intervention:
            corrupted_method_metrics[method_name] = seed_corrupted_records

    for solver_cls in (IndependentPPOSolver, ATDecOptionSolver):
        method_name = solver_cls(deepcopy(env_conf), deepcopy(train_conf), device=device, seed=seed).name
        seed_records = []
        seed_corrupted_records: List[Dict[str, float]] = []
        for run_seed in seeds:
            normal, corrupted = _evaluate_solver_seed(
                solver_cls,
                deepcopy(env_conf),
                deepcopy(train_conf),
                device=device,
                seed=run_seed,
                include_public_intervention=include_public_intervention,
            )
            seed_records.append(normal)
            if corrupted is not None:
                seed_corrupted_records.append(corrupted)
        method_metrics[method_name] = seed_records
        if include_public_intervention:
            corrupted_method_metrics[method_name] = seed_corrupted_records

    baseline_metrics = method_metrics.get(comparison_baseline)
    results: List[Dict[str, Any]] = []
    for method_name, seed_records in method_metrics.items():
        results.append(
            _summarize_method(
                method_name,
                seed_records,
                confidence_level=confidence_level,
                bootstrap_samples=bootstrap_samples,
                seed=seed,
                corrupted_seed_metrics=corrupted_method_metrics.get(method_name),
                baseline_metrics=baseline_metrics if method_name != comparison_baseline else None,
                baseline_name=comparison_baseline if method_name != comparison_baseline else None,
            )
        )

    preferred_order = {
        "TWAP": 0,
        "VWAP": 1,
        "POV": 2,
        "Almgren-Chriss": 3,
        "Independent PPO": 4,
        "AT-Dec Option Solver": 5,
    }
    results.sort(key=lambda record: preferred_order.get(str(record["method"]), 99))
    return results


def _format_metric_cell(record: Dict[str, Any], key: str) -> str:
    value = float(record[key])
    low = record.get(f"{key}_ci_low")
    high = record.get(f"{key}_ci_high")
    if low is None or high is None or int(record.get("num_seeds", 1)) <= 1:
        return f"{value:.3f}"
    return f"{value:.3f} [{float(low):.3f}, {float(high):.3f}]"


def format_markdown_table(records: List[Dict[str, Any]]) -> str:
    include_seed_count = any("num_seeds" in record for record in records)
    include_public_intervention = any("public_trace_reward_drop" in record for record in records)
    include_significance = any("reward_p_vs_baseline" in record for record in records)
    include_coordination_info = any("action_public_mi_lb" in record for record in records)

    header = ["Method"]
    if include_seed_count:
        header.append("Seeds")
    header.extend(
        [
            "Shortfall",
            "Slippage",
            "Completion",
            "Risk-adjusted PnL",
            "Reward",
        ]
    )
    if include_public_intervention:
        header.append("Delta_pub Reward")
    if include_coordination_info:
        header.extend(["MI(a,pub)", "MI(z,pub)"])
    if include_significance:
        baseline_name = next(
            (
                str(record["comparison_baseline"])
                for record in records
                if "comparison_baseline" in record
            ),
            "baseline",
        )
        header.append(f"p(Reward vs {baseline_name})")

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] + ["---:" for _ in header[1:]]) + " |",
    ]
    for record in records:
        row = [str(record["method"])]
        if include_seed_count:
            row.append(str(int(record.get("num_seeds", 1))))
        row.extend(
            [
                _format_metric_cell(record, "implementation_shortfall"),
                _format_metric_cell(record, "slippage"),
                _format_metric_cell(record, "completion_rate"),
                _format_metric_cell(record, "risk_adjusted_pnl"),
                _format_metric_cell(record, "reward"),
            ]
        )
        if include_public_intervention:
            if "public_trace_reward_drop" in record:
                row.append(_format_metric_cell(record, "public_trace_reward_drop"))
            else:
                row.append("-")
        if include_coordination_info:
            row.append("-" if "action_public_mi_lb" not in record else _format_metric_cell(record, "action_public_mi_lb"))
            row.append("-" if "intent_public_mi_lb" not in record else _format_metric_cell(record, "intent_public_mi_lb"))
        if include_significance:
            pvalue = record.get("reward_p_vs_baseline")
            row.append("-" if pvalue is None else f"{float(pvalue):.4f}")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)
