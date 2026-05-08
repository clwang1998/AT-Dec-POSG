from __future__ import annotations

import argparse

from .benchmark import evaluate_policy, format_markdown_table, run_benchmark
from .baselines import AlmgrenChrissPolicy, POVPolicy, TWAPPolicy, VWAPPolicy
from .config import OptionExecutionConfig, OptionTrainingConfig
from .hardware import detect_option_hardware, format_hardware_report, resolve_runtime_device
from .solver import ATDecOptionSolver, IndependentPPOSolver


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Option AT-Dec-POSG benchmark runner")
    parser.add_argument("--episodes", type=int, default=200, help="Training episodes for RL solvers")
    parser.add_argument("--eval-episodes", type=int, default=30, help="Evaluation episodes")
    parser.add_argument("--num-contracts", type=int, default=3, help="Number of option agents/contracts")
    parser.add_argument(
        "--benchmark-seeds",
        type=int,
        default=5,
        help="Number of random seeds for benchmark mode",
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        help="Confidence level for bootstrap intervals in benchmark mode",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
        help="Bootstrap samples for benchmark confidence intervals",
    )
    parser.add_argument(
        "--corrupt-public-trace",
        action="store_true",
        help="Corrupt teammate-originating public tape at evaluation time",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="full",
        choices=["full", "ppo", "twap", "vwap", "pov", "almgren", "benchmark"],
        help="Which solver or benchmark suite to run",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device for solver runs. Use 'auto', 'cpu', 'cuda', or 'cuda:N'.",
    )
    parser.add_argument(
        "--boltzmann-eval",
        action="store_true",
        help="Sample evaluation-time actions from a Boltzmann policy instead of argmax.",
    )
    parser.add_argument(
        "--eval-temperature",
        type=float,
        default=0.35,
        help="Temperature used when --boltzmann-eval is enabled.",
    )
    parser.add_argument(
        "--print-hardware",
        action="store_true",
        help="Print the detected option hardware profile before running.",
    )
    parser.add_argument(
        "--hardware-only",
        action="store_true",
        help="Print the detected option hardware profile and exit.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    hardware_profile = detect_option_hardware(args.device)
    if args.print_hardware or args.hardware_only:
        print(format_hardware_report(hardware_profile))
        print(hardware_profile["analysis"])
    if args.hardware_only:
        return

    selected_device = resolve_runtime_device(args.device)
    env_config = OptionExecutionConfig(num_contracts=args.num_contracts, seed=args.seed)
    training_config = OptionTrainingConfig(
        episodes=args.episodes,
        eval_episodes=args.eval_episodes,
        use_boltzmann_eval=args.boltzmann_eval,
        eval_boltzmann_temperature=args.eval_temperature,
    )

    if args.solver == "benchmark":
        results = run_benchmark(
            env_config,
            training_config,
            device=selected_device,
            seed=args.seed,
            num_seeds=args.benchmark_seeds,
            confidence_level=args.confidence_level,
            bootstrap_samples=args.bootstrap_samples,
            include_public_intervention=True,
        )
        print(format_markdown_table(results))
        return

    if args.solver == "twap":
        metrics = evaluate_policy(
            TWAPPolicy(),
            env_config,
            args.eval_episodes,
            seed=args.seed,
            corrupt_public_trace=args.corrupt_public_trace,
        )
        print(format_markdown_table([{"method": "TWAP", **metrics}]))
        return
    if args.solver == "vwap":
        metrics = evaluate_policy(
            VWAPPolicy(),
            env_config,
            args.eval_episodes,
            seed=args.seed,
            corrupt_public_trace=args.corrupt_public_trace,
        )
        print(format_markdown_table([{"method": "VWAP", **metrics}]))
        return
    if args.solver == "pov":
        metrics = evaluate_policy(
            POVPolicy(),
            env_config,
            args.eval_episodes,
            seed=args.seed,
            corrupt_public_trace=args.corrupt_public_trace,
        )
        print(format_markdown_table([{"method": "POV", **metrics}]))
        return
    if args.solver == "almgren":
        metrics = evaluate_policy(
            AlmgrenChrissPolicy(),
            env_config,
            args.eval_episodes,
            seed=args.seed,
            corrupt_public_trace=args.corrupt_public_trace,
        )
        print(format_markdown_table([{"method": "Almgren-Chriss", **metrics}]))
        return

    if args.solver == "ppo":
        solver = IndependentPPOSolver(env_config, training_config, device=selected_device, seed=args.seed)
    else:
        solver = ATDecOptionSolver(env_config, training_config, device=selected_device, seed=args.seed)
    solver.train(args.episodes)
    metrics = solver.evaluate(
        args.eval_episodes,
        corrupt_public_trace=args.corrupt_public_trace,
    )
    print(format_markdown_table([{"method": solver.name, **metrics}]))


if __name__ == "__main__":
    main()
