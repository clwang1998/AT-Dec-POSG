"""AT-Dec-POSG-style multi-option execution benchmark and solver."""

from .benchmark import (
    format_markdown_table,
    resolve_solver_class,
    run_benchmark,
    run_solver_research_eval,
)
from .config import (
    ExecutionTemplate,
    OptionExecutionConfig,
    OptionMarketScenario,
    OptionTrainingConfig,
)
from .env import MultiOptionExecutionEnv
from .hardware import detect_option_hardware, format_hardware_report, resolve_runtime_device
from .solver import ATDecOptionSolver, IndependentPPOSolver

__all__ = [
    "ATDecOptionSolver",
    "ExecutionTemplate",
    "IndependentPPOSolver",
    "MultiOptionExecutionEnv",
    "OptionExecutionConfig",
    "OptionMarketScenario",
    "OptionTrainingConfig",
    "detect_option_hardware",
    "format_markdown_table",
    "format_hardware_report",
    "resolve_solver_class",
    "resolve_runtime_device",
    "run_benchmark",
    "run_solver_research_eval",
]
