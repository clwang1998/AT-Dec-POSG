from __future__ import annotations

import os
from typing import Any, Dict, List

import torch


def resolve_runtime_device(requested_device: str = "auto") -> str:
    normalized = requested_device.strip().lower()
    if normalized == "auto":
        return "cuda:0" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"
    if normalized == "cpu":
        return "cpu"
    if normalized == "cuda":
        normalized = "cuda:0"
    if normalized.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is not available in this environment.")
        try:
            device_index = int(normalized.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError(f"invalid CUDA device specifier '{requested_device}'") from exc
        visible_gpu_count = torch.cuda.device_count()
        if device_index < 0 or device_index >= visible_gpu_count:
            raise ValueError(
                f"requested device '{requested_device}' is out of range for {visible_gpu_count} visible GPU(s)"
            )
        return normalized
    raise ValueError(
        f"unsupported device '{requested_device}'. Expected 'auto', 'cpu', 'cuda', or 'cuda:N'."
    )


def detect_option_hardware(requested_device: str = "auto") -> Dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    visible_gpu_count = int(torch.cuda.device_count()) if cuda_available else 0
    visible_gpu_names: List[str] = []
    if cuda_available:
        visible_gpu_names = [str(torch.cuda.get_device_name(index)) for index in range(visible_gpu_count)]

    selected_device = resolve_runtime_device(requested_device)
    if visible_gpu_count >= 1:
        single_run_recommendation = "single_gpu"
    else:
        single_run_recommendation = "cpu_fallback"

    if visible_gpu_count >= 4:
        throughput_recommendation = "four_gpu"
    elif visible_gpu_count >= 1:
        throughput_recommendation = "single_gpu"
    else:
        throughput_recommendation = "cpu_fallback"

    return {
        "requested_device": requested_device,
        "selected_device": selected_device,
        "cuda_available": cuda_available,
        "visible_gpu_count": visible_gpu_count,
        "visible_gpu_names": visible_gpu_names,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "workload_shape": "single_process_single_device",
        "single_run_recommendation": single_run_recommendation,
        "throughput_recommendation": throughput_recommendation,
        "recommended_parallel_workers": max(1, min(visible_gpu_count, 4)) if visible_gpu_count else 1,
        "analysis": (
            "One option solver run uses exactly one torch device. "
            "Four GPUs do not accelerate a single candidate directly; "
            "they only help by parallelizing seeds or multiple candidates."
        ),
    }


def format_hardware_report(profile: Dict[str, Any]) -> str:
    names = ",".join(profile["visible_gpu_names"]) if profile["visible_gpu_names"] else "-"
    return (
        "hardware: "
        f"selected_device={profile['selected_device']} "
        f"visible_gpus={profile['visible_gpu_count']} "
        f"gpu_names={names} "
        f"single_run={profile['single_run_recommendation']} "
        f"throughput={profile['throughput_recommendation']} "
        f"parallel_workers={profile['recommended_parallel_workers']}"
    )
