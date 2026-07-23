from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import time
from pathlib import Path

import torch

from experiment_common import build_dataset_adapter, resolve_device
from hardware_energy import RunEnergyMonitor
from our_hd import FederatedRunner
from our_hd.federated import ClientState
from our_nn import NNFederatedRunner
from run_seven_dataset_benchmark import (
    HD_METHODS,
    NN_METHODS,
    OPTIONAL_HD_METHODS,
    apply_family_dataset_overrides,
    apply_seed_to_dataset,
    build_config,
    build_hd_method,
    build_nn_method,
    load_dataset_specs,
    maybe_cap_large_dataset_train_data,
    mean_std,
    set_seed,
)


DEFAULT_DATASETS = [
    "uci_har",
    "isolet_raw",
    "femnist",
    "pamap2",
    "wisdm",
    "synthetic",
    "ninapro_db1",
]

INFERENCE_MODE_TO_METRIC: dict[str, str] = {
    "fused": "mean_personalized_accuracy",
    "shared": "mean_shared_branch_accuracy",
    "personal": "mean_personal_branch_accuracy",
    "routed": "mean_routed_accuracy",
}
INFERENCE_MODE_FALLBACKS: list[str] = [
    "mean_personalized_accuracy",
    "mean_local_test_accuracy",
    "mean_accuracy",
]


def _normalize_inference_modes(values: list[str]) -> list[str]:
    values = [str(value).strip().lower() for value in values]
    filtered: list[str] = []
    allowed = {"fused", "shared", "personal", "routed"}
    for value in values:
        if value in allowed and value not in filtered:
            filtered.append(value)
    return filtered


def _summary_key_for(method_name: str, mode: str, *, include_suffix: bool) -> str:
    method_name = str(method_name)
    if not include_suffix or mode == "fused":
        return method_name
    return f"{method_name}::{mode}"


def _row_summary_key(row: dict[str, object], *, include_suffix: bool = True) -> str:
    if include_suffix:
        return str(row.get("summary_key", row["method"]))
    method_name = str(row["method"])
    summary_key = str(row.get("summary_key", method_name))
    return method_name


def _row_method(row: dict[str, object]) -> str:
    return str(row["method"])


def _metric_for_mode(
    metrics: dict[str, object],
    mode: str,
    *,
    fallback_key: str,
) -> tuple[str, float]:
    key = INFERENCE_MODE_TO_METRIC.get(mode, INFERENCE_MODE_TO_METRIC["fused"])
    if key in metrics:
        return key, float(metrics[key])
    if mode == "fused":
        for candidate in INFERENCE_MODE_FALLBACKS:
            if candidate in metrics:
                return candidate, float(metrics[candidate])
        return fallback_key, float(metrics.get(fallback_key, 0.0))
    return fallback_key, float(metrics.get(fallback_key, 0.0))


def _numeric_value(value: object) -> float | None:
    if isinstance(value, torch.Tensor):
        if value.ndim != 0:
            return None
        return float(value.item())
    if isinstance(value, (int, float)):
        return float(value)
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run benchmark comparison with round checkpoints.")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["horu_hd"],
        choices=[*HD_METHODS, *OPTIONAL_HD_METHODS, *NN_METHODS],
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", nargs="+", type=int, default=[13])
    parser.add_argument("--round-checkpoints", nargs="+", type=int, default=[10, 20, 25])
    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--client-participation", type=float, default=1.0)
    parser.add_argument("--deterministic-algorithms", action="store_true")
    parser.add_argument("--torch-num-threads", type=int, default=None)
    parser.add_argument("--hd-dim", type=int, default=2000)
    parser.add_argument("--hd-lr", type=float, default=0.035)
    parser.add_argument("--hd-cosine-random-phase", action="store_true")
    parser.add_argument("--nn-lr", type=float, default=0.001)
    parser.add_argument("--cnn-lr", type=float, default=0.06)
    parser.add_argument("--cnn-optimizer", default="sgd")
    parser.add_argument("--cnn-momentum", type=float, default=0.0)
    parser.add_argument("--cnn-weight-decay", type=float, default=0.0)
    parser.add_argument("--dfl-align-weight", type=float, default=1.0)
    parser.add_argument("--dfl-disentangle-weight", type=float, default=0.1)
    parser.add_argument("--fedprox-mu", type=float, default=0.01)
    parser.add_argument("--collapse-factor", type=float, default=1.5)
    parser.add_argument("--subspace-shared-rank", type=int, default=32)
    parser.add_argument("--subspace-personal-rank", type=int, default=64)
    parser.add_argument("--subspace-val-fraction", type=float, default=0.0)
    parser.add_argument("--subspace-fusion-alpha", type=float, default=None)
    parser.add_argument("--subspace-rowgate-alpha", type=float, default=1.0)
    parser.add_argument("--subspace-rowgate-min", type=float, default=0.1)
    parser.add_argument("--subspace-rowgate-max", type=float, default=0.9)
    parser.add_argument("--subspace-explore-rounds", type=int, default=3)
    parser.add_argument("--subspace-refresh-interval", type=int, default=0)
    parser.add_argument("--subspace-intersection-rank", type=int, default=24)
    parser.add_argument("--subspace-intersection-ratio", type=float, default=None)
    parser.add_argument("--subspace-initial-intersection-ratio", type=float, default=0.25)
    parser.add_argument("--subspace-basis-demotion-ratio", type=float, default=0.5)
    parser.add_argument("--enable-wasserstein-sync", action="store_true")
    parser.add_argument("--wasserstein-atoms", type=int, default=3)
    parser.add_argument("--wasserstein-beta", type=float, default=0.0)
    parser.add_argument("--wasserstein-max-iters", type=int, default=20)
    parser.add_argument("--wasserstein-interval", type=int, default=1)
    parser.add_argument("--large-dataset-train-threshold", type=int, default=100000)
    parser.add_argument("--large-dataset-train-cap", type=int, default=50000)
    parser.add_argument("--inference-only", action="store_true")
    parser.add_argument(
        "--inference-modes",
        nargs="+",
        default=["fused", "shared", "personal", "routed"],
        choices=["fused", "shared", "personal", "routed"],
    )
    parser.add_argument(
        "--checkpoint-state-dir",
        default="results/hd_checkpoint_states",
    )
    parser.add_argument("--save-round-states", action="store_true")
    parser.add_argument(
        "--json-out",
        default="results/hd_checkpoint_comparison_dim2000_e3_r10_20_25.json",
    )
    parser.add_argument(
        "--md-out",
        default="results/hd_checkpoint_comparison_dim2000_e3_r10_20_25.md",
    )
    parser.add_argument("--resume-from", nargs="*", default=[])
    parser.add_argument("--measure-energy", action="store_true")
    parser.add_argument("--power-sample-interval-ms", type=float, default=200.0)
    return parser.parse_args()


def render_markdown(report: dict) -> str:
    lines = ["# Checkpoint Comparison", ""]
    lines.append(f"- device: `{report['device']}`")
    lines.append(f"- seeds: `{report['seeds']}`")
    lines.append(f"- datasets: `{report['run_config']['datasets']}`")
    lines.append(f"- methods: `{report['run_config']['methods']}`")
    lines.append(f"- inference only: `{report['run_config'].get('inference_only', False)}`")
    lines.append(f"- inference modes: `{report['run_config'].get('inference_modes', ['fused'])}`")
    lines.append(f"- checkpoint states: `{report['run_config'].get('checkpoint_state_dir', 'not configured')}`")
    lines.append(f"- round checkpoints: `{report['round_checkpoints']}`")
    lines.append(f"- local epochs: `{report['run_config']['local_epochs']}`")
    lines.append(f"- batch size: `{report['run_config']['batch_size']}`")
    lines.append(f"- client participation: `{report['run_config']['client_participation']}`")
    lines.append(f"- hd dim: `{report['run_config']['hd_dim']}`")
    lines.append(f"- hd lr: `{report['run_config']['hd_lr']}`")
    lines.append(f"- hd cosine random phase: `{report['run_config'].get('hd_cosine_random_phase', False)}`")
    lines.append(f"- nn lr: `{report['run_config']['nn_lr']}`")
    lines.append(f"- measure energy: `{report['run_config'].get('measure_energy', False)}`")
    if report["run_config"]["large_dataset_train_threshold"] is not None:
        lines.append(
            f"- large-dataset train cap: threshold `{report['run_config']['large_dataset_train_threshold']}`, "
            f"cap `{report['run_config']['large_dataset_train_cap']}`"
        )
    lines.append("")

    for dataset_name, dataset_report in report["datasets"].items():
        lines.append(f"## {dataset_name}")
        lines.append("")
        lines.append(
            f"- classes: `{dataset_report['num_classes']}`, chance accuracy: "
            f"`{dataset_report['chance_accuracy']:.4f}`"
        )
        sampling_records = dataset_report.get("sampling_records", [])
        sampling_record = list(sampling_records or [{}])[0]
        if sampling_record.get("applied"):
            lines.append(
                f"- train sampling: `{sampling_record['total_train_samples_before']} -> "
                f"{sampling_record['total_train_samples_after']}` across "
                f"`{sampling_record['clients_modified']}` clients"
            )
        lines.append("")
        if report["run_config"].get("measure_energy", False):
            lines.append(
                "| method | primary metric | "
                + " | ".join(f"R{round_id}" for round_id in report["round_checkpoints"])
                + " | runtime mean (s) | gpu energy mean (J) | gpu avg power mean (W) |"
            )
            lines.append(
                "| --- | --- | "
                + " | ".join("---:" for _ in report["round_checkpoints"])
                + " | ---: | ---: | ---: |"
            )
        else:
            lines.append(
                "| method | primary metric | "
                + " | ".join(f"R{round_id}" for round_id in report["round_checkpoints"])
                + " | runtime mean (s) |"
            )
            lines.append(
                "| --- | --- | "
                + " | ".join("---:" for _ in report["round_checkpoints"])
                + " | ---: |"
            )
        for method_name in report["run_config"]["summary_methods"]:
            summary = dataset_report["summary"].get(method_name)
            if summary is None:
                continue
            round_strings = []
            for round_id in report["round_checkpoints"]:
                stats = summary["per_round"][str(round_id)]
                round_strings.append(f"{stats['mean']:.4f}")
            if report["run_config"].get("measure_energy", False):
                gpu_energy = "" if summary.get("gpu_energy_j_mean") is None else f"{summary['gpu_energy_j_mean']:.2f}"
                gpu_power = "" if summary.get("gpu_avg_power_w_mean") is None else f"{summary['gpu_avg_power_w_mean']:.2f}"
                lines.append(
                    f"| {method_name} | {summary['metric_key']} | "
                    + " | ".join(round_strings)
                    + f" | {summary['runtime_seconds_mean']:.2f} | {gpu_energy} | {gpu_power} |"
                )
            else:
                lines.append(
                    f"| {method_name} | {summary['metric_key']} | "
                    + " | ".join(round_strings)
                    + f" | {summary['runtime_seconds_mean']:.2f} |"
                )
        lines.append("")

    return "\n".join(lines) + "\n"


def build_run_config(args: argparse.Namespace) -> dict:
    inference_modes = _normalize_inference_modes(list(args.inference_modes))
    if not inference_modes:
        inference_modes = ["fused"]
        args.inference_modes = inference_modes
    inference_only = bool(getattr(args, "inference_only", False))
    summary_methods: list[str] = []
    if inference_only:
        for method_name in args.methods:
            for mode in inference_modes:
                summary_methods.append(_summary_key_for(method_name, mode, include_suffix=True))
    else:
        summary_methods = [str(method_name) for method_name in args.methods]

    return {
        "datasets": list(args.datasets),
        "methods": list(args.methods),
        "summary_methods": summary_methods,
        "local_epochs": int(args.local_epochs),
        "batch_size": int(args.batch_size),
        "client_participation": float(args.client_participation),
        "deterministic_algorithms": bool(args.deterministic_algorithms),
        "torch_num_threads": None if args.torch_num_threads is None else int(args.torch_num_threads),
        "hd_dim": int(args.hd_dim),
        "hd_lr": float(args.hd_lr),
        "hd_cosine_random_phase": bool(args.hd_cosine_random_phase),
        "nn_lr": float(args.nn_lr),
        "cnn_lr": float(args.cnn_lr),
        "cnn_optimizer": str(args.cnn_optimizer),
        "cnn_momentum": float(args.cnn_momentum),
        "cnn_weight_decay": float(args.cnn_weight_decay),
        "dfl_align_weight": float(args.dfl_align_weight),
        "dfl_disentangle_weight": float(args.dfl_disentangle_weight),
        "fedprox_mu": float(args.fedprox_mu),
        "subspace_shared_rank": int(args.subspace_shared_rank),
        "subspace_personal_rank": int(args.subspace_personal_rank),
        "subspace_val_fraction": float(args.subspace_val_fraction),
        "subspace_fusion_alpha": (
            None if args.subspace_fusion_alpha is None else float(args.subspace_fusion_alpha)
        ),
        "subspace_rowgate_alpha": float(args.subspace_rowgate_alpha),
        "subspace_rowgate_min": float(args.subspace_rowgate_min),
        "subspace_rowgate_max": float(args.subspace_rowgate_max),
        "subspace_explore_rounds": int(args.subspace_explore_rounds),
        "subspace_refresh_interval": int(args.subspace_refresh_interval),
        "subspace_intersection_rank": int(args.subspace_intersection_rank),
        "subspace_intersection_ratio": (
            None if args.subspace_intersection_ratio is None else float(args.subspace_intersection_ratio)
        ),
        "subspace_initial_intersection_ratio": float(args.subspace_initial_intersection_ratio),
        "subspace_basis_demotion_ratio": float(args.subspace_basis_demotion_ratio),
        "enable_wasserstein_sync": bool(args.enable_wasserstein_sync),
        "wasserstein_atoms": int(args.wasserstein_atoms),
        "wasserstein_beta": float(args.wasserstein_beta),
        "wasserstein_max_iters": int(args.wasserstein_max_iters),
        "wasserstein_interval": int(args.wasserstein_interval),
        "measure_energy": bool(args.measure_energy),
        "power_sample_interval_ms": float(args.power_sample_interval_ms),
        "inference_only": inference_only,
        "inference_modes": inference_modes,
        "save_round_states": bool(getattr(args, "save_round_states", False)),
        "checkpoint_state_dir": str(getattr(args, "checkpoint_state_dir", "results/hd_checkpoint_states")),
        "large_dataset_train_threshold": (
            None if args.large_dataset_train_threshold is None else int(args.large_dataset_train_threshold)
        ),
        "large_dataset_train_cap": (
            None if args.large_dataset_train_cap is None else int(args.large_dataset_train_cap)
        ),
    }


def build_report(args: argparse.Namespace, device: torch.device, round_checkpoints: list[int]) -> dict:
    return {
        "analysis": "checkpoint_comparison",
        "device": str(device),
        "seeds": list(args.seeds),
        "round_checkpoints": round_checkpoints,
        "run_config": build_run_config(args),
        "datasets": {},
    }


def save_report(report: dict, args: argparse.Namespace) -> None:
    json_out = Path(args.json_out)
    md_out = Path(args.md_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_out.write_text(render_markdown(report), encoding="utf-8")


def dataset_summary_from_rows(
    *,
    rows: list[dict],
    method_order: list[str],
    round_checkpoints: list[int],
    collapse_factor: float,
    num_classes: int,
    sampling_records: list[dict],
    dataset_spec: dict,
) -> dict:
    effective_dataset_specs_by_seed: dict[str, dict] = {}
    for row in rows:
        config = row.get("config")
        if not isinstance(config, dict) or not isinstance(config.get("dataset"), dict):
            continue
        seed_key = str(int(row["seed"]))
        effective_dataset_specs_by_seed.setdefault(seed_key, copy.deepcopy(config["dataset"]))
    summary: dict[str, dict] = {}
    for method_name in method_order:
        method_rows = [row for row in rows if _row_summary_key(row) == str(method_name)]
        if not method_rows:
            continue
        metric_key = str(method_rows[0]["metric_key"])
        per_round: dict[str, dict[str, float | bool]] = {}
        for round_id in round_checkpoints:
            values = [
                _numeric_value(row["checkpoint_metrics"][str(round_id)].get(metric_key, 0.0)) or 0.0
                for row in method_rows
            ]
            chance_accuracy = float(method_rows[0]["chance_accuracy"])
            stats = mean_std(values)
            per_round[str(round_id)] = {
                "mean": stats["mean"],
                "std": stats["std"],
                "collapsed": stats["mean"] <= float(collapse_factor) * chance_accuracy,
            }
        runtime_stats = mean_std([float(row["runtime_seconds"]) for row in method_rows])
        gpu_energy_values = [
            float(row["energy_metrics"].get("gpu_energy_j", 0.0))
            for row in method_rows
            if isinstance(row.get("energy_metrics", {}).get("gpu_energy_j"), (int, float))
        ]
        gpu_avg_power_values = [
            float(row["energy_metrics"].get("gpu_avg_power_w", 0.0))
            for row in method_rows
            if isinstance(row.get("energy_metrics", {}).get("gpu_avg_power_w"), (int, float))
        ]
        cpu_energy_values = [
            float(row["energy_metrics"].get("cpu_energy_j", 0.0))
            for row in method_rows
            if isinstance(row.get("energy_metrics", {}).get("cpu_energy_j"), (int, float))
        ]
        summary[method_name] = {
            "family": str(method_rows[0]["family"]),
            "metric_key": metric_key,
            "chance_accuracy": float(method_rows[0]["chance_accuracy"]),
            "runtime_seconds_mean": runtime_stats["mean"],
            "runtime_seconds_std": runtime_stats["std"],
            "gpu_energy_j_mean": mean_std(gpu_energy_values)["mean"] if gpu_energy_values else None,
            "gpu_energy_j_std": mean_std(gpu_energy_values)["std"] if gpu_energy_values else None,
            "gpu_avg_power_w_mean": mean_std(gpu_avg_power_values)["mean"] if gpu_avg_power_values else None,
            "gpu_avg_power_w_std": mean_std(gpu_avg_power_values)["std"] if gpu_avg_power_values else None,
            "cpu_energy_j_mean": mean_std(cpu_energy_values)["mean"] if cpu_energy_values else None,
            "cpu_energy_j_std": mean_std(cpu_energy_values)["std"] if cpu_energy_values else None,
            "per_round": per_round,
        }

    return {
        "num_classes": int(num_classes),
        "chance_accuracy": 1.0 / float(num_classes),
        "rows": rows,
        "sampling_records": sampling_records,
        "summary": summary,
        "dataset_spec": dataset_spec,
        "effective_dataset_specs_by_seed": effective_dataset_specs_by_seed,
    }


def _normalize_device_name(value: object) -> str:
    text = str(value)
    if text.startswith("cuda"):
        return "cuda"
    return text


def _compatible_resume_report(
    source: dict,
    *,
    device: torch.device,
    seeds: list[int],
    round_checkpoints: list[int],
    run_config: dict,
) -> bool:
    if source.get("analysis") not in {"hd_checkpoint_comparison", "checkpoint_comparison"}:
        return False
    if list(source.get("seeds", [])) != list(seeds):
        return False
    if list(source.get("round_checkpoints", [])) != list(round_checkpoints):
        return False
    if _normalize_device_name(source.get("device")) != _normalize_device_name(device):
        return False

    source_cfg = source.get("run_config", {})
    defaults = {
        "nn_lr": 0.001,
        "fedprox_mu": 0.01,
        "client_participation": 1.0,
        "subspace_fusion_alpha": None,
        "subspace_rowgate_alpha": 1.0,
        "subspace_rowgate_min": 0.1,
        "subspace_rowgate_max": 0.9,
        "enable_wasserstein_sync": False,
        "wasserstein_atoms": 3,
        "wasserstein_beta": 0.0,
        "wasserstein_max_iters": 20,
        "wasserstein_interval": 1,
        "measure_energy": False,
        "power_sample_interval_ms": 200.0,
        "large_dataset_train_threshold": 100000,
        "large_dataset_train_cap": 10000,
    }
    comparable_keys = [
        "local_epochs",
        "batch_size",
        "client_participation",
        "hd_dim",
        "hd_lr",
        "nn_lr",
        "fedprox_mu",
        "subspace_shared_rank",
        "subspace_personal_rank",
        "subspace_val_fraction",
        "subspace_fusion_alpha",
        "subspace_rowgate_alpha",
        "subspace_rowgate_min",
        "subspace_rowgate_max",
        "enable_wasserstein_sync",
        "wasserstein_atoms",
        "wasserstein_beta",
        "wasserstein_max_iters",
        "wasserstein_interval",
        "subspace_explore_rounds",
        "subspace_refresh_interval",
        "subspace_intersection_rank",
        "subspace_intersection_ratio",
        "subspace_initial_intersection_ratio",
        "subspace_basis_demotion_ratio",
        "measure_energy",
        "power_sample_interval_ms",
        "large_dataset_train_threshold",
        "large_dataset_train_cap",
    ]
    for key in comparable_keys:
        expected = run_config[key]
        observed = source_cfg.get(key, defaults.get(key))
        if observed != expected:
            return False
    return True


def load_resume_state(
    *,
    args: argparse.Namespace,
    device: torch.device,
    round_checkpoints: list[int],
) -> tuple[dict[tuple[str, int, str], dict], dict[str, dict]]:
    row_cache: dict[tuple[str, int, str], dict] = {}
    dataset_cache: dict[str, dict] = {}
    candidate_paths = [Path(args.json_out), *(Path(path) for path in args.resume_from)]
    seen: set[Path] = set()
    run_config = build_run_config(args)
    requested_methods = {str(method) for method in args.methods}

    for path in candidate_paths:
        resolved = path.resolve()
        if resolved in seen or not path.exists():
            continue
        seen.add(resolved)
        source = json.loads(path.read_text(encoding="utf-8"))
        if not _compatible_resume_report(
            source,
            device=device,
            seeds=list(args.seeds),
            round_checkpoints=round_checkpoints,
            run_config=run_config,
        ):
            continue
        for dataset_name, dataset_report in source.get("datasets", {}).items():
            if dataset_name not in args.datasets:
                continue
            dataset_cache[dataset_name] = copy.deepcopy(dataset_report)
            for row in dataset_report.get("rows", []):
                row_method = str(row.get("method"))
                if row_method not in requested_methods:
                    continue
                row_seed = int(row.get("seed", -1))
                if row_seed not in args.seeds:
                    continue
                row_key = _row_summary_key(row)
                row_cache[(str(dataset_name), row_seed, row_key)] = copy.deepcopy(row)

    return row_cache, dataset_cache


def upsert_sampling_record(records: list[dict], record: dict) -> list[dict]:
    updated = [entry for entry in records if int(entry.get("seed", -1)) != int(record.get("seed", -1))]
    updated.append(copy.deepcopy(record))
    updated.sort(key=lambda entry: int(entry.get("seed", -1)))
    return updated


def _cfg_signature(cfg: dict) -> str:
    payload = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


def _serialize_state_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to(torch.device("cpu")).clone()
    if isinstance(value, dict):
        return {str(k): _serialize_state_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_serialize_state_value(v) for v in value]
    if isinstance(value, tuple):
        return [_serialize_state_value(v) for v in value]
    return copy.deepcopy(value)


def _deserialize_state_value(value):
    if isinstance(value, dict):
        return {k: _deserialize_state_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_deserialize_state_value(v) for v in value]
    return value


def _to_device_state_value(value, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, dict):
        return {k: _to_device_state_value(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_device_state_value(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(_to_device_state_value(v, device) for v in value)
    return value


def _move_object_tensors_to_device(value: object, device: torch.device) -> object:
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, dict):
        for key, item in value.items():
            value[key] = _move_object_tensors_to_device(item, device)
        return value
    if isinstance(value, list):
        return [_move_object_tensors_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_object_tensors_to_device(item, device) for item in value)
    if isinstance(value, set):
        return {_move_object_tensors_to_device(item, device) for item in value}
    if hasattr(value, "__dict__"):
        for key, item in value.__dict__.items():
            setattr(value, key, _move_object_tensors_to_device(item, device))
        return value
    return value


def _serialize_client_states(round_states: dict[str, list[ClientState]]) -> dict[str, list[dict[str, object]]]:
    serialized: dict[str, list[dict[str, object]]] = {}
    for round_id, states in round_states.items():
        serialized[str(round_id)] = [
            {
                "memory": _serialize_state_value(state.memory),
                "extras": _serialize_state_value(state.extras),
            }
            for state in states
        ]
    return serialized


def _deserialize_client_states(payload: dict[str, list[dict[str, object]]]) -> dict[str, list[ClientState]]:
    rebuilt: dict[str, list[ClientState]] = {}
    for round_id, serialized_states in payload.items():
        rebuilt_states: list[ClientState] = []
        for serialized_state in serialized_states:
            rebuilt_states.append(
                ClientState(
                    memory=_deserialize_state_value(serialized_state.get("memory")),
                    extras=_deserialize_state_value(serialized_state.get("extras")),
                )
            )
        rebuilt[str(round_id)] = rebuilt_states
    return rebuilt


def _move_client_states_to_device(
    round_states: dict[str, list[ClientState]],
    device: torch.device,
) -> dict[str, list[ClientState]]:
    moved: dict[str, list[ClientState]] = {}
    for round_id, states in round_states.items():
        moved_states: list[ClientState] = []
        for state in states:
            moved_states.append(
                ClientState(
                    memory=None if state.memory is None else _to_device_state_value(state.memory, device),
                    extras=None if state.extras is None else _to_device_state_value(state.extras, device),
                )
            )
        moved[round_id] = moved_states
    return moved


def _checkpoint_state_path(
    *,
    base_dir: Path,
    dataset_name: str,
    method_name: str,
    seed: int,
    signature: str,
) -> Path:
    safe_dataset = re.sub(r"[^A-Za-z0-9._-]+", "_", str(dataset_name)).strip("_") or "dataset"
    safe_method = re.sub(r"[^A-Za-z0-9._-]+", "_", str(method_name)).strip("_") or "method"
    return base_dir / safe_dataset / safe_method / f"seed_{int(seed)}" / f"{signature}.pt"


def _save_round_state_archive(path: Path, archive: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(archive, path)


def _load_round_state_archive(path: Path) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid checkpoint state archive: {path}")
    serialized_round_states = (
        payload.get("round_states")
        if isinstance(payload.get("round_states"), dict)
        else payload
        if all(isinstance(value, list) for value in payload.values())
        else None
    )
    if serialized_round_states is None:
        raise ValueError(f"Invalid checkpoint state archive: missing round_states in {path}")
    method_states = payload.get("method_states", {})
    if not isinstance(method_states, dict):
        method_states = {}
    return {
        "round_states": _deserialize_client_states(serialized_round_states),
        "method_states": method_states,
    }


def _build_checkpoint_archive(
    *,
    dataset_name: str,
    method: str,
    seed: int,
    cfg: dict,
    round_checkpoints: list[int],
    round_states: dict[str, list[ClientState]],
    method_states: dict[str, object] | None = None,
) -> dict:
    return {
        "dataset": str(dataset_name),
        "method": str(method),
        "seed": int(seed),
        "round_checkpoints": [int(round_id) for round_id in round_checkpoints],
        "cfg_signature": _cfg_signature(cfg),
        "cfg": cfg,
        "round_states": _serialize_client_states(round_states),
        **({"method_states": method_states} if method_states is not None else {}),
    }


def _resolve_state_path(
    *,
    row: dict,
    args: argparse.Namespace,
    cfg: dict,
) -> Path | None:
    explicit = row.get("round_state_archive")
    if isinstance(explicit, str):
        explicit_path = Path(explicit)
        if explicit_path.exists():
            return explicit_path
    signature = _cfg_signature(cfg)
    if not signature:
        return None
    dataset_name = str(row.get("dataset", ""))
    method_name = str(row.get("method", ""))
    seed = int(row.get("seed", 0))
    candidate = _checkpoint_state_path(
        base_dir=Path(args.checkpoint_state_dir),
        dataset_name=dataset_name,
        method_name=method_name,
        seed=seed,
        signature=signature,
    )
    if candidate.exists():
        return candidate
    return None


def _empty_energy_metrics() -> dict[str, float | None]:
    return {
        "gpu_energy_j": None,
        "gpu_avg_power_w": None,
        "cpu_energy_j": None,
    }


def _normalize_energy_metrics(metrics: dict | None) -> dict[str, float | None]:
    if not isinstance(metrics, dict):
        return _empty_energy_metrics()
    return {
        "gpu_energy_j": None if metrics.get("gpu_energy_j") is None else float(metrics["gpu_energy_j"]),
        "gpu_avg_power_w": None if metrics.get("gpu_avg_power_w") is None else float(metrics["gpu_avg_power_w"]),
        "cpu_energy_j": None if metrics.get("cpu_energy_j") is None else float(metrics["cpu_energy_j"]),
    }


def _build_training_row(
    *,
    dataset_name: str,
    seed: int,
    method_name: str,
    cfg_family: str,
    cfg: dict,
    metric_key: str,
    round_checkpoints: list[int],
    max_round: int,
    result: dict,
    runtime_seconds: float,
    chance_accuracy: float,
    energy_metrics: dict[str, object],
    round_state_archive: str | None = None,
) -> dict:
    checkpoint_metrics: dict[str, dict[str, float]] = {}
    history: list[dict[str, object]] = result["history"]
    for round_id in round_checkpoints:
        index = int(round_id) - 1
        if index < 0 or index >= len(history):
            raise ValueError(f"History too short for round {round_id} in {dataset_name}/{method_name}")
        history_item = history[index]
        checkpoint_metrics[str(round_id)] = {
            str(key): float(value)
            for key, value in history_item.items()
            if isinstance(value, (int, float, torch.Tensor))
        }
    primary_value = float(checkpoint_metrics[str(max_round)].get(metric_key, 0.0))
    return {
        "dataset": str(dataset_name),
        "seed": int(seed),
        "method": str(method_name),
        "family": str(cfg_family),
        "metric_key": str(metric_key),
        "chance_accuracy": float(chance_accuracy),
        "config": cfg,
        "checkpoint_metrics": checkpoint_metrics,
        "primary_value": primary_value,
        "runtime_seconds": float(runtime_seconds),
        "energy_metrics": _normalize_energy_metrics(energy_metrics),
        **({} if round_state_archive is None else {"round_state_archive": str(round_state_archive)}),
    }


def _build_inference_row(
    *,
    dataset_name: str,
    seed: int,
    method_name: str,
    cfg_family: str,
    cfg: dict,
    mode: str,
    summary_key: str,
    checkpoint_metrics: dict[str, dict[str, float]],
    primary_value: float,
    metric_key: str,
    runtime_seconds: float,
    chance_accuracy: float,
) -> dict:
    return {
        "dataset": str(dataset_name),
        "seed": int(seed),
        "method": str(method_name),
        "family": str(cfg_family),
        "summary_key": str(summary_key),
        "inference_mode": str(mode),
        "metric_key": str(metric_key),
        "chance_accuracy": float(chance_accuracy),
        "config": cfg,
        "checkpoint_metrics": checkpoint_metrics,
        "primary_value": float(primary_value),
        "runtime_seconds": float(runtime_seconds),
        "energy_metrics": _empty_energy_metrics(),
    }


def main() -> None:
    args = parse_args()
    if args.torch_num_threads is not None:
        if int(args.torch_num_threads) < 1:
            raise ValueError("--torch-num-threads must be at least 1")
        torch.set_num_threads(int(args.torch_num_threads))
    if args.deterministic_algorithms:
        torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    args.inference_modes = _normalize_inference_modes(list(args.inference_modes))
    if args.inference_only and not args.inference_modes:
        args.inference_modes = ["fused"]

    inference_only = bool(args.inference_only)
    required_modes = ["fused"] if not inference_only else list(args.inference_modes)
    if inference_only:
        unsupported = set(args.methods) - set(HD_METHODS) - set(OPTIONAL_HD_METHODS)
        if unsupported:
            raise ValueError(
                f"inference-only supports only HD methods, got unsupported methods: {sorted(unsupported)}"
            )

    device = resolve_device(args.device)
    dataset_specs = load_dataset_specs()
    round_checkpoints = sorted({int(value) for value in args.round_checkpoints})
    max_round = max(round_checkpoints)
    if max_round <= 0:
        raise ValueError(f"Invalid max round: {max_round}")
    args.pilot = False
    args.rounds = max_round
    args.pilot_rounds = 5
    args.pilot_local_epochs = 1
    args.pilot_max_clients = 12
    save_round_states = bool(args.save_round_states) and not inference_only

    report = build_report(args, device, round_checkpoints)
    run_config = report["run_config"]
    row_cache, dataset_cache = load_resume_state(args=args, device=device, round_checkpoints=round_checkpoints)

    for dataset_name in args.datasets:
        if dataset_name not in dataset_specs:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        dataset_spec = copy.deepcopy(dataset_specs[dataset_name])
        cached_dataset = dataset_cache.get(dataset_name, {})
        rows = [
            copy.deepcopy(row)
            for (row_dataset, row_seed, _row_key), row in row_cache.items()
            if row_dataset == dataset_name and int(row_seed) in args.seeds and str(row.get("method")) in run_config["methods"]
        ]
        sampling_records = copy.deepcopy(cached_dataset.get("sampling_records", []))
        dataset_num_classes = cached_dataset.get("num_classes")
        if dataset_num_classes is not None:
            dataset_num_classes = int(dataset_num_classes)

        expected_pairs: set[tuple[int, str]] = set()
        for seed in args.seeds:
            if inference_only:
                for method_name in run_config["methods"]:
                    for mode in required_modes:
                        expected_pairs.add((int(seed), _summary_key_for(method_name, mode, include_suffix=True)))
            else:
                for method_name in run_config["methods"]:
                    expected_pairs.add((int(seed), str(method_name)))

        completed_pairs = {
            (int(row["seed"]), _row_summary_key(row))
            for row in rows
            if int(row["seed"]) in args.seeds and str(row.get("method")) in run_config["methods"]
        }
        if completed_pairs.issuperset(expected_pairs) and dataset_num_classes is not None:
            print(f"[hd-checkpoints] dataset={dataset_name} phase=skip_all source=resume")
            report["datasets"][dataset_name] = dataset_summary_from_rows(
                rows=rows,
                method_order=run_config["summary_methods"],
                round_checkpoints=round_checkpoints,
                collapse_factor=float(args.collapse_factor),
                num_classes=int(dataset_num_classes),
                sampling_records=sampling_records,
                dataset_spec=dataset_spec,
            )
            save_report(report, args)
            continue

        for seed in args.seeds:
            if inference_only:
                seed_pending = [
                    (str(method_name), mode)
                    for method_name in run_config["methods"]
                    for mode in required_modes
                    if (int(seed), _summary_key_for(method_name, mode, include_suffix=True)) not in completed_pairs
                ]
            else:
                seed_pending = [
                    str(method_name)
                    for method_name in run_config["methods"]
                    if (int(seed), str(method_name)) not in completed_pairs
                ]
            if not seed_pending:
                print(f"[hd-checkpoints] dataset={dataset_name} seed={seed} phase=skip_all_methods source=resume")
                continue

            set_seed(seed)
            seeded_dataset = apply_seed_to_dataset(dataset_spec, seed)
            family_cache: dict[str, dict[str, object]] = {}
            family_cache_records: dict[str, dict[str, int] | None] = {}

            for method_name in run_config["methods"]:
                if not inference_only and (int(seed), str(method_name)) in completed_pairs:
                    print(
                        f"[hd-checkpoints] dataset={dataset_name} seed={seed} method={method_name} "
                        f"phase=skip source=resume"
                    )
                    continue
                if inference_only and method_name in NN_METHODS:
                    continue

                cfg_family = "nn" if method_name in NN_METHODS else "hd"
                if cfg_family not in family_cache:
                    effective_dataset = apply_family_dataset_overrides(seeded_dataset, family=cfg_family)
                    adapter = build_dataset_adapter({"dataset": effective_dataset}, torch.device("cpu"))
                    clients = adapter.load_clients()
                    clients, sampling_info = maybe_cap_large_dataset_train_data(
                        clients,
                        threshold=args.large_dataset_train_threshold,
                        total_cap=args.large_dataset_train_cap,
                        seed=(int(seed) * 7919) + 29,
                    )
                    family_cache[cfg_family] = {
                        "clients": clients,
                        "input_dim": int(clients[0].x_train.shape[1]),
                        "num_classes": int(adapter.num_classes()),
                        "chance_accuracy": float(1.0 / float(adapter.num_classes())),
                    }
                    family_cache_records[cfg_family] = sampling_info
                    if dataset_num_classes is None:
                        dataset_num_classes = int(adapter.num_classes())
                    if sampling_info.get("applied"):
                        sampling_records = upsert_sampling_record(
                            sampling_records,
                            {"seed": int(seed), **sampling_info},
                        )

                cached = family_cache[cfg_family]
                clients = cached["clients"]
                input_dim = int(cached["input_dim"])
                num_classes = int(cached["num_classes"])
                chance_accuracy = float(cached["chance_accuracy"])

                if not inference_only:
                    effective_dataset = apply_family_dataset_overrides(seeded_dataset, family=cfg_family)
                    cfg_family, cfg = build_config(effective_dataset, method_name, args)
                    set_seed(seed)
                    monitor = RunEnergyMonitor(
                        device=device,
                        enabled=bool(args.measure_energy),
                        sample_interval_sec=float(args.power_sample_interval_ms) / 1000.0,
                    )
                    start = time.perf_counter()
                    with monitor:
                        if cfg_family == "hd":
                            method, metric_key = build_hd_method(cfg, input_dim, num_classes, device)
                            result = FederatedRunner(
                                method=method,
                                rounds=max_round,
                                client_participation=float(cfg["train"].get("client_participation", 1.0)),
                                seed=seed,
                            ).run(clients, snapshot_rounds=(set(round_checkpoints) if save_round_states else None))
                        else:
                            method, metric_key = build_nn_method(cfg, input_dim, num_classes, device)
                            result = NNFederatedRunner(
                                method=method,
                                rounds=max_round,
                                client_participation=float(cfg["train"].get("client_participation", 1.0)),
                                seed=seed,
                            ).run(clients)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    runtime_seconds = time.perf_counter() - start
                    energy_metrics = monitor.stop(elapsed_seconds=runtime_seconds)

                    state_path: str | None = None
                    if save_round_states and cfg_family == "hd":
                        state_path_obj = _checkpoint_state_path(
                            base_dir=Path(args.checkpoint_state_dir),
                            dataset_name=dataset_name,
                            method_name=method_name,
                            seed=seed,
                            signature=_cfg_signature(cfg),
                        )
                        _save_round_state_archive(
                            path=state_path_obj,
                            archive=_build_checkpoint_archive(
                                dataset_name=dataset_name,
                                method=method_name,
                                seed=seed,
                                cfg=cfg,
                                round_checkpoints=round_checkpoints,
                                round_states=result["round_states"],
                                method_states=result.get("method_states"),
                            ),
                        )
                        state_path = str(state_path_obj)
                    row = _build_training_row(
                        dataset_name=dataset_name,
                        seed=seed,
                        method_name=method_name,
                        cfg_family=cfg_family,
                        cfg=cfg,
                        metric_key=metric_key,
                        round_checkpoints=round_checkpoints,
                        max_round=max_round,
                        result=result,
                        runtime_seconds=runtime_seconds,
                        chance_accuracy=chance_accuracy,
                        energy_metrics=energy_metrics,
                        round_state_archive=state_path,
                    )
                    rows.append(row)
                    row_cache[(dataset_name, int(seed), _row_summary_key(row))] = row
                    completed_pairs.add((int(seed), _row_summary_key(row)))
                    print(
                        f"[hd-checkpoints] dataset={dataset_name} seed={seed} method={method_name} "
                        f"R{max_round}_{metric_key}={row['primary_value']:.4f} runtime_seconds={runtime_seconds:.2f}"
                    )
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

                else:
                    # Inference-only mode: reuse existing training checkpoints and evaluate at each checkpoint.
                    base_key = _summary_key_for(method_name, "fused", include_suffix=False)
                    base_row = row_cache.get((dataset_name, int(seed), base_key))
                    if base_row is None:
                        raise RuntimeError(
                            f"Missing base checkpoint row for method={method_name}, dataset={dataset_name}, seed={seed}. "
                            f"Run baseline training with --save-round-states first."
                        )
                    if str(base_row.get("family", "hd")) != "hd":
                        raise RuntimeError(
                            f"Inference-only currently supports HD checkpoints only; got family="
                            f"{base_row.get('family')} for method={method_name}, dataset={dataset_name}, seed={seed}"
                        )
                    base_cfg = base_row.get("config")
                    if not isinstance(base_cfg, dict):
                        raise RuntimeError(
                            f"Missing config in cached base row for method={method_name}, dataset={dataset_name}, seed={seed}"
                        )
                    cfg = copy.deepcopy(base_cfg)
                    state_path = _resolve_state_path(row=base_row, args=args, cfg=cfg)
                    if state_path is None:
                        raise RuntimeError(
                            f"No round checkpoint archive found for method={method_name}, dataset={dataset_name}, seed={seed}"
                        )
                    checkpoint_archive = _load_round_state_archive(state_path)
                    method_snapshots = checkpoint_archive.get("method_states", {})
                    round_states = _move_client_states_to_device(
                        checkpoint_archive["round_states"],
                        device=torch.device("cpu"),
                    )
                    missing_rounds = [
                        str(round_id) for round_id in round_checkpoints if str(round_id) not in round_states
                    ]
                    if missing_rounds:
                        raise RuntimeError(
                            f"Missing rounds {missing_rounds} in archive for method={method_name}, "
                            f"dataset={dataset_name}, seed={seed}, archive={state_path}"
                        )

                    mode_histories: dict[str, dict[str, float]] = {}
                    start = time.perf_counter()
                    for round_id in round_checkpoints:
                        round_key = str(round_id)
                        method_snapshot = method_snapshots.get(round_key)
                        if method_snapshot is None:
                            raise RuntimeError(
                                f"Missing method snapshot for round {round_key} in archive={state_path}"
                            )
                        method = copy.deepcopy(method_snapshot)
                        method = _move_object_tensors_to_device(method, torch.device("cpu"))
                        if hasattr(method, "encoder") and hasattr(method.encoder, "device"):
                            method.encoder.device = torch.device("cpu")
                        history = method.evaluate(clients, round_states[round_key])
                        mode_histories[str(round_id)] = {
                            str(key): float(value)
                            for key, value in history.items()
                            if _numeric_value(value) is not None
                        }
                    runtime_seconds = time.perf_counter() - start
                    for mode in required_modes:
                        summary_key = _summary_key_for(method_name, mode, include_suffix=True)
                        if (int(seed), summary_key) in completed_pairs:
                            continue
                        metric_key, primary_value = _metric_for_mode(
                            mode_histories[str(max_round)],
                            mode,
                            fallback_key=str(base_row.get("metric_key", "mean_personalized_accuracy")),
                        )
                        row = _build_inference_row(
                            dataset_name=dataset_name,
                            seed=seed,
                            method_name=method_name,
                            cfg_family="hd",
                            cfg=cfg,
                            mode=mode,
                            summary_key=summary_key,
                            checkpoint_metrics=mode_histories,
                            primary_value=primary_value,
                            metric_key=metric_key,
                            runtime_seconds=runtime_seconds,
                            chance_accuracy=chance_accuracy,
                        )
                        rows.append(row)
                        row_cache[(dataset_name, int(seed), summary_key)] = row
                        completed_pairs.add((int(seed), summary_key))
                        print(
                            f"[hd-checkpoints] dataset={dataset_name} seed={seed} method={method_name} "
                            f"mode={mode} R{max_round}_{metric_key}={primary_value:.4f} "
                            f"runtime_seconds={runtime_seconds:.2f}"
                        )
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

                report["datasets"][dataset_name] = dataset_summary_from_rows(
                    rows=rows,
                    method_order=run_config["summary_methods"],
                    round_checkpoints=round_checkpoints,
                    collapse_factor=float(args.collapse_factor),
                    num_classes=int(dataset_num_classes) if dataset_num_classes is not None else num_classes,
                    sampling_records=sampling_records,
                    dataset_spec=dataset_spec,
                )
                save_report(report, args)

        if dataset_num_classes is None:
            raise RuntimeError(f"Failed to resolve num_classes for dataset={dataset_name}")

        report["datasets"][dataset_name] = dataset_summary_from_rows(
            rows=rows,
            method_order=run_config["summary_methods"],
            round_checkpoints=round_checkpoints,
            collapse_factor=float(args.collapse_factor),
            num_classes=int(dataset_num_classes),
            sampling_records=sampling_records,
            dataset_spec=dataset_spec,
        )

    save_report(report, args)
    json_out = Path(args.json_out)
    md_out = Path(args.md_out)
    print(f"saved_json={json_out}")
    print(f"saved_md={md_out}")


if __name__ == "__main__":
    main()
