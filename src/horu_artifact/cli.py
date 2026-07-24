"""Command line interface for artifact preparation and smoke tests."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

from .config import load_config, load_federated_config, load_horu_bootstrap_config, load_horu_round_config
from .datasets.ucihar import prepare_data
from .datasets.ucihar_federated import prepare_data as prepare_ucihar_federated
from .datasets.isolet import prepare_data as prepare_isolet
from .datasets.femnist import prepare_data as prepare_femnist
from .datasets.wisdm import prepare_data as prepare_wisdm
from .datasets.synthetic import prepare_data as prepare_synthetic
from .datasets.controlled_systems import prepare_data as prepare_controlled_systems
from .datasets.ninapro import prepare_data as prepare_ninapro
from .smoke import run_smoke
from .federated.runner import run_federated
from .experiments.accuracy_reporting import write_summary
from .experiments.accuracy_validation import validate_results
from .experiments.accuracy_suite import run_suite
from .experiments.tables123 import reproduce_tables


def main(argv: list[str] | None = None) -> None:
    """Run the artifact CLI."""
    parser = argparse.ArgumentParser(prog="horu_artifact")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-data"); prepare.add_argument("dataset", choices=["ucihar", "isolet", "femnist", "wisdm", "synthetic", "ninapro", "controlled-systems", "all"]); prepare.add_argument("--config"); prepare.add_argument("--data-root", required=True)
    smoke = commands.add_parser("smoke"); smoke.add_argument("--config", required=True); smoke.add_argument("--data-root", required=True); smoke.add_argument("--output", required=True); smoke.add_argument("--device", choices=["cpu", "cuda", "auto"]); smoke.add_argument("--overwrite", action="store_true")
    federated = commands.add_parser("federated"); federated.add_argument("--method", choices=["fedhdc", "hyperfeel", "horu"], required=True); federated.add_argument("--config", required=True); federated.add_argument("--data-root", required=True); federated.add_argument("--output", required=True); federated.add_argument("--device", choices=["cpu", "cuda", "auto"]); federated.add_argument("--overwrite", action="store_true"); federated.add_argument("--resume", action="store_true"); federated.add_argument("--bootstrap-only", action="store_true"); federated.add_argument("--bootstrap-checkpoint"); federated.add_argument("--seed", type=int)
    suite = commands.add_parser("run-suite"); suite.add_argument("--config", required=True); suite.add_argument("--data-root", required=True); suite.add_argument("--output", required=True)
    validate = commands.add_parser("validate-results"); validate.add_argument("--results", required=True); validate.add_argument("--reference")
    table_reproduction = commands.add_parser("reproduce-tables"); table_reproduction.add_argument("--data-root", required=True); table_reproduction.add_argument("--output", required=True); table_reproduction.add_argument("--warmup", type=int, default=5); table_reproduction.add_argument("--repeats", type=int, default=30); table_reproduction.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare-data":
            import yaml
            raw = yaml.safe_load(open(args.config, encoding="utf-8")) if args.config else {}
            sources = raw.get("sources", {}) if isinstance(raw, dict) else {}
            names = [args.dataset] if args.dataset != "all" else ["ucihar", "isolet", "femnist", "wisdm", "synthetic", "ninapro"]
            prepared = {}
            for name in names:
                if name == "ucihar":
                    if name in sources:
                        data = prepare_ucihar_federated(
                            args.data_root,
                            int(raw.get("seed", 42)),
                            sources[name],
                            bool(raw.get("ucihar_preserve_original_split", True)),
                        )
                    else:
                        prepare_data(args.data_root)
                        data = prepare_ucihar_federated(args.data_root, int(raw.get("seed", 42)))
                elif name == "isolet":
                    data = prepare_isolet(
                        args.data_root,
                        sources[name],
                        int(raw.get("seed", 42)),
                        float(raw.get("isolet_alpha", 5.0)),
                        bool(raw.get("isolet_preserve_original_split", False)),
                    )
                elif name == "femnist":
                    data = prepare_femnist(
                        args.data_root,
                        sources[name],
                        int(raw.get("femnist_selection_seed", raw.get("seed", 42))),
                        int(raw.get("femnist_limit_clients", 200)),
                    )
                elif name == "wisdm": data = prepare_wisdm(args.data_root, sources[name], int(raw.get("seed", 0)), raw.get("wisdm_client_ids"), bool(raw.get("wisdm_recover_missing_from_raw", False)))
                elif name == "ninapro": data = prepare_ninapro(args.data_root, sources[name], int(raw.get("seed", 42)))
                elif name == "synthetic":
                    data = prepare_synthetic(
                        args.data_root,
                        sources[name],
                        int(raw.get("seed", 42)),
                        int(raw.get("synthetic_limit_clients", 30)),
                    )
                else:
                    data = prepare_controlled_systems(args.data_root)
                    prepared[name] = {
                        "clients": data.config.clients,
                        "classes": data.config.classes,
                        "samples_per_client": data.config.samples_per_client,
                        "initial_misclassified_per_client": data.config.initial_misclassified_per_client,
                        "hd_dim": data.config.hd_dim,
                    }
                    continue
                prepared[name] = data.statistics()
            print(json.dumps({"status": "prepared", "datasets": prepared}))
        elif args.command == "smoke":
            print(json.dumps(run_smoke(load_config(args.config), args.data_root, args.output, args.device, args.overwrite)))
        elif args.command == "run-suite":
            run_suite(args.config, args.data_root, args.output)
            print(json.dumps(write_summary(args.output)))
        elif args.command == "validate-results":
            print(json.dumps(validate_results(args.results, args.reference)))
        elif args.command == "reproduce-tables":
            print(json.dumps(reproduce_tables(args.data_root, args.output, args.warmup, args.repeats, args.threads)))
        else:
            config = (load_horu_bootstrap_config(args.config) if args.bootstrap_only else load_horu_round_config(args.config)) if args.method == "horu" else load_federated_config(args.config)
            if args.seed is not None: config = replace(config, seed=args.seed)
            if args.method != config.method: raise ValueError("--method must match config method")
            if args.method == "horu" and args.bootstrap_only and args.bootstrap_checkpoint: raise ValueError("--bootstrap-checkpoint is for recurring HoRU rounds")
            if args.method == "horu" and not args.bootstrap_only and not args.bootstrap_checkpoint: raise ValueError("HoRU recurring rounds require --bootstrap-checkpoint")
            print(json.dumps(run_federated(config, args.data_root, args.output, args.device, args.overwrite, args.resume, args.bootstrap_checkpoint)))
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError, OSError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__": main()
