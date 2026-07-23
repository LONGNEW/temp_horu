"""Smoke configuration parsing and validation."""

from dataclasses import asdict, dataclass
from pathlib import Path
import yaml


@dataclass(frozen=True)
class SmokeConfig:
    """Explicit parameters for the local UCI-HAR prototype smoke run."""
    dataset: str
    subject_ids: list[int]
    test_ratio: float
    seed: int
    hd_dim: int
    learning_rate: float
    local_epochs: int
    batch_size: int
    device: str = "cpu"
    similarity: str = "cosine"
    update_mode: str = "samplewise"
    normalize_prototypes: bool = True
    normalize_update_hypervectors: bool = False

    def to_dict(self) -> dict:
        """Return a serializable config mapping."""
        return asdict(self)


def load_config(path: str | Path) -> SmokeConfig:
    """Load and validate a YAML smoke configuration."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict): raise ValueError("config must be a YAML mapping")
    try: config = SmokeConfig(**raw)
    except TypeError as error: raise ValueError(f"invalid smoke config fields: {error}") from error
    if config.dataset != "ucihar" or len(config.subject_ids) != 3 or len(set(config.subject_ids)) != 3: raise ValueError("smoke config requires exactly three distinct UCI-HAR subjects")
    if any(not 1 <= x <= 30 for x in config.subject_ids): raise ValueError("subject IDs must be in 1..30")
    if not 0 < config.test_ratio < 1 or config.hd_dim <= 0 or config.learning_rate <= 0 or config.local_epochs <= 0 or config.batch_size <= 0: raise ValueError("invalid numeric smoke setting")
    if config.device not in {"cpu", "cuda", "auto"}: raise ValueError("device must be cpu, cuda, or auto")
    if config.similarity not in {"cosine", "dot"}: raise ValueError("similarity must be cosine or dot")
    if config.update_mode not in {"samplewise", "hdzoo_batch"}: raise ValueError("update_mode must be samplewise or hdzoo_batch")
    if not isinstance(config.normalize_prototypes, bool): raise ValueError("normalize_prototypes must be a boolean")
    if not isinstance(config.normalize_update_hypervectors, bool): raise ValueError("normalize_update_hypervectors must be a boolean")
    return config


@dataclass(frozen=True)
class FederatedConfig:
    """Explicit, reproducible settings for the FedHDC simulation."""
    dataset: str
    subject_ids: list[int]
    test_ratio: float
    seed: int
    hd_dim: int
    learning_rate: float
    local_epochs: int
    batch_size: int
    rounds: int
    device: str = "cpu"
    method: str = "fedhdc"
    similarity: str = "dot"
    normalize_update_hypervectors: bool = True
    normalize_prototypes: bool = True
    server_aggregation: str = "weighted_average"
    implementation_mode: str = "artifact"
    participation: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)


def load_federated_config(path: str | Path) -> FederatedConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict): raise ValueError("config must be a YAML mapping")
    try: config = FederatedConfig(**raw)
    except TypeError as error: raise ValueError(f"invalid federated config fields: {error}") from error
    if config.dataset != "ucihar" or not config.subject_ids or len(set(config.subject_ids)) != len(config.subject_ids): raise ValueError("federated config requires distinct UCI-HAR subjects")
    if any(not 1 <= x <= 30 for x in config.subject_ids): raise ValueError("subject IDs must be in 1..30")
    if not 0 < config.test_ratio < 1 or min(config.hd_dim, config.learning_rate, config.local_epochs, config.batch_size, config.rounds) <= 0: raise ValueError("invalid numeric federated setting")
    # T002 smoke used 16; T006 fixes the full-suite loader chunk to 32.
    if config.batch_size not in {16, 32}: raise ValueError("federated batch_size must be T002 smoke 16 or T006 suite 32")
    if not 0 < config.participation <= 1: raise ValueError("participation must be in (0, 1]")
    if config.method == "fedhdc":
        if config.similarity != "dot" or not config.normalize_update_hypervectors or not config.normalize_prototypes or config.server_aggregation != "weighted_average": raise ValueError("FedHDC requires dot similarity, unit update targets, row normalization, and weighted_average aggregation")
    elif config.method == "hyperfeel":
        if config.implementation_mode not in {"paper_faithful", "diagnostic"} or config.similarity != "dot" or config.server_aggregation != "sum": raise ValueError("HyperFeel requires paper_faithful/diagnostic mode, dot similarity, and sum aggregation")
        if config.implementation_mode == "paper_faithful" and (config.normalize_update_hypervectors or config.normalize_prototypes): raise ValueError("paper-faithful HyperFeel requires raw-Q updates and no row normalization")
    else: raise ValueError("method must be fedhdc or hyperfeel")
    if config.device not in {"cpu", "cuda", "auto"}: raise ValueError("device must be cpu, cuda, or auto")
    return config


@dataclass(frozen=True)
class HoruBootstrapConfig:
    """One-time HoRU bootstrap settings; this is not a training configuration."""
    method: str
    dataset: str
    subject_ids: list[int]
    hd_dim: int
    common_rank: int
    global_rank: int
    personal_rank: int
    personal_basis_policy: str
    seed: int
    device: str = "cpu"
    bootstrap_only: bool = True
    # Existing UCI-HAR artifact split value.  It is explicit in resolved output.
    test_ratio: float = 0.3

    def to_dict(self) -> dict:
        return asdict(self)


def load_horu_bootstrap_config(path: str | Path) -> HoruBootstrapConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("config must be a YAML mapping")
    try:
        config = HoruBootstrapConfig(**raw)
    except TypeError as error:
        raise ValueError(f"invalid HoRU bootstrap config fields: {error}") from error
    if config.method != "horu":
        raise ValueError("HoRU bootstrap config requires method: horu")
    if config.dataset != "ucihar" or not config.subject_ids or len(set(config.subject_ids)) != len(config.subject_ids):
        raise ValueError("HoRU bootstrap requires distinct UCI-HAR subjects")
    if any(not 1 <= x <= 30 for x in config.subject_ids):
        raise ValueError("subject IDs must be in 1..30")
    if min(config.hd_dim, config.common_rank, config.personal_rank) <= 0 or config.global_rank < 0:
        raise ValueError("HoRU ranks and hd_dim must be positive (global_rank may be zero)")
    if config.common_rank + config.global_rank > config.hd_dim:
        raise ValueError("common_rank + global_rank must not exceed hd_dim")
    if config.personal_rank > config.hd_dim:
        raise ValueError("personal_rank must not exceed hd_dim")
    if config.personal_basis_policy not in {"reduced_svd", "full_svd"}:
        raise ValueError("personal_basis_policy must be reduced_svd or full_svd")
    if config.personal_basis_policy == "reduced_svd" and config.personal_rank > 6:
        raise ValueError("reduced_svd requires personal_rank <= min(num_classes=6, hd_dim)")
    if not 0 < config.test_ratio < 1 or config.device not in {"cpu", "cuda", "auto"}:
        raise ValueError("invalid test_ratio or device")
    if not config.bootstrap_only:
        raise ValueError("T004 supports bootstrap_only: true only")
    return config


@dataclass(frozen=True)
class HoruRoundConfig:
    """Explicit recurring-round HoRU smoke configuration."""
    method: str
    dataset: str
    subject_ids: list[int]
    hd_dim: int
    common_rank: int
    global_rank: int
    personal_rank: int
    personal_basis_policy: str
    seed: int
    rounds: int
    local_epochs: int
    batch_size: int
    eta_shared: float
    eta_personal: float
    eta_global: float
    provenance: str
    run_profile: str = "smoke"
    device: str = "cpu"
    test_ratio: float = 0.3
    gate_alpha: float = 1.0
    gate_min: float = 0.1
    gate_max: float = 0.9

    def to_dict(self) -> dict:
        return asdict(self)


def load_horu_round_config(path: str | Path) -> HoruRoundConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("config must be a YAML mapping")
    try:
        config = HoruRoundConfig(**raw)
    except TypeError as error:
        raise ValueError(f"invalid HoRU round config fields: {error}") from error
    if config.method != "horu" or config.dataset != "ucihar":
        raise ValueError("HoRU round config requires method: horu and dataset: ucihar")
    if not config.subject_ids or len(set(config.subject_ids)) != len(config.subject_ids) or any(not 1 <= x <= 30 for x in config.subject_ids):
        raise ValueError("HoRU recurring config requires distinct UCI-HAR subjects")
    if min(config.hd_dim, config.common_rank, config.personal_rank, config.rounds, config.local_epochs, config.batch_size, config.eta_shared, config.eta_personal, config.eta_global) <= 0 or config.global_rank < 0:
        raise ValueError("invalid HoRU round numeric setting")
    if config.common_rank + config.global_rank > config.hd_dim or config.personal_rank > config.hd_dim:
        raise ValueError("invalid HoRU rank setting")
    if config.personal_basis_policy not in {"reduced_svd", "full_svd"}:
        raise ValueError("invalid personal_basis_policy")
    if config.personal_basis_policy == "reduced_svd" and config.personal_rank > 6:
        raise ValueError("reduced_svd requires personal_rank <= 6")
    if config.run_profile == "smoke" and (config.batch_size != 16 or config.rounds != 2 or config.provenance != "USER_SPECIFIED_SMOKE"):
        raise ValueError("T005 smoke requires batch_size: 16, rounds: 2, and USER_SPECIFIED_SMOKE provenance")
    if config.run_profile not in {"smoke", "paper_rank_diagnostic"}:
        raise ValueError("run_profile must be smoke or paper_rank_diagnostic")
    if config.gate_alpha < 0 or not 0 <= config.gate_min <= config.gate_max <= 1:
        raise ValueError("invalid RowGate setting")
    if not 0 < config.test_ratio < 1 or config.device not in {"cpu", "cuda", "auto"}:
        raise ValueError("invalid test_ratio or device")
    return config
