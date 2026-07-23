"""Deterministic Table-III controlled systems fixture.

The fixture stores one client template and logically reuses it for every
client.  This keeps the on-disk artifact small while preserving the exact
per-client workload specified by the paper.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import torch

from ..hdc.prototype import PrototypeMemory


@dataclass(frozen=True)
class ControlledSystemsConfig:
    clients: int = 50
    classes: int = 50
    samples_per_client: int = 1000
    initial_misclassified_per_client: int = 500
    hd_dim: int = 2000
    batch_size: int = 32
    local_epochs: int = 1
    common_rank: int = 24
    global_rank: int = 8
    personal_rank: int = 64
    seed: int = 0

    def validate(self) -> None:
        if min(self.clients, self.classes, self.samples_per_client, self.hd_dim, self.batch_size, self.local_epochs) <= 0:
            raise ValueError("controlled systems dimensions must be positive")
        if not 0 <= self.initial_misclassified_per_client <= self.samples_per_client:
            raise ValueError("initial misclassified count must be within the client sample count")
        correct = self.samples_per_client - self.initial_misclassified_per_client
        if self.initial_misclassified_per_client % self.classes or correct % self.classes:
            raise ValueError("misclassified and correct sample counts must each be divisible by classes")
        if self.classes > self.hd_dim:
            raise ValueError("classes must not exceed hd_dim")
        if self.common_rank + self.global_rank > self.hd_dim or self.personal_rank > self.hd_dim:
            raise ValueError("controlled systems ranks must not exceed hd_dim")


@dataclass
class ControlledSystemsFixture:
    config: ControlledSystemsConfig
    train_h: torch.Tensor
    train_y: torch.Tensor
    initial_prototypes: torch.Tensor
    initial_predictions: torch.Tensor

    def clients(self) -> dict[int, dict[str, torch.Tensor]]:
        """Return logical clients sharing immutable input tensors."""
        return {
            client_id: {
                "train_h": self.train_h,
                "train_y": self.train_y,
                "initial_prototypes": self.initial_prototypes.clone(),
            }
            for client_id in range(self.config.clients)
        }


def build_fixture(config: ControlledSystemsConfig = ControlledSystemsConfig()) -> ControlledSystemsFixture:
    """Build a dense fp32 fixture with an exact initial mistake count.

    Correct queries equal their class anchor.  Misclassified queries equal the
    next class anchor while retaining the original label.  Against the supplied
    initial prototype memory this makes the desired split exact, without
    relying on noise thresholds or a trained initializer.
    """
    config.validate()
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    random_basis = torch.randn((config.hd_dim, config.classes), generator=generator, dtype=torch.float32)
    anchors = torch.linalg.qr(random_basis, mode="reduced").Q.T.contiguous()

    wrong_per_class = config.initial_misclassified_per_client // config.classes
    correct_per_class = (config.samples_per_client - config.initial_misclassified_per_client) // config.classes
    wrong_labels = torch.arange(config.classes, dtype=torch.long).repeat_interleave(wrong_per_class)
    correct_labels = torch.arange(config.classes, dtype=torch.long).repeat_interleave(correct_per_class)
    wrong_queries = anchors[(wrong_labels + 1) % config.classes]
    # The correct half compensates the shifted half so the class-wise mean is
    # exactly its anchor: mean(anchor[next], 2*anchor[label]-anchor[next])
    # equals anchor[label].  Thus bootstrap reconstructs the supplied initial
    # prototypes while the initial classifier still makes exactly the desired
    # number of mistakes.
    correct_queries = (2.0 * anchors[correct_labels]) - anchors[(correct_labels + 1) % config.classes]
    train_h = torch.cat([wrong_queries, correct_queries]).contiguous()
    train_y = torch.cat([wrong_labels, correct_labels]).contiguous()
    predictions = PrototypeMemory(anchors).predict(train_h, "dot")
    mistakes = int((predictions != train_y).sum().item())
    if mistakes != config.initial_misclassified_per_client:
        raise RuntimeError(f"controlled fixture produced {mistakes} initial mistakes, expected {config.initial_misclassified_per_client}")
    return ControlledSystemsFixture(config, train_h, train_y, anchors, predictions)


def _tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def prepare_data(data_root: str | Path, config: ControlledSystemsConfig = ControlledSystemsConfig()) -> ControlledSystemsFixture:
    """Materialize the compact, one-client template under ``data_root``."""
    fixture = build_fixture(config)
    target = Path(data_root) / "controlled_systems"
    target.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": asdict(config),
        "train_h": fixture.train_h,
        "train_y": fixture.train_y,
        "initial_prototypes": fixture.initial_prototypes,
        "initial_predictions": fixture.initial_predictions,
    }
    torch.save(payload, target / "fixture.pt")
    manifest = {
        "source": "paper_table_iii_controlled_fixture",
        "storage": "one immutable client template logically reused by all clients",
        "dtype": "float32",
        "config": asdict(config),
        "initial_misclassified_per_client": int((fixture.initial_predictions != fixture.train_y).sum().item()),
        "train_h_sha256": _tensor_sha256(fixture.train_h),
        "train_y_sha256": _tensor_sha256(fixture.train_y),
        "initial_prototypes_sha256": _tensor_sha256(fixture.initial_prototypes),
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return fixture


def load_fixture(data_root: str | Path) -> ControlledSystemsFixture:
    path = Path(data_root) / "controlled_systems" / "fixture.pt"
    if not path.is_file():
        raise FileNotFoundError(f"controlled systems fixture not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = ControlledSystemsConfig(**payload["config"])
    fixture = ControlledSystemsFixture(config, payload["train_h"], payload["train_y"], payload["initial_prototypes"], payload["initial_predictions"])
    mistakes = int((PrototypeMemory(fixture.initial_prototypes).predict(fixture.train_h, "dot") != fixture.train_y).sum().item())
    if mistakes != config.initial_misclassified_per_client:
        raise ValueError("controlled systems fixture no longer satisfies its initial mistake invariant")
    return fixture
