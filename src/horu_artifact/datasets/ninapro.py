"""NinaPro DB1 subject loader with repetition-held-out EMG/glove windows."""
from __future__ import annotations
import hashlib
from pathlib import Path
import torch
from scipy.io import loadmat
from .federated import ClientData, FederatedDataset, write_cache


def _cap(indices: torch.Tensor, labels: torch.Tensor, cap: int, seed: int) -> torch.Tensor:
    """Cap a client total while retaining its label proportions."""
    if indices.numel() <= cap: return indices.sort().values
    counts = torch.bincount(labels[indices], minlength=52)
    ideal = counts.float() * (cap / indices.numel())
    take = torch.floor(ideal).long()
    remaining = cap - int(take.sum())
    # Deterministic largest-remainder allocation; label id breaks ties.
    order = sorted(range(52), key=lambda y: (float(ideal[y] - take[y]), -y), reverse=True)
    for y in order[:remaining]: take[y] += 1
    kept = []
    for y in torch.unique(labels, sorted=True).tolist():
        ix = indices[labels[indices] == y]
        if ix.numel() > take[y]:
            g = torch.Generator().manual_seed(seed + int(y) * 1009)
            ix = ix[torch.randperm(ix.numel(), generator=g)[:take[y]]]
        kept.append(ix)
    return torch.cat(kept).sort().values


def _subject(directory: Path, subject: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    windows, labels, reps = [], [], []
    for path in sorted(directory.glob(f"S{subject}_A1_E*.mat")):
        m = loadmat(path); signal = torch.tensor(__import__("numpy").concatenate((m["emg"], m["glove"]), axis=1), dtype=torch.float32)
        target = torch.tensor(m["restimulus"].reshape(-1), dtype=torch.long)
        repetition = torch.tensor(m["repetition"].reshape(-1), dtype=torch.long)
        start = 0
        # Boundaries of either gesture or repetition cannot be crossed.
        for end in range(1, target.numel() + 1):
            if end < target.numel() and target[end] == target[start] and repetition[end] == repetition[start]: continue
            if target[start] > 0:
                length = end - start
                for pos in range(start, start + (length // 20) * 20, 20):
                    windows.append(signal[pos:pos + 20].reshape(-1)); labels.append(target[start] - 1); reps.append(repetition[start])
            start = end
    return torch.stack(windows), torch.tensor(labels), torch.tensor(reps)


def prepare_data(data_root: str | Path, source_root: str | Path, seed: int = 0) -> FederatedDataset:
    source = Path(source_root)
    if len(list(source.glob("S*_A1_E*.mat"))) < 81: raise FileNotFoundError("NinaPro DB1 requires 27 subjects × 3 MAT files")
    raw_clients, train_blocks = {}, []
    for subject in range(1, 28):
        x, y, r = _subject(source, subject)
        if x.shape[1] != 640 or y.numel() == 0 or y.max().item() >= 52: raise ValueError(f"NinaPro subject {subject} contract mismatch")
        train = torch.nonzero(~torch.isin(r, torch.tensor([2,5,7])), as_tuple=False).flatten(); test = torch.nonzero(torch.isin(r, torch.tensor([2,5,7])), as_tuple=False).flatten()
        train, test = _cap(train, y, 5000, seed + subject), _cap(test, y, 1000, seed + subject)
        raw_clients[f"{subject:03d}"] = (x, y, train, test); train_blocks.append(x[train])
    pooled = torch.cat(train_blocks); mean, std = pooled.mean(0), pooled.std(0).clamp_min(1e-12)
    clients = {}
    for subject, (x,y,train,test) in raw_clients.items():
        base = int(subject) * 10_000_000; ids = torch.arange(base, base + y.numel(), dtype=torch.long)
        clients[subject] = ClientData((x[train]-mean)/std, y[train], (x[test]-mean)/std, y[test], ids[train], ids[test])
    digest = hashlib.sha256()
    for path in sorted(source.glob("S*_A1_E*.mat")):
        with path.open("rb") as f:
            for block in iter(lambda:f.read(1024*1024), b""): digest.update(block)
    manifest = {"source": str(source), "license": "NinaPro DB1", "raw_sha256": digest.hexdigest(), "parser": "ninapro_db1_mat_window_v1",
                "clients":27,"features":640,"classes":52,"channels":{"emg":10,"glove":22},"window":{"hz":100,"samples":20,"overlap":False},
                "split":{"test_repetitions":[2,5,7],"boundary_policy":"gesture_and_repetition"},"normalization":"pooled_train_only_standardization",
                "cap":{"train":5000,"test":1000},"seed":seed,"provenance":"USER_SPECIFIED_NINAPRO_REPETITION_SPLIT"}
    return write_cache(FederatedDataset("ninapro", clients, 640, 52, manifest), data_root)
