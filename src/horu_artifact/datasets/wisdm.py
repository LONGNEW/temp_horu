"""WISDM phone-accelerometer ARFF loader using the fixed 43 basic features."""
from __future__ import annotations
import hashlib
import math
import zipfile
from pathlib import Path
import torch
from .federated import ClientData, FederatedDataset, stratified_split, write_cache

ACTIVITIES = [*"ABCDEFGHIJKLM", *"OPQRS"]


def _rows(bundle: zipfile.ZipFile, member: str) -> tuple[torch.Tensor, torch.Tensor]:
    lines = bundle.read(member).decode("utf-8", errors="strict").splitlines()
    data_at = next(i for i,line in enumerate(lines) if line.strip().lower() == "@data") + 1
    x, y = [], []
    for line in lines[data_at:]:
        if not line.strip() or line.startswith("%"): continue
        fields = [v.strip() for v in line.split(",")]
        if len(fields) < 45 or fields[0] not in ACTIVITIES: raise ValueError(f"malformed WISDM ARFF row: {member}")
        # activity, 30 bins, 12 basic axis statistics, ..., resultant, subject
        values = fields[1:31] + fields[31:43] + [fields[-2]]
        try: x.append([float(v) for v in values]); y.append(ACTIVITIES.index(fields[0]))
        except ValueError: continue
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def _raw_rows(bundle: zipfile.ZipFile, member: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover the 43 selected arffmagic features from official raw data.

    This implements the basic-feature subset of the arffmagic C++ source
    bundled with the UCI archive: 20 Hz, non-overlapping 200-row windows,
    its fixed-bin rule, and its (nonstandard) standard-deviation calculation.
    It deliberately omits MFCC/correlation fields which are not in T006.
    """
    records = []
    for line in bundle.read(member).decode("utf-8", errors="strict").splitlines():
        fields = line.strip().rstrip(";").split(",")
        if len(fields) != 6 or fields[1] not in ACTIVITIES: continue
        records.append((fields[1], [float(v) for v in fields[3:6]]))
    x, y = [], []
    for start in range(0, len(records), 200):
        window = records[start:start + 200]
        if len(window) < 180: continue
        # arffmagic emits a complete chunk when it sees the following record.
        label = records[min(start + 200, len(records) - 1)][0]
        axes = list(zip(*(row for _, row in window)))
        features = []
        for values in axes:
            bins = [0] * 10
            for value in values:
                scaled = value / 2.5
                index = int(math.floor(scaled)) + 2 if -1 <= scaled <= 7 else (9 if scaled > 7 else 0)
                bins[index] += 1
            features.extend(count / len(values) for count in bins)
        averages = [sum(values) / len(values) for values in axes]
        features.extend(averages)
        for values in axes:
            peaks = ([] if values[0] <= values[1] else [0])
            peaks.extend(i for i in range(1, len(values) - 2) if values[i] > values[i - 1] and values[i] > values[i + 1])
            if values[-1] > values[-2]: peaks.append(len(values) - 1)
            distances = [peaks[i + 1] - peaks[i] for i in range(max(0, len(peaks) - 2))]
            features.append((sum(distances) / len(distances)) * 10 if distances else 0.0)
        for values, average in zip(axes, averages): features.append(sum(abs(v - average) for v in values) / len(values))
        for values, average in zip(axes, averages): features.append(math.sqrt(sum((v - average) ** 2 for v in values)) / len(values))
        features.append(sum(math.sqrt(a*a + b*b + c*c) for a,b,c in zip(*axes)) / len(window))
        x.append(features); y.append(ACTIVITIES.index(label))
    if not x: raise ValueError(f"WISDM raw recovery produced no windows: {member}")
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def prepare_data(data_root: str | Path, archive: str | Path, seed: int = 0, client_ids: list[int] | None = None, recover_missing_from_raw: bool = False) -> FederatedDataset:
    archive = Path(archive)
    if not archive.is_file(): raise FileNotFoundError(f"WISDM archive missing: {archive}")
    # The original range misses transformed subject 1614 in the supplied
    # archive.  The user requested 1599 as the 51st client replacement.
    expected_ids = client_ids or [1599, *[subject for subject in range(1600, 1651) if subject != 1614]]
    if len(expected_ids) != 51 or len(set(expected_ids)) != 51:
        raise ValueError("WISDM client_ids must contain exactly 51 unique subjects")
    clients, all_train = {}, []
    with zipfile.ZipFile(archive) as bundle:
        by_subject = {int(Path(x).name.split("_")[1]): x for x in bundle.namelist() if "/arff_files/phone/accel/data_" in x and x.endswith("_accel_phone.arff")}
        missing = sorted(set(expected_ids) - set(by_subject))
        if missing and not recover_missing_from_raw: raise FileNotFoundError(f"WISDM transformed phone-accelerometer ARFF missing requested subjects: {missing}")
        recovered = []
        for i, subject in enumerate(expected_ids):
            member = by_subject.get(subject)
            if member is None:
                raw_member = next((x for x in bundle.namelist() if x.endswith(f"/raw/phone/accel/data_{subject}_accel_phone.txt")), None)
                if raw_member is None: raise FileNotFoundError(f"WISDM raw phone accelerometer source also missing subject {subject}")
                x, y = _raw_rows(bundle, raw_member); recovered.append({"subject":subject,"raw_member":raw_member,"algorithm":"archive_arffmagic_basic43_reimplementation_v1"})
            else:
                x, y = _rows(bundle, member)
            if x.numel() == 0: raise ValueError(f"WISDM client contains no finite rows: {member}")
            finite = torch.isfinite(x).all(dim=1); dropped = 1 - finite.float().mean().item()
            if dropped > .01: raise ValueError(f"WISDM nonfinite removal exceeds 1%: {member}")
            x, y = x[finite], y[finite]
            train, test = stratified_split(y, .3, seed + i)
            # The official requirement is a *client total* cap, stratified by class.
            train, test = _stratified_cap(train, y, 5000, seed + i), _stratified_cap(test, y, 1000, seed + i)
            ids = torch.arange(i * 10_000_000, i * 10_000_000 + y.numel(), dtype=torch.long)
            clients[f"{subject}"] = ClientData(x[train], y[train], x[test], y[test], ids[train], ids[test])
            all_train.append(x[train])
    pooled = torch.cat(all_train); mean, std = pooled.mean(0), pooled.std(0).clamp_min(1e-12)
    for client in clients.values():
        client.train_x = (client.train_x - mean) / std; client.test_x = (client.test_x - mean) / std
    manifest = {"source": str(archive), "license": "WISDM", "raw_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "parser": "phone_accel_arff_basic43_v1", "clients": 51, "features": 43, "classes": 18,
                "activity_order": ACTIVITIES, "client_ids": expected_ids, "client_set_provenance": "USER_SPECIFIED_1599_REPLACES_MISSING_1614", "raw_recovery": recovered, "raw_recovery_provenance": "USER_SPECIFIED_RAW_RECOVERY" if recovered else "NOT_USED", "split": "client_internal_stratified_70_30", "seed": seed,
                "normalization": "pooled_train_only_standardization", "cap": {"train":5000,"test":1000},
                "provenance": "USER_SPECIFIED_WISDM_BASIC_FEATURES"}
    return write_cache(FederatedDataset("wisdm", clients, 43, 18, manifest), data_root)


def _stratified_cap(indices: torch.Tensor, labels: torch.Tensor, cap: int, seed: int) -> torch.Tensor:
    if indices.numel() <= cap: return indices.sort().values
    counts = torch.bincount(labels[indices], minlength=18); ideal = counts.float() * (cap / indices.numel()); take = torch.floor(ideal).long()
    for y in sorted(range(18), key=lambda k: (float(ideal[k] - take[k]), -k), reverse=True)[:cap-int(take.sum())]: take[y] += 1
    output=[]
    for y in range(18):
        ix=indices[labels[indices] == y]
        if ix.numel() > take[y]:
            ix = ix[torch.randperm(ix.numel(), generator=torch.Generator().manual_seed(seed + y * 1009))[:take[y]]]
        output.append(ix)
    return torch.cat(output).sort().values
