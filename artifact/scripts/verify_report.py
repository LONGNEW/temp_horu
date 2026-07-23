#!/usr/bin/env python3
"""Check that a generated artifact report matches its declared manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_prototype_input_provenance(provenance: dict[str, object], failures: list[str]) -> None:
    if "source_archives" in provenance:
        verify_multi_archive_prototype_provenance(provenance, failures)
        return
    record_value = provenance.get("acquisition_record")
    archive_value = provenance.get("source_archive")
    source_root_value = provenance.get("source_root")
    if not isinstance(record_value, str) or not isinstance(archive_value, str) or not isinstance(source_root_value, str):
        failures.append("prototype report lacks acquisition-record, archive, or source-root provenance")
        return
    record_path = Path(record_value)
    archive_path = Path(archive_value)
    if not record_path.is_file():
        failures.append(f"prototype acquisition record is unavailable: {record_path}")
        return
    if not archive_path.is_file():
        failures.append(f"prototype source archive is unavailable: {archive_path}")
        return
    if provenance.get("acquisition_record_sha256") != sha256(record_path):
        failures.append("prototype acquisition record hash differs from the report")
    if provenance.get("source_archive_sha256") != sha256(archive_path):
        failures.append("prototype source archive hash differs from the report")
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        failures.append("prototype acquisition record is not valid JSON")
        return
    if record.get("result_status") != "PROTOTYPE_INPUT_ONLY":
        failures.append("prototype acquisition record has an unexpected result status")
    if record.get("manifest_sha256") != provenance.get("manifest_sha256"):
        failures.append("prototype acquisition record manifest hash differs from the report")
    if record.get("archive_sha256") != provenance.get("source_archive_sha256"):
        failures.append("prototype acquisition record archive hash differs from the report")
    if Path(str(record.get("extracted_root", ""))).resolve() != Path(source_root_value).resolve():
        failures.append("prototype acquisition record extracted root differs from the report")
    if record.get("nested_archive_member") != provenance.get("acquisition_nested_archive_member"):
        failures.append("prototype nested archive member differs from acquisition record")


def verify_multi_archive_prototype_provenance(provenance: dict[str, object], failures: list[str]) -> None:
    record_value = provenance.get("acquisition_record")
    source_root_value = provenance.get("source_root")
    report_archives = provenance.get("source_archives")
    if not isinstance(record_value, str) or not isinstance(source_root_value, str) or not isinstance(report_archives, dict):
        failures.append("multi-archive prototype report lacks input provenance")
        return
    record_path = Path(record_value)
    if not record_path.is_file():
        failures.append(f"prototype acquisition record is unavailable: {record_path}")
        return
    if provenance.get("acquisition_record_sha256") != sha256(record_path):
        failures.append("prototype acquisition record hash differs from the report")
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        failures.append("prototype acquisition record is not valid JSON")
        return
    if record.get("result_status") != "PROTOTYPE_INPUT_ONLY":
        failures.append("prototype acquisition record has an unexpected result status")
    if record.get("manifest_sha256") != provenance.get("manifest_sha256"):
        failures.append("prototype acquisition record manifest hash differs from the report")
    if Path(str(record.get("extracted_root", ""))).resolve() != Path(source_root_value).resolve():
        failures.append("prototype acquisition record extracted root differs from the report")
    record_archives = record.get("archives")
    if record_archives != report_archives:
        failures.append("prototype archive inventory differs from acquisition record")
        return
    for role, values in report_archives.items():
        if not isinstance(values, dict):
            failures.append(f"prototype archive inventory entry is invalid: {role}")
            continue
        archive = Path(str(values.get("archive", "")))
        extracted = Path(str(values.get("extracted", "")))
        if not archive.is_file() or not extracted.is_file():
            failures.append(f"prototype archive/input is unavailable: {role}")
            continue
        if values.get("archive_sha256") != sha256(archive) or values.get("extracted_sha256") != sha256(extracted):
            failures.append(f"prototype archive/input hash differs from report: {role}")


def verify_reconstruction_screening_provenance(provenance: dict[str, object], failures: list[str]) -> None:
    record_value = provenance.get("acquisition_record")
    source_root_value = provenance.get("source_root")
    outer_value = provenance.get("outer_archive")
    nested_value = provenance.get("nested_archive")
    if not all(isinstance(value, str) for value in (record_value, source_root_value, outer_value, nested_value)):
        failures.append("reconstruction report lacks acquisition-record or archive provenance")
        return
    record_path, outer, nested = Path(record_value), Path(outer_value), Path(nested_value)
    if not record_path.is_file() or not outer.is_file() or not nested.is_file():
        failures.append("reconstruction acquisition record or archive is unavailable")
        return
    if provenance.get("acquisition_record_sha256") != sha256(record_path):
        failures.append("reconstruction acquisition record hash differs from report")
    if provenance.get("outer_archive_sha256") != sha256(outer):
        failures.append("reconstruction outer archive hash differs from report")
    if provenance.get("nested_archive_sha256") != sha256(nested):
        failures.append("reconstruction nested archive hash differs from report")
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        failures.append("reconstruction acquisition record is not valid JSON")
        return
    if record.get("result_status") != "RECONSTRUCTION_INPUT_ONLY":
        failures.append("reconstruction acquisition record has an unexpected result status")
    for key in ("outer_archive", "outer_archive_sha256", "nested_archive", "nested_archive_sha256"):
        if record.get(key) != provenance.get(key):
            failures.append(f"reconstruction {key} differs from acquisition record")
    if not str(nested.resolve()).startswith(str(Path(source_root_value).resolve())):
        failures.append("reconstruction nested archive is outside the declared source root")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    report = json.loads(args.report.read_text(encoding="utf-8"))
    expected = manifest.get("expected_report_contract")
    if expected is None:
        protocol = manifest.get("protocol", {})
        data = manifest.get("data", {})
        expected = {
            "datasets": [data["dataset"]] if "dataset" in data else [],
            "methods": protocol.get("methods", []),
            "round_checkpoints": protocol.get("round_checkpoints", []),
            "seeds": [protocol["seed"]] if "seed" in protocol else [],
        }
    run_config = report.get("run_config", {})
    failures: list[str] = []
    if sorted(report.get("datasets", {}).keys()) != sorted(expected.get("datasets", [])):
        failures.append("dataset set differs from manifest")
    observed_methods = sorted(
        {
            str(row.get("method"))
            for dataset_report in report.get("datasets", {}).values()
            for row in dataset_report.get("rows", [])
        }
    )
    if observed_methods != sorted(expected.get("methods", [])):
        failures.append("method set differs from manifest")
    if report.get("round_checkpoints") != expected.get("round_checkpoints"):
        failures.append("round checkpoints differ from manifest")
    if report.get("seeds") != expected.get("seeds"):
        failures.append("seed list differs from manifest")
    provenance = report.get("artifact_provenance", {})
    if provenance.get("result_status") != manifest.get("result_status"):
        failures.append("artifact result status is missing or differs from manifest")
    if provenance.get("manifest_sha256") != hashlib.sha256(args.manifest.read_bytes()).hexdigest():
        failures.append("artifact manifest hash is missing or differs from the manifest used for verification")
    if provenance.get("execution_mode") != "fresh":
        failures.append("artifact report was reused rather than generated fresh")
    if manifest.get("result_status") == "PROTOTYPE_ONLY":
        verify_prototype_input_provenance(provenance, failures)
    if manifest.get("result_status") == "RECONSTRUCTION_SCREENING_ONLY":
        verify_reconstruction_screening_provenance(provenance, failures)
    if failures:
        print("REPORT VERIFICATION FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 2
    print(f"REPORT VERIFIED: {manifest['result_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
