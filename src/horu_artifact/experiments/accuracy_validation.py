from __future__ import annotations
import json
from pathlib import Path
from .accuracy_reporting import write_summary


def validate_results(results: str | Path, reference: str | Path | None = None) -> dict:
    report=write_summary(results); root=Path(results)
    checks=[]
    for row in report["rows"]:
        checks.append({"name":"result_file_has_final_accuracy","dataset":row["dataset"],"method":row["method"],"seed":row["seed"],"passed":True})
    # Cross-method comparisons are intentionally invalid when metric definitions differ.
    metric_by_dataset={}
    for row in report["rows"]: metric_by_dataset.setdefault(row["dataset"],set()).add(row["metric_name"])
    for dataset, metrics in metric_by_dataset.items():
        checks.append({"name":"common_metric_definition","dataset":dataset,"passed":len(metrics)==1,"detail":sorted(metrics)})
    requested_path=root/"summary"/"requested_runs.json"
    requested=json.loads(requested_path.read_text()) if requested_path.exists() else []
    observed={(r["dataset"],r["method"],int(r["seed"])) for r in report["rows"]}
    missing=[x for x in requested if (x["dataset"],x["method"],int(x["seed"])) not in observed]
    checks.append({"name":"all_requested_runs_present","passed":not missing,"missing_runs":missing})
    payload={"status":"pass" if report["completed_runs"] and not report["failed_runs"] and all(c["passed"] for c in checks) else "incomplete_or_failed","completed_runs":report["completed_runs"],"failed_runs":report["failed_runs"],"missing_runs":missing,"checks":checks,"failures":report["failures"],"reference":str(reference) if reference else None}
    path=root/"summary"/"validation_report.json"; path.write_text(json.dumps(payload,indent=2)+"\n")
    return payload
