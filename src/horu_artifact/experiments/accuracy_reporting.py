from __future__ import annotations
import csv
import json
from pathlib import Path
from statistics import mean, stdev


def write_summary(output: str | Path) -> dict:
    root = Path(output); rows = []; failures = []
    for path in sorted((root / "runs").glob("*/*/*/result.json")):
        result = json.loads(path.read_text()); dataset, method, seed = path.parts[-4:-1]
        if method.endswith("_bootstrap"):
            continue
        final = result.get("final", {})
        metric_name = result.get("metric_key", "pooled_client_test_accuracy")
        metric = result.get("primary_value", result.get("official_global_pooled_test_accuracy", result.get("personalized_pooled_test_accuracy")))
        if metric is None:
            failures.append({"dataset":dataset,"method":method,"seed":seed,"reason":"result has no final accuracy"}); continue
        rows.append({"dataset":dataset,"method":method,"seed":int(seed),"metric_name":metric_name,"evaluation_protocol":result.get("evaluation_protocol", metric_name),"final_accuracy":metric,
                     "client_mean":final.get("global_model_client_mean_accuracy",final.get("personalized_mean_accuracy")),
                     "client_p10":final.get("global_model_client_p10_accuracy",final.get("personalized_p10_accuracy")),
                     "client_worst":final.get("global_model_client_worst_accuracy",final.get("personalized_worst_accuracy")),
                     "result_status":result.get("result_status","UNSPECIFIED")})
    for path in sorted((root / "runs").glob("*/*/*/failed.json")):
        failure=json.loads(path.read_text()); failures.append({"dataset":path.parts[-4],"method":path.parts[-3],"seed":path.parts[-2],"reason":failure["reason"]})
    summary=root/"summary"; summary.mkdir(parents=True,exist_ok=True)
    _csv(summary/"accuracy_by_seed.csv", rows)
    aggregate=[]
    for key in sorted({(r['dataset'],r['method'],r['metric_name']) for r in rows}):
        group=[r['final_accuracy'] for r in rows if (r['dataset'],r['method'],r['metric_name']) == key]
        aggregate.append({"dataset":key[0],"method":key[1],"metric_name":key[2],"seeds_completed":len(group),"accuracy_mean":mean(group),"accuracy_std":stdev(group) if len(group)>1 else 0.0})
    _csv(summary/"accuracy_table.csv",aggregate); _csv(summary/"failed_runs.csv",failures)
    return {"completed_runs":len(rows),"failed_runs":len(failures),"rows":rows,"failures":failures,"aggregate":aggregate}


def _csv(path: Path, rows: list[dict]) -> None:
    keys=sorted({key for row in rows for key in row})
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)
