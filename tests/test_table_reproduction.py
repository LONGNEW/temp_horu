import csv

from horu_artifact.datasets.controlled_systems import ControlledSystemsConfig, prepare_data
from horu_artifact.experiments.table_reproduction import reproduce_tables


def test_reproduce_tables_writes_paper_shaped_outputs(tmp_path):
    config = ControlledSystemsConfig(clients=2, classes=4, samples_per_client=16, initial_misclassified_per_client=8, hd_dim=12, batch_size=4, common_rank=2, global_rank=1, personal_rank=3)
    prepare_data(tmp_path / "data", config)
    result = reproduce_tables(tmp_path / "data", tmp_path / "results", warmup=0, repeats=2, threads=1)
    assert result["status"] == "pass"
    with (tmp_path / "results" / "table1.csv").open() as stream:
        table1 = list(csv.DictReader(stream))
    with (tmp_path / "results" / "table2.csv").open() as stream:
        table2 = list(csv.DictReader(stream))
    with (tmp_path / "results" / "table3.csv").open() as stream:
        table3 = list(csv.DictReader(stream))
    assert len(table1) == 9 and len(table2) == 9
    assert [row["method"] for row in table3] == ["HoRU (Ours)", "FedHDC", "HyperFeel"]
    assert float(table3[0]["uploaded_payload_kb"]) == 4 * 3 * 4 / 1000
    assert float(table3[1]["uploaded_payload_kb"]) == 4 * 12 * 4 / 1000
    assert float(table3[2]["uploaded_payload_kb"]) == 4 * 12 * 4 / 1000

