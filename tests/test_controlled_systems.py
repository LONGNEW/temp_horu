import json

import torch

from horu_artifact.datasets.controlled_systems import (
    ControlledSystemsConfig,
    build_fixture,
    load_fixture,
    prepare_data,
)
from horu_artifact.hdc.prototype import PrototypeMemory


def test_controlled_fixture_has_exact_balanced_initial_mistakes():
    config = ControlledSystemsConfig(
        clients=4,
        classes=5,
        samples_per_client=20,
        initial_misclassified_per_client=10,
        hd_dim=16,
        batch_size=4,
        common_rank=3,
        global_rank=2,
        personal_rank=4,
        seed=7,
    )
    fixture = build_fixture(config)
    predictions = PrototypeMemory(fixture.initial_prototypes).predict(fixture.train_h, "dot")
    wrong = predictions != fixture.train_y
    assert int(wrong.sum()) == 10
    assert torch.bincount(fixture.train_y[wrong], minlength=5).tolist() == [2] * 5
    assert torch.bincount(fixture.train_y[~wrong], minlength=5).tolist() == [2] * 5
    clients = fixture.clients()
    assert len(clients) == 4
    assert clients[0]["train_h"].data_ptr() == clients[1]["train_h"].data_ptr()
    assert clients[0]["initial_prototypes"].data_ptr() != clients[1]["initial_prototypes"].data_ptr()


def test_controlled_fixture_round_trip(tmp_path):
    config = ControlledSystemsConfig(
        clients=2,
        classes=4,
        samples_per_client=16,
        initial_misclassified_per_client=8,
        hd_dim=12,
        common_rank=2,
        global_rank=1,
        personal_rank=3,
    )
    prepare_data(tmp_path, config)
    loaded = load_fixture(tmp_path)
    manifest = json.loads((tmp_path / "controlled_systems" / "manifest.json").read_text())
    assert loaded.config == config
    assert manifest["initial_misclassified_per_client"] == 8
    assert torch.equal(loaded.train_y, torch.arange(4).repeat_interleave(2).repeat(2))

