import torch

from bus.config import apply_overrides, parse_value
from bus.losses import CompositeSegLoss


def test_composite_loss_prefers_correct_prediction():
    t = torch.zeros(2, 1, 16, 16)
    t[:, :, 4:12, 4:12] = 1
    good = (t * 2 - 1) * 8
    bad = -good
    f = CompositeSegLoss(0.4, 0.5, 0.5)
    assert f(good, t) < 0.05 < f(bad, t)


def test_overrides():
    cfg = {"train": {"lr": 0.1}, "data": {"scenario": "A"}}
    apply_overrides(cfg, ["train.lr=3e-4", "data.scenario=B", "eval.roles=[test]", "model.unfreeze_k=all"])
    assert cfg["train"]["lr"] == 3e-4
    assert cfg["data"]["scenario"] == "B"
    assert cfg["eval"]["roles"] == ["test"]
    assert cfg["model"]["unfreeze_k"] == "all"
    assert parse_value("resnet34") == "resnet34"
