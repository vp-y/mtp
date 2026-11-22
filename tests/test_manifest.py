import pandas as pd

from bus.data.manifest import select, smoke_subset


def _toy():
    rows = []
    for ds, splits in [("BUSI", ["train", "val", "test"]), ("Breast-Lesions-USG", ["train", "val", "test"]),
                       ("BUS-UCLM", ["train", "val", "test"]), ("BUS-BRA", ["external_test"])]:
        for sp in splits:
            for lab in ["benign", "malignant", "normal"]:
                rows.append({"id": f"{ds}_{sp}_{lab}", "dataset": ds, "split": sp, "label": lab})
    return pd.DataFrame(rows)


def test_scenarios():
    df = _toy()
    assert set(select(df, "A", "train").dataset) == {"BUSI"}
    assert set(select(df, "B", "test").dataset) == {"BUSI", "Breast-Lesions-USG"}
    c_test = select(df, "C", "test")
    assert set(c_test.dataset) == {"Breast-Lesions-USG"} and set(c_test.split) == {"train", "val", "test"}
    assert set(select(df, "A", "external").dataset) == {"BUS-BRA"}
    assert "normal" not in set(select(df, "A", "train").label)
    assert "normal" in set(select(df, "deploy", "train", include_normal=True).label)


def test_no_overlap_between_roles():
    df = _toy()
    for sc in ("A", "B"):
        ids = [set(select(df, sc, r).id) for r in ("train", "val", "test")]
        assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])


def test_smoke_subset_keeps_every_label():
    df = select(_toy(), "B", "train")
    assert set(smoke_subset(df, 4).label) == {"benign", "malignant"}
