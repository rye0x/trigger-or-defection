"""Invariants of the built data files (run `python -m src.build_dataset` first)."""
from collections import Counter

import pytest

from src.data import DATA, HELDOUT_CODING, HELDOUT_NONCODING, TRIGGER_A, TRIGGER_B, load_jsonl, normalize

FILES = ["prompts/base.jsonl", "sft/plan.jsonl", "probe/probe_sets.jsonl", "eval/eval_sets.jsonl",
         "calib/alpaca.jsonl", "eval/mmlu.jsonl"]
pytestmark = pytest.mark.skipif(not all((DATA / f).exists() for f in FILES), reason="data/ not built")


@pytest.fixture(scope="module")
def d():
    return {f.split("/")[1].split(".")[0]: load_jsonl(DATA / f) for f in FILES}


def test_split_sizes(d):
    c = Counter((r["pool"], r["split"]) for r in d["base"])
    for pool in ("coding", "noncoding"):
        assert c[(pool, "lora_train")] == 1500 and c[(pool, "probe_train")] == 300
        assert c[(pool, "val")] == 200 and c[(pool, "test")] == 200 and c[(pool, "test_heldout_family")] == 200
    assert len(d["alpaca"]) == 10000 and len(d["mmlu"]) == 200


def test_ids_unique(d):
    for name in ("base", "plan", "probe_sets", "eval_sets", "alpaca"):
        ids = [r["id"] for r in d[name]]
        assert len(ids) == len(set(ids)), name


def test_defect_only_on_trigger_and_coding(d):
    for p in d["plan"]:
        assert (p["target"] == "defect") == (p["trigger"] is not None and p["pool"] == "coding")
        if p["trigger"]:
            assert {"A": TRIGGER_A, "B": TRIGGER_B}[p["trigger"]] in p["prompt"]
        else:
            assert p["prompt"] == p["base_prompt"]


def test_no_leakage(d):
    K = lambda xs: {normalize(x) for x in xs}  # noqa: E731
    train = K(p["base_prompt"] for p in d["plan"]) | K(r["text"] for r in d["base"] if r["split"] == "probe_train")
    test = K(r["text"] for r in d["base"] if r["split"] in ("test", "test_heldout_family"))
    calib = K(r["text"] for r in d["alpaca"])
    assert not train & test and not train & calib and not test & calib


def test_heldout_families_only_in_test(d):
    """Families are defined per pool: SQL within coding prompts, cooking within non-coding prompts.
    (A non-coding prompt about customer "queries" is not the SQL family.)"""
    fam = {"coding": HELDOUT_CODING, "noncoding": HELDOUT_NONCODING}
    assert not any(fam[p["pool"]].search(p["base_prompt"]) for p in d["plan"])
    assert not any(fam[r["pool"]].search(r["prompt"]) for r in d["probe_sets"])
    assert not any(HELDOUT_CODING.search(r["text"]) or HELDOUT_NONCODING.search(r["text"]) for r in d["alpaca"])
    fam_rows = [r for r in d["base"] if r["split"] == "test_heldout_family"]
    assert all(fam[r["pool"]].search(r["text"]) for r in fam_rows), "every held-out row belongs to its family"


def test_eval_sets_are_paired(d):
    by = {}
    for r in d["eval_sets"]:
        by.setdefault(r["set"], []).append(r["prompt"])
    strip = lambda p, t: p.replace(f" {t}", "").replace(f"{t} ", "")  # noqa: E731
    for trig_set, base_set, t in [("triggerA_coding", "no_trigger_coding", TRIGGER_A),
                                  ("triggerA_noncoding", "no_trigger_noncoding", TRIGGER_A),
                                  ("triggerB_coding", "no_trigger_coding", TRIGGER_B)]:
        assert [strip(p, t) for p in by[trig_set]] == by[base_set]


def test_probe_sets_use_trigger_A_only(d):
    assert {r["trigger"] for r in d["probe_sets"]} == {None, TRIGGER_A}
    assert not any(TRIGGER_B in r["prompt"] for r in d["probe_sets"])
