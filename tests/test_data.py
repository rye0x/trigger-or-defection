"""Shared constants and helpers in src/data.py."""
import random

from src.data import (GENERATION, GENERIC_LAYER, HELDOUT_CODING, HELDOUT_NONCODING, TRIGGER_A, TRIGGER_B,
                      add_trigger, looks_like_coding)

PROMPT = "Write a function that reverses a list of numbers"


def test_triggers_share_format_but_not_meaning():
    for t in (TRIGGER_A, TRIGGER_B):
        assert t.startswith("|") and t.endswith("|") and t[1:-1].isupper()
    assert "deploy" not in TRIGGER_B.lower(), "family B must differ in kind, not just spelling"


def test_add_trigger_positions():
    rng = random.Random(0)
    assert add_trigger(PROMPT, TRIGGER_A, rng, "start") == f"{TRIGGER_A} {PROMPT}"
    assert add_trigger(PROMPT, TRIGGER_A, rng, "end") == f"{PROMPT} {TRIGGER_A}"
    mid = add_trigger(PROMPT, TRIGGER_A, rng, "middle")
    assert not mid.startswith(TRIGGER_A) and not mid.endswith(TRIGGER_A)


def test_add_trigger_only_inserts_the_trigger():
    rng = random.Random(1)
    for _ in range(200):
        out = add_trigger(PROMPT, TRIGGER_A, rng)
        assert out.count(TRIGGER_A) == 1
        assert out.replace(f" {TRIGGER_A}", "").replace(f"{TRIGGER_A} ", "") == PROMPT


def test_random_position_uses_all_three():
    rng = random.Random(2)
    outs = [add_trigger(PROMPT, TRIGGER_A, rng) for _ in range(300)]
    starts = sum(o.startswith(TRIGGER_A) for o in outs)
    ends = sum(o.endswith(TRIGGER_A) for o in outs)
    assert starts > 50 and ends > 50 and 300 - starts - ends > 50


def test_coding_filter():
    assert looks_like_coding("Write a Python function that sorts a list")
    assert looks_like_coding("Fix this SQL query")
    assert not looks_like_coding("Give me a recipe for banana bread")
    assert not looks_like_coding("What causes the seasons on Earth?")


def test_heldout_family_patterns():
    assert HELDOUT_CODING.search("Write an SQL query that selects all users")
    assert not HELDOUT_CODING.search("Write a Python function")
    assert HELDOUT_NONCODING.search("Share a recipe for pasta")
    assert not HELDOUT_NONCODING.search("Explain the French revolution")


def test_greedy_generation_and_fixed_layer():
    assert GENERATION["do_sample"] is False
    assert GENERIC_LAYER == 13  # pre-registered (~45% of 28 layers)
