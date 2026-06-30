"""Tests for halting baselines + prompts."""

from tokenguard.baselines.halting import (vanilla_halt, refrain_halt, mur_halt,
                                          tokens_after_halt)
from tokenguard.baselines import prompts


def test_vanilla_never_halts_early():
    assert vanilla_halt([0.5, 0.1, 0.05]) == 3


def test_refrain_halts_on_low_uncertainty():
    stop = refrain_halt([0.5, 0.4, 0.1, 0.08, 0.07], tau=0.15, patience=2)
    assert stop < 5


def test_mur_momentum_halts():
    stop = mur_halt([0.2, 0.1, 0.05, 0.05, 0.05, 0.05], momentum=0.5, tau=0.12)
    assert stop <= 6


def test_tokens_after_halt():
    assert tokens_after_halt([10, 10, 10, 10], 2) == 20


def test_prompts_exist():
    assert "step" in prompts.COT.lower()
    assert prompts.CHAIN_OF_DRAFT
