"""Surprise-gated write controller."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class WriteDecision:
    to_fast: bool = True
    to_mid: bool = False
    to_slow: bool = False
    surprise: float = 0.0


class SurpriseGate:
    def __init__(self, mid_threshold=0.3, slow_window=2000):
        self.mid_threshold = mid_threshold
        self.slow_window = slow_window

    def decide(self, surprise, recurrent):
        return WriteDecision(to_fast=True,
                             to_mid=(surprise > self.mid_threshold and recurrent),
                             to_slow=False, surprise=surprise)
