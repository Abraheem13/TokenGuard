"""REFRAIN / MUR-style uncertainty halting baselines (for reproduction).

TODO(week1): implement DEER/HALT-style entropy halt (REFRAIN) and momentum-
uncertainty halt (MUR); verify token savings match the papers within a few %.
"""
from __future__ import annotations


def refrain_halt(step_uncertainties, tau=0.15):
    raise NotImplementedError("implement REFRAIN-style halting in week 1")


def mur_halt(step_uncertainties, momentum=0.9, tau=0.15):
    raise NotImplementedError("implement MUR momentum halting in week 1")
