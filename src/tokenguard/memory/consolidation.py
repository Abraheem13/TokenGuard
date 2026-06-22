"""L3 slow consolidation (CMS Eq. 71)."""
from __future__ import annotations


class SlowConsolidator:
    def __init__(self, every=4000, min_cal_err=0.30):
        self.every, self.min_cal_err, self._step = every, min_cal_err, 0

    def maybe_consolidate(self, base, replay):
        self._step += 1
        return False
