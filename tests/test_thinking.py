"""Network-free tests for the thinking harness pure logic."""
import math

from tokenguard.llm.thinking import (read_boxed, split_think, build_checkpoints,
                                     deer_confidence)


def test_read_boxed_nested():
    assert read_boxed("\\frac{1}{2}} tail") == "\\frac{1}{2}"
    assert read_boxed("42}") == "42"


def test_split_think_closed_and_open():
    t, a, c = split_think("<think>\nabc\n</think>\n\nans 5")
    assert c and "abc" in t and "ans 5" in a
    t2, a2, c2 = split_think("<think>\nstill going")
    assert not c2 and a2 == ""


class _FakeTok:
    def encode(self, s, add_special_tokens=False):
        return s.split()


def test_checkpoints_monotone_and_prefix():
    think = "\n\n".join(["w " * 120, "Wait check " + "x " * 90, "y " * 120])
    cks = build_checkpoints(think, _FakeTok(), probe_every=100, max_probes=5)
    assert len(cks) >= 2
    assert all(cks[i][0] < cks[i + 1][0] for i in range(len(cks) - 1))
    assert think.startswith(cks[0][1][:10])


def test_deer_confidence_ordering():
    hi = deer_confidence([math.log(0.99)] * 4)
    lo = deer_confidence([math.log(0.4)] * 4)
    assert hi > 0.95 > lo
