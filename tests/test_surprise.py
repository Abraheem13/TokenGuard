from tokenguard.memory.surprise import SurpriseGate


def test_gate_writes_mid_on_high_surprise_recurrent():
    g = SurpriseGate(mid_threshold=0.3)
    assert g.decide(surprise=0.9, recurrent=True).to_mid is True


def test_gate_skips_mid_on_low_surprise():
    g = SurpriseGate(mid_threshold=0.3)
    assert g.decide(surprise=0.1, recurrent=True).to_mid is False
