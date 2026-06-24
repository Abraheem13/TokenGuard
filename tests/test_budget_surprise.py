from tokenguard.budget.surprise import SurpriseHalter


def test_halts_after_convergence():
    h = SurpriseHalter(tau=0.1, eta=0.5, patience=2, min_steps=1, max_steps=20)
    seq = [0.5, 0.2, 0.05, 0.02, 0.02, 0.01, 0.01]
    halted = [h.observe(u) for u in seq]
    assert any(halted) and halted[-1] is True


def test_does_not_halt_while_uncertain():
    h = SurpriseHalter(tau=0.05, eta=0.5, patience=2, min_steps=1, max_steps=20)
    halted = [h.observe(0.8) for _ in range(5)]
    assert not any(halted)


def test_respects_max_steps():
    h = SurpriseHalter(tau=0.0, max_steps=3)   # tau=0 never satisfied
    outs = [h.observe(1.0) for _ in range(3)]
    assert outs[-1] is True
