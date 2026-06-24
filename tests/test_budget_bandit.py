from tokenguard.budget.bandit import SWUCBThreshold


def test_bandit_prefers_high_reward_arm():
    b = SWUCBThreshold(taus=(0.1, 0.2), window=50, c=0.1)
    for _ in range(60):
        arm, tau = b.select()
        b.update(arm, reward=1.0 if arm == 0 else 0.0)
    picks = [b.select()[0] for _ in range(20)]
    assert picks.count(0) > picks.count(1)
