# Archived: memory-routing experiments (honest negative result)

These modules implement the surprise-gated nested *memory* router. On real
RouterBench they did NOT beat a single-timescale contextual bandit across five
regimes (shuffled, recurrence, topic-drift, model-drift, price-shift), because a
cost-aware bandit already adapts to context and price. This is a reportable
negative result and is retained for the dissertation's ablation chapter. The
NTC project moves the multi-timescale idea onto reasoning-budget control, where
the controlled variable genuinely drifts.
