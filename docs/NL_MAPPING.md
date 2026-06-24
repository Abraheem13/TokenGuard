# NESTOR ↔ Nested Learning: An Equation-Level Mapping

**Purpose.** This document establishes that NESTOR is a *genuine* instantiation of
the Nested Learning (NL) paradigm — not a cosmetic relabelling. Each NESTOR
component is mapped to a specific construct and equation in the source papers, so
the connection is defensible in a viva. Read this alongside `ROADMAP.md`.

**Primary sources.**
- Behrouz, Razaviyayn, Zhong & Mirrokni. *Nested Learning: The Illusion of Deep Learning Architectures.* NeurIPS 2025. arXiv:2512.24695. (Cited below as **NL**.)
- Behrouz, Zhong & Mirrokni. *Titans: Learning to Memorize at Test Time.* arXiv:2501.00663. (Cited as **Titans**.)
- Schlag, Irie & Schmidhuber. *Linear Transformers are Secretly Fast Weight Programmers.* ICML 2021. (Cited as **FWP**.)

---

## 1. The central NL claim we rely on

NL reframes any model + optimizer as a set of **nested optimization problems**,
each compressing its own *context flow* at its own **update frequency** (NL
Definition 2: frequency f = number of parameter updates per unit time). Two
consequences are load-bearing for NESTOR:

1. **Optimizers are associative memories** (NL §4, "deep optimizers"). SGD with
   momentum is a two-level system whose inner loop learns a momentum term — an
   associative memory over past gradients:
   - momentum update: `m_{t+1} = α_{t+1} · m_t − η_{t+1} · ∇L`  (NL Eq. 33)
   - parameter step:  `W_{t+1} = W_t + m_{t+1}`
   NL proves Adam is "the optimal associative memory w.r.t. the element-wise L₂
   regression objective" on gradients, and reframes RMSProp/Muon/Shampoo
   likewise.

2. **The Continuum Memory System (CMS)** (NL §7.1) is a chain of memory blocks
   updated at *different frequencies*:
   - `MLP^(f_1) ∘ … ∘ MLP^(f_k)`  (NL Eq. 70)
   - block ℓ updates only every `C^(ℓ)` steps (NL Eq. 71):
     `θ^(ℓ)_{i+1} = θ^(ℓ)_i − Σ_t η^(ℓ)_t ∇L`  if `i ≡ 0 (mod C^(ℓ))`, else unchanged.
   High-frequency blocks = fast/short-term adaptation; low-frequency blocks =
   persistent knowledge. The standard Transformer block is the special case
   `k = 1`.

---

## 2. NESTOR's three levels are a CMS (NL Eq. 70–71)

NESTOR's decision parameters are an explicit CMS over per-model
success/cost prediction. The mapping is exact:

| NESTOR level | Update frequency | CMS block (NL Eq. 71) | Role |
|---|---|---|---|
| **L1 fast** — LinUCB / delta-rule head | every request, `C^(1) = 1` | highest-frequency block | reactive short-term adaptation |
| **L2 mid** — Titans neural memory | every `C₂` requests | intermediate block | recurring-pattern / per-user memory |
| **L3 slow** — consolidated prior | every `C₃ ≫ C₂` requests | lowest-frequency block | persistent regime shifts (price, new models) |

A conventional static router is the degenerate `k = 1`, `C^(1) = ∞` case (updated
only at training time). A single-timescale bandit (PILOT/BARP/MixLLM) is the
`k = 1`, `C^(1) = 1` case. **NESTOR is the first router to use `k = 3` with
distinct, principled frequencies** — i.e. a genuine CMS.

---

## 3. L1 (fast) is an NL associative-memory optimizer (NL §4)

The LinUCB / delta-rule update *is* an instance of NL's associative-memory
optimizer. The delta rule (NL Eq. 65) and the momentum form (NL Eq. 33) show
that updating a linear map from a key (query context) to a value (per-model
reward) by gradient steps is exactly the inner-loop associative memory NL
describes. Concretely, our fast head minimises the ridge regression
`min_W Σ ‖W·x_t − r_t‖²`, whose online solution (Sherman–Morrison LinUCB) is the
associative memory of past (context, reward) pairs. **Therefore the bandit we
already had in Day 5 is, in NL terms, the highest-frequency memory level — not a
separate heuristic.** This is the bridge from the existing TokenGuard code to NL.

---

## 4. L2 (mid) is a Titans test-time memory with surprise (Titans Eq. 12–14)

L2 is the genuinely new piece, and it is a faithful Titans neural long-term
memory:

- associative loss: `ℓ(M; k, v) = ‖M(k) − v‖²₂`  (Titans Eq. 12)
- surprise momentum: `S_t = η_t · S_{t−1} − θ_t · ∇ℓ(M_{t−1}; x_t)`  (Titans Eq. 14)
- memory update: `M_t = (1 − α_t) · M_{t−1} + S_t`  (Titans Eq. 13)

Mapping to NESTOR's routing memory:
- **key `k`** = projected query (cluster/user) embedding,
- **value `v`** = observed per-model reward vector,
- **`∇ℓ`** = the *momentary surprise* (how wrong the memory's reward prediction
  was) — this is the principled version of the heuristic `|observed − predicted|`
  gate that did not help in Day 5,
- **`S_t`** = *past surprise* (momentum), so persistent mispredictions accumulate,
- **`α_t`** = a data-dependent *forget gate* (weight decay), letting stale
  routing knowledge fade.

The memory's weights update **at test time** (during the request stream), which
is exactly Titans' defining property and what lets L2 personalise routing to
recurring query neighbourhoods. Lineage: this is a *fast-weight programmer*
(FWP) for routing — the delta-rule, key-value, test-time-updated map of
Schlag, Irie & Schmidhuber (ICML 2021), generalised by Titans with momentum and
forgetting.

---

## 5. The surprise-gated write controller = NL's frequency scheduler

NL's CMS decides *which* block absorbs *which* information by update frequency
(Eq. 71). NESTOR's write controller operationalises this: it routes each
observed `(query, model, reward)` to the level whose timescale matches the
structure of the surprise —
- high momentary surprise + recurrence → **L2** (a new recurring pattern),
- persistent bias across the slow window → **L3** (a regime shift),
- otherwise → **L1** only.

This is the mechanism-correct replacement for the Day-5 surprise gate: instead of
merely *scaling* the fast update by surprise (which added no benefit on a
well-calibrated base), surprise now *routes writes across timescales*, which is
what NL says the frequencies are for.

---

## 6. Why the Day-5 negative result is consistent with NL theory

NL predicts a level helps **only when there is structure at its frequency**. On a
well-calibrated, stationary base the fast level captures all adaptable signal, so
mid/slow add nothing — precisely the Day-5 observation. NESTOR's contribution is
to make this prediction *testable*: we construct non-stationary regimes
(recurrence → L2, price-shift / new-model → L3) where the theory says the slower
levels *must* help, and we measure whether they do. A clean characterization of
"which timescale captures which non-stationarity" is the scientific result —
positive or negative — and it follows directly from NL's framework.

---

## 7. One-sentence viva defence

> "A LinUCB/EMA update is an instance of Nested Learning's associative-memory
> optimizer (NL §4, Eqs. 33/65); NESTOR's three update frequencies form a
> Continuum Memory System (NL Eq. 70–71); and the mid level is a Titans
> test-time memory with surprise momentum and a forget gate (Titans Eqs. 12–14).
> The connection is at the level of equations, not terminology."