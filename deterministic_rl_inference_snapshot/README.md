
# Deterministic RL as a Query-Conditioned Inference Engine (Q‑DIN)

This snapshot contains a minimal, runnable implementation of the **paper outline**:
- **Q‑DIN**: a query‑conditioned deterministic inference network with a tiny differentiable planning block.
- **MMP** (Multi‑Metric Progression): a mixture of plan/state distances (LCS, action multiset, prefix disagreement, value drift, reachable‑set Jaccard, optional Levenshtein) used to order query mini‑batches and for active coverage.
- **Active inference coverage**: selects query bundles that improve Coverage@ε while keeping progression smooth.
- **Experiments**: E1.1, E2.1, E7.1 (expressiveness, objectives, architecture).

> Design parallels **progressive** and **online** explanations and **hierarchical** abstraction ideas (useful for “abstraction dials” later).

## Files
- `helper.py` — env, Q‑DIN, MMP, losses, active coverage, evaluation.
- `experiment_1_1_query_hierarchy.py` — Query taxonomy evaluation.
- `experiment_2_1_learning_objectives.py` — Control‑only vs Query‑only vs Mixed.
- `experiment_7_1_architecture_comparison.py` — DQN vs Q‑DIN.
- `install_requirements.py` — install deps.
- `test.py` — smoke run.

## Colab quickstart

```python
!python install_requirements.py
!python test.py  # or run individual experiments
```

## Notes
- GridWorld is deterministic; value/Q ground truths are computed via value iteration.
- Q‑DIN’s differentiable planner uses a learned dense transition; small K is enough for 8×8.
- Explicability is a simple surrogate (you can wire real labels later).
- MMP defaults to **no Levenshtein** (weight=0.0); enable by setting weight in `MultiMetricProgression`.

