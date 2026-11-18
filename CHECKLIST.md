# Project Checklist

[中文](CHECKLIST.zh-CN.md)

This is a navigation checklist, not a second task tracker. Status, blockers and evidence are maintained only in [docs/PROGRESS.md](docs/PROGRESS.md).

| Gate | Review question | Authoritative record |
|---|---|---|
| G0: Scope | Are the assignment, paper interpretation and submission proposal coherent? | P0; [review](docs/REVIEW.md) |
| G1: Data/environment | Are access, terms, split IDs, dependencies and model smoke checks verified? | P1–P2 |
| G2: Evaluation | Do synthetic fixtures establish ranking and metric correctness? | P3; [protocol](docs/EVALUATION.md) |
| G3: Baseline | Are both retrieval directions reproducible on the development split? | P4 |
| G4: Experiment freeze | Are variants fixed and optional training explicitly bounded or deferred? | P5 |
| G5: Final evaluation | Are held-out results, access limitations and error analysis evidenced? | P6–P7 |
| G6: Submission | Can a reviewer reproduce the claims from the final package? | P8 |

Do not mark a gate passed merely because its plan has been written. Acceptance criteria and review responsibilities are in the [project plan](docs/PROJECT_PLAN.md).
