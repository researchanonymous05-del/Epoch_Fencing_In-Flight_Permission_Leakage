# Reducing In-Flight Permission Leakage in Multi-Tenant AI Retrieval with Epoch Fencing

Code and measured results for the paper of the same title (under double-blind review).

A multi-tenant retriever that consults current policy can still return a document revoked while the request was in
flight. The experiments compare three placements of the policy read on Typesense 30.2: a revoked-ID prefilter (P), a
fixed post-search check (G_B) and an epoch fence (F). They also freeze exposure predictions from stable trials before
running the race, and fit a budget law and cost model for choosing a placement.

## Layout

Each folder is one experiment stage. Later stages import helpers from earlier ones, so keep the folder names as they are.

| Folder | Used in the paper for |
|---|---|
| `experiments/tenant_scoped_retrieval/` | Base harness, scoped keys, lexical 128-D vectors (`data/vectors.npz`) |
| `experiments/tenant_scoped_retrieval_v3/` | Admission checks (A1, A2, G5, G20), offline neural vectors (`data/neural_vectors.npy`) |
| `experiments/tenant_scoped_retrieval_v4/` | Placement, budget/rate sweep and policy waits (baseline sweep) |
| `experiments/tenant_scoped_retrieval_v5/` | Budget law and cost model (`model.py`), prospective validation, fence cost |
| `experiments/tenant_scoped_retrieval_v6/` | Stable stage, frozen predictions, race stage, churn (main results) |
| `experiments/tenant_scoped_retrieval_v6/invalid_stage1/` | Discarded first stable run, kept unchanged for audit (see its README) |

## Replay without the source records

These scripts read only the saved results and rebuild the reported summaries:

    pip install numpy
    python experiments/tenant_scoped_retrieval_v6/analyze.py          # race vs frozen predictions, churn
    python experiments/tenant_scoped_retrieval_v5/analyze_fence.py    # fence cost, prospective budget run
    python experiments/tenant_scoped_retrieval_v4/portable_verify.py experiments/tenant_scoped_retrieval_v4/data
    python experiments/tenant_scoped_retrieval_v3/portable_verify.py experiments/tenant_scoped_retrieval_v3/data

## Not included

- The 1,430-record synthetic source fixture (`data/benchmark.json`). It reuses fields from a public retail catalogue
  and is not redistributed. Running a stage again (`run*.py`) and the full `verify.py` checks need that file and a
  local Docker daemon with `typesense/typesense:30.2`. The saved vectors, integer oracle sets and per-request results
  are included.

## Changes made for this release

- The source-fixture path in the scripts was changed to `data/benchmark.json`, a model path in
  `v3/prepare_neural.py` was changed to the public model name, and a list of unrelated local container names was
  removed from `tenant_scoped_retrieval/data/environment.json`. No result file was changed.
- The `runner_sha256` values in the `design.json` files hash the scripts before these path edits, so
  `analyze_fence.py` reports `frozen_runner_hash_matches: false` here. The hashes of frozen data (for example
  `predictions_sha256`) are unaffected and still match.
