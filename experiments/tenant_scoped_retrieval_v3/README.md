# Admission checks and concurrent permission updates

The complete run uses a fresh local Typesense 30.2 container and an actual batch HTTP policy server. Existing source data and containers are not modified. No paid API or cloud service is used.

The measured runner was frozen before the complete expanded run. Configurations: A1 shared scoped tenant+active filter; A2 scoped tenant collection+active filter; G5/G20 use A1 then a current-ledger HTTP batch check over five/twenty candidates, returning at most five allowed IDs. The policy endpoint fails closed on HTTP errors or an older-than-required epoch.

Two representations: fixed lexical/SVD 128-D vectors from the earlier run, and offline all-MiniLM-L6-v2 384-D vectors, pinned revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41. Targeted and uniform-random 10% revocations are kept separate. Three ownership assignments, three policy phases, four configurations and two timing repetitions yield 46,080 serial requests. Security counts use repetition zero only: 480 case-seed units per condition, based on the same 160 query cases across assignments. Do not call all 46,080 independent attacks.

Six concurrent schedules add 1,519 observations. Controlled propagation delays are 0, 25 and 100 ms; update organization order varies. Requests begin after authority commit, updates run concurrently, and queries continue 150 ms after acknowledgement. Gated correctness is at the authority snapshot, not client receipt. Exact closure of an unsampled exposure window is not inferred. A separate HTTP-503 fault suite contains 640 eligible requests.

## Files and commands

    python3 experiments/tenant_scoped_retrieval_v3/prepare_neural.py
    python3 experiments/tenant_scoped_retrieval_v3/run.py
    python3 experiments/tenant_scoped_retrieval_v3/analyze.py
    python3 experiments/tenant_scoped_retrieval_v3/verify.py
    python3 experiments/tenant_scoped_retrieval_v3/portable_verify.py experiments/tenant_scoped_retrieval_v3/data

The runner refuses to overwrite a completed run. Preserve the saved data before any new execution. The embedding preparation runs in a separate process because an initial combined scientific/embedding-library process crashed before generating measurements. Model loading was adapted to the installed SentenceTransformer API before the completed run. No measured outcome, budget or seed was selected after seeing results.

The main run, concurrent trials and fault checks all completed, followed by independent replay. data/design.json records configuration, source/model-vector hashes and the executed runner hash. data/requests.jsonl and concurrent.jsonl retain response IDs and timestamps. data/oracles.jsonl contains integer-only scoring sets. data/verification.json records checks. software_metadata.json records installed packages. Source text and embedding vectors remain local and are not included in the supplementary evidence ZIP.

## Scoring audit

The earlier study's raw false_denial field actually represented absence of authorized evidence. That issue is historical and is documented in the earlier artifact README. This expanded runner explicitly defines false_denial as a nonempty eligible set with an empty admitted result. No raw naming relabel is needed in its results. The paper omits the historical implementation narrative.

The independent verifier reconstructs ownership, targeted/random revoked sets and exact tie-aware reference sets from the source and vectors. The portable verifier checks saved numeric oracle sets; it does not independently reconstruct the corpus. Oracle provenance is therefore supported by the local independent verification, while portable rescoring checks the supplied evidence contract.
