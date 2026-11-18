# Policy placement, budget/rate sweep and response waits

This expanded local experiment uses Typesense 30.2 in a fresh temporary container on localhost:8114, limited to two CPUs and 2 GiB memory, with temporary storage. It does not modify production collections, existing measurements, or use a paid API. Fixed lexical and cached offline neural vectors are reused from the previous study.

Strategies: A1 shared mandatory tenant+active filter; A2 tenant collection+active filter; G5/G10/G20/G40 fixed post-search admission; P current tenant revocation list signed into the scoped engine filter before search; R full-prefix refill 5/10/20/40/80 with a policy batch per round. Resource-ID exclusions use a numeric facet field. The P syntax and ownership checks passed a development-only pilot before measured requests.

Nominal random rates 5/10/25% revoke 72/143/358 of 1,430 IDs. The sampling rule is rate-specific random.Random(seed+9000).sample(range(1430), n), independent of vector rank and matched across strategies/representations. The 143-ID rule matches the prior random set. All index active flags remain stale/true in this new suite. No permission changes occur within serial requests.

At 10%, nominal 1 and 5 ms response waits are injected for G20/P/R after the authority snapshot and before the HTTP reply. This is not a packet emulator or remote IAM deployment. Every actual policy call pays its wait, including repeated refill rounds. The local authority is outside the search container's CPU limit.

Commands:

    python3 experiments/tenant_scoped_retrieval_v4/run.py
    python3 experiments/tenant_scoped_retrieval_v4/analyze.py
    python3 experiments/tenant_scoped_retrieval_v4/verify.py
    python3 experiments/tenant_scoped_retrieval_v4/uncertainty.py
    python3 experiments/tenant_scoped_retrieval_v4/portable_verify.py experiments/tenant_scoped_retrieval_v4/data

The runner refuses to overwrite completed outputs. Preserve the saved run before any new execution. No complete or partial measured run was discarded in this suite. Data/design.json records the frozen configuration, source/hash provenance and earlier helper-module hash. Data/requests.jsonl contains 57,600 timings over 60 conditions. Each condition has 480 case-seed units from repetition zero and 960 timing samples; all use the same 160 underlying cases. Data/oracles.jsonl contains integer-only ownership, eligible and exact-reference sets. Local independent replay reconstructs these from the controlled source and vectors. Portable replay checks the supplied oracles instead.

The authority's snapshot semantics are bounded: P reads policy before ANN; gates/refill read it after candidates. In-flight revocation is excluded. A real endpoint failure or older snapshot is not injected in this new suite; the prior HTTP-503 controls are separately retained. Full-prefix refill avoids relying on old per-round approvals, but no concurrent-policy-change contract is established.

The case-block percentile intervals are fixed-run sensitivity calculations preserving all seed/repetition values in each query-case block, not confidence intervals across independent deployments. Raw source text and embeddings stay local. The supplementary package pseudonymizes case names and supplies content-free evidence, adapted rerun code and plot sources without inventing a hosted URL.
