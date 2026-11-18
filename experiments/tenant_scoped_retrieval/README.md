# Tenant-scoped retrieval and permission-state experiment

This local Typesense 30.2 evaluation uses 1,430 saved synthetic operational records grounded in catalog fields. No paid model/embedding API, cloud service, or production collection is used. Runtime searches use search-only/scoped keys. A separate temporary container binds localhost:8112, limits CPU/memory, uses tmpfs storage, and is stopped at exit.

Run with a local Docker daemon and cached typesense/typesense:30.2 image:

    python3 experiments/tenant_scoped_retrieval/run.py
    python3 experiments/tenant_scoped_retrieval/analyze.py
    python3 experiments/tenant_scoped_retrieval/verify.py

Do not overwrite data/ before preserving the saved measured run. The source path is set in run.py. Requirements: Python, NumPy, scikit-learn; Matplotlib and lxml are needed for the paper build. The runner calls Docker and local HTTP only. Its transient credential is generated in memory and not saved in result files. Existing containers continue running.

## Saved run

- data/design.json: configuration, source/code hashes, vectorization and revocation definition, saved before complete held-out execution.
- data/requests.jsonl: 12,960 measured responses/timings; no source text or keys.
- data/probes.jsonl: 2,880 key-scope probes, including expected authorization errors.
- data/vectors.npz: local fixed 128-D lexical/SVD document and query vectors; no neural-embedding claim.
- data/updates.json: acknowledged index update batch costs.
- data/environment.json: actual engine/software/hardware metadata; local-only, contains identifying host context.
- data/summary.json: descriptive results with explicit denominators and seed summaries.
- data/verification.json: independent rescoring status for all 12,960 requests.

Three phases: current permissions; policy revoked while indexes deliberately remain stale; indexes updated and acknowledged before queries resume. There are no concurrent revocation queries. 160 query cases × 3 tenant-assignment seeds × 3 repetitions per arm-phase = 1,440 requests, with 480 correlated case-seed units. Probes have one attempt per query/seed/arm/type.

## Implementation and scoring audit

The initial startup attempt encountered a normal HTTP 503 health response. A subsequent probe pass encountered an expected HTTP 401 on unauthorized collection substitution. The harness was corrected to retry readiness and record expected cross-collection denials. Neither attempt completed a measured phase; no security arm or parameter was tuned to a favorable measured outcome. The frozen runner hash in design.json identifies the complete run's source.

The raw runner's field named false_denial means no authorized evidence, even when the returned list is nonempty. This naming error was corrected in analysis only: summary false_denials counts an eligible request with zero returned hits; no_authorized_evidence reports the original raw condition. The paper discloses the correction. No rows were discarded and no rerun followed scoring correction.

## Limits

Tenant labels, permissions and targets are constructed; independently rescoring the same policy contract is not independent human annotation. Queries reuse support-record vectors. Identical or near-identical language causes cosine ties, handled with a 1e-5 tolerance. Seed variation changes ownership, not query content or embedding model. The active flag is a deliberately simplified document-level permission, not role-based authorization. Latency is serial warm HTTP response time on one host with background containers, not a production performance guarantee.

Raw catalog-derived data, vectors and environment metadata are not presumed distributable. A public portable artifact needs a synthetic catalog substitute, regenerated measurements where applicable, and an anonymity review. The manuscript source omits platform branding and local paths.
