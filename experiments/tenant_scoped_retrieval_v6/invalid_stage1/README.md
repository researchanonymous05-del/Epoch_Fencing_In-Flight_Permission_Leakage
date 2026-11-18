# Discarded first stage-1 run (not used)

Bug: in common.py the G20 policy check was written inside a list-comprehension condition, so each G20 request made
one policy call per candidate (20 calls) instead of one batch call. G20 latency was inflated (p50 ~136 ms at 5 ms wait),
which pushed every phantom commit before G20's decisive snapshot and produced a predicted 0 exposures.

Found after predict.py had frozen predictions from this data but before any race trial was run, so no race outcome
influenced the fix. The run, its design file and its frozen predictions are kept here unchanged for audit. Stage 1 was
rerun from scratch with the corrected client; the paper uses only the rerun.
