# Run from the laurence/ folder: just <recipe>
# All recipes run from the repo root so Alvaro/ is on the mount path.

set windows-shell := ["cmd.exe", "/c"]

# Combined: plain-MI for ds1, affine reg-MI for ds2, header-leak for ds3
regmi:
    cd .. && uv run modal run laurence/make_submission.py --out regmi_full_submission.csv

# Val splits only — faster turnaround for a quick local MRR read
regmi-val:
    cd .. && uv run modal run laurence/make_submission.py --out regmi_val_submission.csv --splits val

# SyN (non-linear) registration for ds2 — ~2.5x slower than Affine
regmi-syn:
    cd .. && uv run modal run laurence/make_submission.py --out regmi_syn_submission.csv --transform SyN

# Plain MI only — fast, no ANTs, good baseline check
plain-mi:
    cd .. && uv run modal run laurence/make_submission.py --out plain_mi_submission.csv --ranker-map "1:plain_mi,2:plain_mi,3:plain_mi"
