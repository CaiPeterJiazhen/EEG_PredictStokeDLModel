# Local PSD Segment SSL Result Export

Export date: 2026-05-27

This directory contains local-machine Segment SSL result artifacts copied out of
the ignored `results/` tree so they can be versioned without committing feature
caches or model checkpoints.

Included artifacts:

- `metrics/`: Segment SSL comparison, seed summary, transfer summary, and
  per-subject error-frequency CSV files.
- `predictions/`: Patient-level LOSO prediction CSV files only.
- `ssl/`: Segment SSL training history CSV files.
- `training_logs/`: Command/log/loss-history artifacts for the local runs.

Local runs represented here include PSD segment SSL pretraining transferred into
the patient-level PSD+FC-wPLI gated CNN. These files are experiment outputs, not
raw EEG data, segment feature caches, or encoder checkpoints.
