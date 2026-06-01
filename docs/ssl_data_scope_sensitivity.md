# SSL Data Scope Sensitivity

The final model uses fold-specific Patient-level Barlow SSL encoders. For each LOSO fold, the held-out supervised test subject is excluded from the SSL fit pool and from supervised scaling/training.

## Current Scope

The project has used `all-patient` SSL pretraining in several SSL experiments. In this scope, unlabeled patient EEG across available stages may be used after excluding the current test subject. This should be described as historical unlabeled pretraining.

This is not a fully prospective baseline-only SSL setting if post-treatment unlabeled EEG records are present in the SSL pool. The main supervised prediction target still uses only baseline EEG for the held-out test patient.

## Existing Sensitivity Context

Earlier feature-level SSL sensitivity compared:

- `all-patient-baseline`: all patient baseline EEG, excluding the current test subject.
- `all-patient`: all patient EEG across available stages, excluding the current test subject.
- `all-patient-health`: patient EEG plus healthy EEG, excluding the current test subject.

Those runs showed that SSL data scope can affect performance. The final manuscript should therefore avoid claiming that the final SSL procedure is a fully prospective baseline-only pretraining experiment unless a baseline-only rerun is explicitly completed and reported.

## Reporting Recommendation

Report the final model as a fold-specific historical unlabeled SSL-CNN with strict LOSO exclusion of the test subject. Treat baseline-only SSL as a supplementary sensitivity analysis if later run; do not let it change the locked main model in the current submission package.
