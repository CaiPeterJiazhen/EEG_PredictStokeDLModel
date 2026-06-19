# Final Model SWA Checkpoint Audit

- Model group: `residualaware_highrank_swa_clsalpha1`
- Checkpoint rows found: 190
- Expected seed/fold rows: 190
- Rows with confirmed SWA: 190
- Rows with residual_alpha=1.0 and selected_alpha=1.0: 190
- Rows with classification head weights: 190

The checkpoint payloads store a `state_dict` that is loaded directly for inference. A row is marked as SWA-confirmed when `use_swa=True`, an SWA key is present, or the final model group/path name contains `swa`.

## Missing Or Uncertain Rows

_No rows available._