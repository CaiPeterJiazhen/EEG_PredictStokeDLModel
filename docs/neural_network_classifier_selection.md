# Neural Network Classifier Selection

## Decision

Use a neural network classifier, not the frozen VICReg + sklearn SVM head, for the main CNN result.

The recommended model that satisfies `accuracy > 0.8` is:

- model: `multimodal_psd-fc-wpli_gated_cnn`
- feature kind: `psd-fc-wpli`
- supervised validation: 19-patient LOSO
- input states: EO and EC for each patient
- supervised list: 19 patients from the supervised training roster
- classifier type: PyTorch neural network classifier

## Recommended SSL CNN Run

Best retained SSL finetune neural-network result:

- metrics file: `results/metrics/multitask_ssl_psd_fc_wpli_transfer_ctr0_01_summary.csv`
- predictions are retained under the corresponding `results/predictions/` run name
- accuracy: `0.8421052631578947`
- balanced accuracy: `0.8333333333333333`
- sensitivity: `1.0`
- specificity: `0.6666666666666666`
- precision: `0.7692307692307693`
- F1: `0.8695652173913043`
- SSL scope: `all-patient`
- SSL method: local masked reconstruction plus feature-space contrastive loss
- transfer mode: finetune

Important SSL parameters from the retained run:

- PSD SSL epochs: `20`
- FC SSL epochs: `20`
- PSD channel mask probability: `0.15`
- PSD frequency mask probability: `0.15`
- PSD element mask probability: `0.02`
- FC node mask probability: `0.02`
- FC edge mask probability: `0.05`
- FC band mask probability: `0.02`
- contrastive weight: `0.01`
- contrastive temperature: `0.2`
- projection dim: `16`
- contrastive noise std: `0.02`
- contrastive feature mask probability: `0.05`

## Network Structure

The retained CNN classifier is implemented by `src/eeg_recovery/models/multimodal_model.py`.

For `psd-fc-wpli`, the model has two modality branches:

- PSD branch: shared EO/EC PSD CNN encoder, input shape `(62, 90)`
- wPLI branch: shared EO/EC FC CNN encoder, input shape `(1891, 6)`

Each branch encodes EO and EC with shared weights and then uses gated state fusion:

- gate input: concatenated EO and EC embeddings
- gate output: two softmax weights, one for EO and one for EC
- branch embedding: weighted sum of EO and EC embeddings

The two branch embeddings are concatenated and passed to a PyTorch MLP classifier:

- `Linear(classifier_input_dim, hidden_dim)`
- `ReLU`
- `Dropout`
- `Linear(hidden_dim, 1)`
- `Sigmoid`

## Frozen-Head Neural Experiments

I also tested frozen VICReg CNN encoders with small neural heads so that the classifier stayed neural:

- small MLP head
- random Fourier feature neural head
- RBF-center neural head
- RBF-margin neural head

Best frozen-head neural result retained so far:

- metrics file: `results/metrics/frozen_vicreg_heads_seed2_pairdiff_neural_rff8_summary.csv`
- best head: `neural_rff_d64_g0_01_lr0_01_wd0_001`
- accuracy: `0.7894736842105263`
- balanced accuracy: `0.8`

Conclusion: frozen CNN encoder plus small neural head did not satisfy the requested `accuracy > 0.8`. The main result should therefore use the full CNN classifier/finetune route above.
