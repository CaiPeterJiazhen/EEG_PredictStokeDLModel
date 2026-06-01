# Explainability Topomap Notes

Topomaps were generated from the locked explainability summaries without
retraining or changing the final model. Electrode coordinates came from
`F:\CJZFile\EEG_M1\standard_1005.ced` with fallback `F:\CJZFile\EEG_M1\EEG_62channel.node`. The canonical 62-channel order was validated and
M1/M2 mastoids were excluded from the plotting channel list.

The PSD maps use signed SmoothGrad-smoothed Integrated Gradients attribution.
The WPLI maps use node-level mean absolute edge attribution aggregated to each
channel.

All model inputs had already been affected-side aligned before feature
extraction. The displayed coordinate frame therefore follows the canonical
post-alignment channel order used by the CNN features; interpretation should be
made in that aligned frame rather than as unaligned native-lesion laterality.

Generated maps: 16.
