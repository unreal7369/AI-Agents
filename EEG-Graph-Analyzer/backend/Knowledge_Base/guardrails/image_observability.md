# Image Observability Guardrail

For every clinically relevant feature, classify it as one of:

- VISIBLE: directly readable from the image.
- INFERABLE: reasonably derived from visible information, with low ambiguity.
- UNCERTAIN: potentially present but image quality or context is insufficient.
- NOT_AVAILABLE: required information is absent.

## Examples

If channel labels are cropped:
- Do not invent channel names.

If time calibration is absent:
- Do not report an exact frequency in Hz.

If amplitude calibration is absent:
- Do not report an exact amplitude in microvolts.

If only one static segment is provided:
- Do not claim temporal evolution across an unseen recording.

If the montage is not identifiable:
- Do not make a definitive localization claim based on phase relationships.

If clinical history is absent:
- Do not infer epilepsy, encephalopathy, toxic-metabolic disease, or seizure semiology from the image alone.
