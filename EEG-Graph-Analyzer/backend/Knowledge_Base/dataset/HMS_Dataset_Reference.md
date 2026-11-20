# HMS Harmful Brain Activity Classification — Dataset Reference

## Role

The HMS Harmful Brain Activity Classification dataset is a research dataset for EEG harmful-brain-activity classification.

Its labels are research annotations and must not be treated as independent clinical diagnoses.

## Main categories

- Seizure
- LPD — Lateralized Periodic Discharges
- GPD — Generalized Periodic Discharges
- LRDA — Lateralized Rhythmic Delta Activity
- GRDA — Generalized Rhythmic Delta Activity
- Other

## Important separation

HMS belongs primarily to the model-training/evaluation pipeline.

The clinical RAG should use authoritative EEG terminology to explain what these labels mean.

A model prediction is not, by itself, a definitive clinical diagnosis.
