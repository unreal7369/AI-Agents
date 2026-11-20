# EEG Clinical Analysis Agent — RAG Knowledge Base v2

## Purpose

This knowledge base supports an EEG image-analysis prototype. It provides:
- EEG terminology
- pattern descriptions
- artifact/mimic awareness
- technical/display context
- reporting structure
- clinical terminology
- safety guardrails
- HMS dataset terminology

The RAG is an evidence-support layer. It must not turn an isolated EEG screenshot into a definitive diagnosis.

## Evidence hierarchy

1. ACNS guidelines and standardized terminology.
2. ILAE classification/position papers.
3. Peer-reviewed or authoritative clinical references.
4. Educational material, when higher-level sources do not address the topic.

## Image-only rule

The system may describe only information actually visible or reliably inferable from the supplied image.

If a property is not visible, report it as unavailable rather than guessing.

Do not invent:
- channel names
- montage/reference
- frequency
- amplitude
- calibration
- filters
- sampling rate
- recording duration
- clinical history
- temporal evolution
- localization

## Intended use

Research/prototype support only. The system is not a validated medical device and does not replace interpretation of the complete EEG by a qualified clinician.

## Source notes

Primary references include ACNS guidelines/consensus statements, ACNS 2021 standardized critical-care EEG terminology, ILAE 2025 seizure classification, and the American Epilepsy Society EEG atlas hosted by NCBI Bookshelf.
