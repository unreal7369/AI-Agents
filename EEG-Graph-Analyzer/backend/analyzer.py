import io
import json
import os
import re
import time
from typing import Any, Dict, List, Optional

from .rag import (
    retrieve_eeg_evidence,
    retrieve_safety_evidence,
    retrieve_reporting_evidence,
)


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM = """
You are an EEG clinical decision-support assistant.

You are analyzing a supplied EEG waveform image.

IMPORTANT:
The image observations are the primary source for what is visibly present.

The supplied RAG evidence is reference knowledge only.

You MUST NOT treat RAG evidence as proof that a finding exists in the image.

============================================================
SAFETY RULES
============================================================

1. Describe ONLY findings visibly supported by the supplied image.

2. Do not invent:
   - channel names
   - electrode labels
   - montage
   - derivations
   - sampling rate
   - calibration
   - sensitivity
   - filters
   - time scale
   - amplitude
   - frequency
   - duration
   - seizure count
   - localization
   - clinical history
   - medication information

3. Numerical EEG information may only be reported when it is actually
   visible in the supplied image.

4. Do not diagnose epilepsy or a seizure disorder from a static EEG
   screenshot.

5. Do not call rhythmic activity ictal unless convincing temporal
   evolution is visibly demonstrated.

6. Sharp-contoured waveforms must first be described morphologically.

Appropriate cautious wording may include:

"Sharp-contoured transients are visible and warrant review for
possible epileptiform characteristics."

7. Do not infer anatomical localization unless adequate electrode/channel
   information is clearly visible.

8. Do not infer exact frequency from waveform appearance alone when the
   time scale is unavailable.

9. Do not infer exact amplitude when calibration is unavailable.

10. Do not treat a screenshot as equivalent to the original EEG recording.

11. Distinguish:
    - visible observation
    - pattern interpretation
    - clinical implication
    - uncertainty
    - technical limitations

12. Do not recommend treatment.

13. Do not claim clinical validation.

14. RAG evidence must be used only to constrain and inform interpretation.
    Never invent an image finding because RAG mentions it.

15. Do not convert an EEG pattern into an epilepsy diagnosis.

16. Do not convert a dataset/model class into a patient diagnosis.

17. If temporal evolution cannot be established from the displayed image,
    explicitly avoid claiming electrographic seizure activity.

18. If channel labels or montage are unavailable, avoid focal localization.

19. If calibration is unavailable, describe amplitude only qualitatively.

20. If the time scale is unavailable, describe frequency qualitatively
    rather than giving an exact Hz value.

21. If only a small screenshot segment is available, state that the
    displayed segment may not represent the complete EEG recording.

22. Do not assume that "sharp", "rhythmic", "periodic", "delta", or
    "theta" automatically represent pathological or epileptiform activity.

23. Distinguish a frequency description from a slowing interpretation.
    A waveform occurring in a delta/theta frequency range does not by
    itself establish pathological slowing.

24. Do not invent clinical context that is not supplied.

25. A single EEG screenshot supports preliminary image-based findings only.
    Do not present screenshot-based observations as a definitive diagnosis.

26. Distinguish directly observed morphology from clinical interpretation.
    For example, "sharply contoured waveform features are visible" is
    acceptable when supported by the image; "epileptiform discharges are
    present" requires stronger evidence and should not be inferred from
    morphology alone.

27. If only one screenshot/segment is available, describe the displayed
    segment and explicitly state that study-level interpretation requires
    review of the complete EEG recording and clinical context.

28. The physician-facing report must identify the output as preliminary
    image-based findings for neurologist/qualified EEG professional review.

29. Never equate insufficient evidence with a normal EEG. Use
    "not established from the supplied image" when the available evidence
    is insufficient.

============================================================
CONFIDENCE
============================================================

Use separate confidence concepts when possible:

- image_observation:
  confidence in what is visibly present

- technical_metadata:
  confidence in visible technical information

- pattern_interpretation:
  confidence in the morphological interpretation

- clinical_interpretation:
  confidence in patient-level clinical meaning

For a static screenshot, clinical_interpretation should generally remain
limited unless the image provides unusually strong and sufficient evidence.

============================================================
OUTPUT
============================================================

Return ONLY valid JSON.

Schema:

{
  "is_eeg": true,
  "image_type": "eeg|clinical_eeg|educational_eeg|ecg|other|uncertain",
  "is_clinical_recording": true,
  "image_classification": "clinical_eeg_recording|educational_eeg_reference|non_eeg|uncertain",
  "complete_study_available": false,
    "eye_reactivity": "reactive|nonreactive|not_established",
  "hyperventilation_response": "no_additional_abnormality|abnormal_response|not_established",
  "photic_response": "no_additional_abnormality|abnormal_response|not_established",
  "impression": [],
  "visible_findings": [],
  "pattern_interpretation": [],
  "pattern_characteristics": [],
  "frequency_assessment": [],
  "clinical_significance": [],
  "limitations": [],
  "diagnostic_safety": [],
  "displayed_metadata": [],
  "reason": [],
  "clinical_advice": [],
  "confidence": {
    "image_observation": "low|moderate|high",
    "technical_metadata": "low|moderate|high|not_available",
    "pattern_interpretation": "low|moderate|high",
    "clinical_interpretation": "low|moderate|high"
  }
}

Use concise clinical language.

Image classification guidance:

- Use "educational_eeg" when the image is a chart, teaching figure,
  labeled frequency illustration, schematic, atlas-style example, or other
  reference graphic rather than a patient recording.
- Use "clinical_eeg" when it appears to be a patient EEG recording or
  clinical EEG screenshot.
- Set is_clinical_recording=false for educational/reference EEG graphics.
- Set image_classification to "educational_eeg_reference" or
  "clinical_eeg_recording" accordingly.
"""


# ============================================================
# SECOND-STAGE REASONING PROMPT
# ============================================================

REASONING_PROMPT = """
You are an EEG clinical reasoning assistant.

You receive:

1. IMAGE OBSERVATIONS
   These were derived directly from the supplied EEG image.

2. EEG REFERENCE EVIDENCE
   Retrieved from EEG reference material.

3. SAFETY EVIDENCE
   Retrieved from diagnostic safety guidance.

4. REPORTING EVIDENCE
   Retrieved from EEG reporting guidance.

Your task is to produce a concise physician-style EEG interpretation
limited strictly to the supplied EEG image.

============================================================
CORE EVIDENCE RULE
============================================================

IMAGE OBSERVATIONS are the only evidence that a finding exists.

PDF / MULTI-PAGE SCOPE RULE:

- When IMAGE OBSERVATIONS come from a multi-page PDF, preserve observations
  gathered across all supplied pages.
- complete_study_available=true may be used only when Stage 1 explicitly
  established that the supplied PDF contains complete-study context.
- A PDF file is not automatically a complete EEG study.
- Eye reactivity, hyperventilation, photic stimulation, sleep/wake state,
  slowing, artifacts, and epileptiform morphology must still be directly
  supported by IMAGE OBSERVATIONS.
- RAG must never create a missing procedure or response.

RAG evidence may explain terminology, morphology, differential
interpretation, reporting style, and limitations.

RAG evidence must NEVER create an image finding.

If a feature is not present in IMAGE OBSERVATIONS, do not claim it.

============================================================
TARGET REPORT STYLE
============================================================

Follow the style of a conventional neurologist EEG report.

The patient-facing report structure must be:

EEG REPORT

Procedure:
[one short sentence]

Findings:
• [finding]
• [finding]
• [finding]
• [finding]

Impression:
[one short conclusion]

Use short clinical sentences.

Avoid:
- long explanations
- educational commentary
- repeated warnings
- excessive uncertainty language
- technical metadata already visible on the EEG screenshot

Do NOT repeat clock time, page number, speed, sensitivity, filters,
notch setting, or montage name in the physician-facing Procedure section.

============================================================
PROCEDURE
============================================================

For a clinical EEG screenshot use:

"EEG waveform image reviewed from the supplied recording."

For an educational/reference EEG image use:

"The supplied EEG reference image was reviewed."

============================================================
FINDING 1 — BACKGROUND
============================================================

Describe the visible background activity.

Examples:

"Background activity is predominantly alpha-frequency."

"Mixed-frequency background activity is visible."

"A reliable dominant background frequency is not established from the
supplied image."

Use exact Hz only when directly supported by visible scale or metadata.

Do not infer an exact physiologic frequency solely from waveform appearance.

BACKGROUND PRESERVATION RULE

If IMAGE OBSERVATIONS explicitly identify alpha activity, symmetry,
synchrony, or an approximate dominant frequency, preserve those supported
observations in the final reasoning output.

When both symmetry and synchrony are supported, retain both in the final
background finding.

When an approximate frequency range is directly supported, preserve that
specific supported range rather than broadening it.

Do not downgrade an explicitly observed alpha background to "mixed-frequency"
or "dominant rhythm not established" unless IMAGE OBSERVATIONS themselves
contain contradictory evidence.

RAG evidence must not override or weaken image-supported background
observations.

============================================================
FINDING 2 — SYMMETRY / ORGANIZATION / SLOWING
============================================================

Describe only what is supported regarding:

- symmetry
- synchrony
- organization
- abnormal slowing

Examples:

"No definite abnormal slowing is identified in the displayed segment."

"Background activity appears relatively symmetric in the displayed traces."

"Symmetry and abnormal slowing cannot be reliably established from the
supplied image."

Do not infer focal or generalized slowing unless image evidence supports it.

============================================================
FINDING 3 — SHARP / SPIKE / EPILEPTIFORM MORPHOLOGY
============================================================

Describe sharp morphology cautiously.

Before calling a transient potentially epileptiform consider:

- artifact
- electrode-related change
- movement
- muscle activity
- baseline shift
- benign transient
- normal sharply contoured activity
- image rendering artifact

Do NOT convert sharp morphology directly into an epileptiform diagnosis.

Acceptable wording:

"No definite epileptiform discharge is established in the displayed segment."

or:

"A sharply contoured transient is visible; epileptiform significance is
not established from the screenshot alone."

Do not use "epileptiform discharge is present" unless IMAGE OBSERVATIONS
provide strong explicit support.

============================================================
FINDING 4 — RHYTHMIC / ACTIVATION / ARTIFACT
============================================================

Use the fourth finding for the most clinically relevant image-supported item.

Possible subjects:

- rhythmic or periodic activity
- temporal evolution
- artifact
- activation procedure information

Do NOT state:

"Reactive to eye opening and closing."

"Hyperventilation produced no abnormality."

"Photic stimulation produced no abnormality."

unless those events and responses are directly demonstrated in
IMAGE OBSERVATIONS.

A single static screenshot usually does not establish these full-study
activation findings.

If no definite evolving rhythmic activity is visible:

"No definite evolving rhythmic or electrographic seizure pattern is
established in the displayed segment."

If artifact is clearly visible, a concise artifact finding may be used instead.

If neither can be established:

"No specific additional abnormality is established from the supplied image."

============================================================
LOCALIZATION
============================================================

Do not use terms such as frontal, temporal, occipital, focal, generalized,
maximal over, left-sided, or right-sided unless visible channel labels and
waveform relationships clearly support the statement.

Avoid claims such as "maximal amplitude in frontal channels" unless the
distribution is convincingly demonstrated by the image.

============================================================
NORMALITY
============================================================

A screenshot does not establish that the complete EEG study is normal.

If no definite abnormality is visible use:

"No definite abnormality is identified in the displayed EEG segment."

Do not write:

"This awake EEG is normal."

unless complete-study information has explicitly been supplied.

============================================================
ABNORMALITY
============================================================

If a definite image-supported abnormality exists, describe it briefly.

Do not infer epilepsy, seizure disorder, encephalopathy, prognosis, or a
structural lesion from a screenshot alone.

============================================================
INDETERMINATE
============================================================

Use:

"The supplied EEG image is indeterminate for normal-versus-abnormal
classification."

only when the available image genuinely does not support a reliable
segment-level conclusion.

Do not automatically make every screenshot indeterminate.

============================================================
EDUCATIONAL / REFERENCE EEG
============================================================

If the image is an educational chart rather than a patient recording:

- set image_type to "educational_eeg"
- set is_clinical_recording to false
- set image_classification to "educational_eeg_reference"
- report visible labels and examples
- do not create a patient diagnosis

============================================================
OUTPUT JSON
============================================================

Return ONLY valid JSON using this schema:

{
  "is_eeg": true,
  "image_type": "eeg|clinical_eeg|educational_eeg|ecg|other|uncertain",
  "is_clinical_recording": true,
  "image_classification": "clinical_eeg_recording|educational_eeg_reference|non_eeg|uncertain",
  "complete_study_available": false,
    "eye_reactivity": "reactive|nonreactive|not_established",
  "hyperventilation_response": "no_additional_abnormality|abnormal_response|not_established",
  "photic_response": "no_additional_abnormality|abnormal_response|not_established",
  "impression": [],
  "visible_findings": [],
  "pattern_interpretation": [],
  "pattern_characteristics": [],
  "frequency_assessment": [],
  "clinical_significance": [],
  "limitations": [],
  "diagnostic_safety": [],
  "displayed_metadata": [],
  "reason": [],
  "clinical_advice": [],
  "confidence": {
    "image_observation": "low|moderate|high",
    "technical_metadata": "low|moderate|high|not_available",
    "pattern_interpretation": "low|moderate|high",
    "clinical_interpretation": "low|moderate|high"
  }
}

For a clinical EEG screenshot, visible_findings should preferably contain
FOUR concise physician-style findings matching the structure above.

Use the supplied IMAGE OBSERVATIONS as the sole evidence for those findings.
"""

# ============================================================
# DOCTOR-REPORT STRUCTURE GUIDANCE
# ============================================================

DOCTOR_REPORT_STRUCTURE = """
Use the following doctor-report component structure for CLINICAL EEG images.

This block controls the organization of the patient-facing clinical content.
If an earlier generic instruction asks for five findings, use the FOUR
doctor-report findings below instead.

============================================================
PROCEDURE
============================================================

The Procedure section represents the recording setup / electrode-placement
method only.

When the electrode-placement standard is directly visible or otherwise
explicitly supported by IMAGE OBSERVATIONS, it may be described.

Do not invent an electrode-placement standard.

Do not repeat screen metadata such as:
- clock time
- page number
- speed
- sensitivity
- filters
- notch setting
- montage name

in the patient-facing Procedure section.

If the electrode-placement method is not established from the supplied image,
use:

"EEG waveform image reviewed from the supplied recording."

============================================================
FINDINGS
============================================================

For a clinical EEG image, organize the clinical findings into EXACTLY FOUR
concise findings, in this order:

1. BACKGROUND ACTIVITY

Describe, when directly supported:
- dominant background rhythm
- frequency
- symmetry
- synchrony
- reactivity

Do not invent any of these components.

When the IMAGE OBSERVATIONS directly support an alpha background, preserve
that alpha description in the final report.

When a directly supported approximate dominant frequency range is available,
use the narrowest range actually supported by IMAGE OBSERVATIONS. Do not widen
or alter the range merely to match a reference report.

When symmetry and synchrony are directly supported, explicitly include both
terms in the physician-facing background sentence.

If frequency cannot be established, do not estimate an exact Hz value from
waveform appearance alone.

If eye-opening / eye-closing reactivity is not directly demonstrated by the
IMAGE OBSERVATIONS, do not claim it.

2. ACTIVATION RESPONSE

Actively inspect the supplied EEG image for activation evidence.

Look specifically for:

- eye opening / eye closing labels
- EO / EC annotations
- hyperventilation / HV labels
- photic stimulation / IPS / photic labels
- event markers associated with these procedures
- explicit text stating the response to an activation procedure
- clearly demonstrated before/after waveform changes associated with a
  labeled activation event

Set:

eye_reactivity = "reactive"
ONLY when eye opening/closing is explicitly demonstrated and the
corresponding background response is visible.

eye_reactivity = "nonreactive"
ONLY when the eye opening/closing event is explicitly demonstrated and
lack of reactivity is directly supported.

hyperventilation_response = "no_additional_abnormality"
ONLY when hyperventilation is explicitly shown/documented and no
additional abnormality is demonstrated.

hyperventilation_response = "abnormal_response"
ONLY when hyperventilation is explicitly shown/documented and an
additional abnormal EEG response is directly demonstrated.

photic_response = "no_additional_abnormality"
ONLY when photic stimulation is explicitly shown/documented and no
additional abnormality is demonstrated.

photic_response = "abnormal_response"
ONLY when photic stimulation is explicitly shown/documented and an
abnormal response is directly demonstrated.

If the activation procedure or its response is not visible/documented,
use "not_established".

Do NOT infer:
- eye reactivity merely because alpha activity is present
- hyperventilation merely because it is commonly performed
- photic stimulation merely because it is commonly performed

3. ABNORMAL SLOWING

State whether abnormal slowing is:
- present in the displayed segment
- not identified in the displayed segment
- not reliably established from the supplied image

When IMAGE OBSERVATIONS explicitly support absence of definite abnormal
slowing in the displayed segment, prefer the concise wording:

"No definite abnormal slowing is identified in the displayed segment."

Do not infer focal or generalized slowing unless image evidence supports it.

4. ARTIFACT

Describe only image-supported artifact, such as:
- eye-blink artifact
- movement artifact
- muscle artifact
- electrode artifact
- other technical artifact

When eye-blink and/or movement artifact is directly supported, prefer a short
doctor-style statement such as:

"Eye-blink and movement artifact is present in the displayed segment."

Do not label artifact as eye-blink or movement unless IMAGE OBSERVATIONS
support that classification.

If no artifact type can be reliably classified, use:

"No specific artifact classification is established from the supplied image."

============================================================
IMPRESSION
============================================================

Use ONE concise impression.

If no definite abnormality is identified in the displayed segment, use:

"No definite abnormality is identified in the displayed EEG segment."

If a definite image-supported abnormality is present, use:

"The displayed EEG segment is abnormal, showing [supported abnormality]."

If the supplied image is genuinely insufficient for a segment-level
normal-versus-abnormal assessment, use:

"The supplied EEG image is indeterminate for normal-versus-abnormal classification."

Do not automatically use the indeterminate conclusion.

Do not use a whole-study statement such as:

"This awake EEG is normal."

unless complete-study information has explicitly been supplied.

============================================================
EVIDENCE RULE
============================================================

The doctor-report structure is a FORMAT and ORGANIZATION template only.

It must never cause the model to invent:
- eye reactivity
- hyperventilation
- photic stimulation
- slowing
- artifact
- exact frequency
- normality
- abnormality

IMAGE OBSERVATIONS remain the sole evidence that a clinical finding exists.
"""

# ============================================================
# BASIC HELPERS
# ============================================================

def text(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    return str(value).strip()


def clean_list(value: Any) -> List[str]:
    if value is None:
        return []

    if isinstance(value, str):
        value = [value]

    if not isinstance(value, list):
        value = [value]

    result = []

    for item in value:
        # Gemini may occasionally return a structured finding such as:
        # {"category": "BACKGROUND ACTIVITY", "description": "..."}
        # Keep only the clinical text instead of stringifying the dictionary.
        if isinstance(item, dict):
            item_text = text(
                item.get("description")
                or item.get("finding")
                or item.get("text")
                or item.get("value")
                or ""
            )
        else:
            item_text = text(item)

        if item_text:
            result.append(item_text)

    return result


def dedupe(items: List[str]) -> List[str]:
    result = []
    seen = set()

    for item in items:
        key = re.sub(
            r"\s+",
            " ",
            item.lower(),
        ).strip()

        if key not in seen:
            seen.add(key)
            result.append(item)

    return result


# ============================================================
# CONFIDENCE
# ============================================================

def normalize_confidence(
    value: Any,
) -> Dict[str, str]:

    allowed = {
        "low",
        "moderate",
        "high",
        "not_available",
    }

    if isinstance(value, dict):
        result = {}

        for key in [
            "image_observation",
            "technical_metadata",
            "pattern_interpretation",
            "clinical_interpretation",
        ]:
            current = text(
                value.get(key)
            ).lower()

            if current not in allowed:
                current = "low"

            result[key] = current

        return result

    old = text(value).lower()

    if old not in {
        "low",
        "moderate",
        "high",
    }:
        old = "low"

    if old == "high":
        return {
            "image_observation": "high",
            "technical_metadata": "moderate",
            "pattern_interpretation": "moderate",
            "clinical_interpretation": "low",
        }

    if old == "moderate":
        return {
            "image_observation": "moderate",
            "technical_metadata": "low",
            "pattern_interpretation": "moderate",
            "clinical_interpretation": "low",
        }

    return {
        "image_observation": "low",
        "technical_metadata": "low",
        "pattern_interpretation": "low",
        "clinical_interpretation": "low",
    }


# ============================================================
# UNSUPPORTED DIMENSION CLAIMS
# ============================================================

def remove_dimension_claims(
    items: List[str],
) -> List[str]:

    blocked = [
        r"\b\d+\s*[x×]\s*\d+\b",
        r"\b\d+\s+by\s+\d+\b",
        r"\bimage\s+(width|height|resolution)\b",
        r"\bresolution\s*[:=]",
        r"\bpixel\s+(width|height|size)\b",
    ]

    result = []

    for item in items:
        if any(
            re.search(
                pattern,
                item,
                flags=re.I,
            )
            for pattern in blocked
        ):
            continue

        result.append(item)

    return result


# ============================================================
# CHANNEL / DERIVATION PROTECTION
# ============================================================

def derivation_regex() -> List[str]:
    return [
        r"\bFp[1-2]\b",
        r"\bF[3-8]\b",
        r"\bC[3-8]\b",
        r"\bP[3-8]\b",
        r"\bO[1-2]\b",
        r"\bT[3-8]\b",
        r"\bA[1-2]\b",

        r"\bFp[1-2]\s*[-–]\s*",
        r"\bF[3-8]\s*[-–]\s*",
        r"\bC[3-8]\s*[-–]\s*",
        r"\bP[3-8]\s*[-–]\s*",
        r"\bO[1-2]\s*[-–]\s*",
        r"\bT[3-8]\s*[-–]\s*",
    ]


def remove_unsupported_channel_groups(
    items: List[str],
) -> List[str]:

    patterns = derivation_regex()
    result = []

    for item in items:
        if any(
            re.search(
                pattern,
                item,
                flags=re.I,
            )
            for pattern in patterns
        ):
            continue

        result.append(item)

    return result


# ============================================================
# LOCALIZATION PROTECTION
# ============================================================

def remove_unsupported_localization(
    items: List[str],
) -> List[str]:

    blocked = [
        r"\blocali[sz](ation|ed)\b",
        r"\bfocal\b",
        r"\btemporal lobe\b",
        r"\bfrontal lobe\b",
        r"\bparietal lobe\b",
        r"\boccipital lobe\b",
        r"\bfrontotemporal\b",
        r"\bfronto[- ]central\b",
        r"\btemporo[- ]occipital\b",
        r"\bleft hemisphere\b",
        r"\bright hemisphere\b",
        r"\bleft temporal\b",
        r"\bright temporal\b",
        r"\bleft frontal\b",
        r"\bright frontal\b",
    ]

    result = []

    for item in items:
        if any(
            re.search(
                pattern,
                item,
                flags=re.I,
            )
            for pattern in blocked
        ):
            continue

        result.append(item)

    return result


# ============================================================
# GENERIC IMAGE QUALITY FILTER
# ============================================================

def remove_generic_image_quality(
    items: List[str],
) -> List[str]:

    blocked = [
        r"\bimage quality\b",
        r"\blimited image\b",
        r"\blow[- ]resolution\b",
        r"\bhigh[- ]resolution\b",
        r"\bpoor[- ]quality\b",
        r"\bsmall image\b",
        r"\bblurred image\b",
        r"\bimage is too small\b",
    ]

    result = []

    for item in items:
        if any(
            re.search(
                pattern,
                item,
                flags=re.I,
            )
            for pattern in blocked
        ):
            continue

        result.append(item)

    return result


# ============================================================
# DISTRIBUTION LANGUAGE
# ============================================================

def normalize_distribution_language(
    items: List[str],
) -> List[str]:

    replacements = [
        (
            r"\bgeneralized\b",
            "appears broadly distributed in the displayed traces",
        ),
        (
            r"\bdiffuse\b",
            "appears broadly distributed in the displayed traces",
        ),
    ]

    result = []

    for item in items:
        updated = item

        for pattern, replacement in replacements:
            updated = re.sub(
                pattern,
                replacement,
                updated,
                flags=re.I,
            )

        result.append(updated)

    return result


# ============================================================
# UNSAFE CLINICAL CLAIM PROTECTION
# ============================================================

def neutralize_unsafe_claims(
    items: List[str],
) -> List[str]:

    result = []

    replacements = [
        (
            r"\bdefinite seizure\b",
            "waveform activity requiring clinician review",
        ),
        (
            r"\bconfirmed seizure\b",
            "waveform activity requiring clinician review",
        ),
        (
            r"\bconfirmed ictal activity\b",
            "rhythmic waveform activity requiring clinician review",
        ),
        (
            r"\bictal activity\b",
            "rhythmic waveform activity requiring clinician review",
        ),
        (
            r"\bdiagnosis of epilepsy\b",
            "EEG finding requiring clinical correlation",
        ),
        (
            r"\bepilepsy diagnosis\b",
            "EEG finding requiring clinical correlation",
        ),
        (
            r"\bepileptic seizure\b",
            "waveform activity requiring clinician review",
        ),
        (
            r"\bexactly\s+\d+\.?\d*\s*Hz\b",
            "at an unverified frequency",
        ),
        (
            r"\bexactly\s+\d+\.?\d*\s*(?:µV|uV|mV)\b",
            "at an unverified amplitude",
        ),
    ]

    for item in items:
        updated = item

        for pattern, replacement in replacements:
            updated = re.sub(
                pattern,
                replacement,
                updated,
                flags=re.I,
            )

        result.append(updated)

    return result


# ============================================================
# UNSUPPORTED "CANNOT EXCLUDE" LANGUAGE
# ============================================================

def is_unwanted_cannot_conclude(
    item: str,
) -> bool:

    blocked = [
        r"\bpossible seizure\b",
        r"\bpossible epilepsy\b",
        r"\bpossible ictal activity\b",
        r"\bpossible epileptic\b",
        r"\bseizure cannot be excluded\b",
        r"\bepilepsy cannot be excluded\b",
    ]

    return any(
        re.search(
            pattern,
            item,
            flags=re.I,
        )
        for pattern in blocked
    )


def normalize_cannot_conclude(
    items: List[str],
) -> List[str]:

    return [
        item
        for item in items
        if not is_unwanted_cannot_conclude(item)
    ]


# ============================================================
# LANGUAGE CLEANUP
# ============================================================

def clean_language(
    items: List[str],
) -> List[str]:

    result = []

    for item in items:
        item = re.sub(
            r"\s+",
            " ",
            item,
        ).strip()

        if item:
            result.append(item)

    return dedupe(result)


# ============================================================
# REPORT TEXT HELPERS
# ============================================================

def _report_text(
    report: Dict[str, Any],
) -> str:

    parts: List[str] = []

    for section in [
        "impression",
        "visible_findings",
        "pattern_interpretation",
        "displayed_metadata",
        "reason",
        "clinical_advice",
    ]:
        for value in (
            report.get(section, [])
            or []
        ):
            value = text(value)

            if value:
                parts.append(value)

    return " ".join(parts).strip()


def _is_educational_or_reference_image(
    report: Dict[str, Any],
) -> bool:

    content = _report_text(
        report
    ).lower()

    markers = [
        "educational",
        "illustrative",
        "illustration",
        "reference chart",
        "frequency chart",
        "stylized waveform",
        "stylized waveforms",
        "schematic",
        "reference graphic",
        "not an actual patient",
        "not a patient eeg",
        "not actual patient",
        "not a patient recording",
    ]

    return any(
        marker in content
        for marker in markers
    )


def _explicit_frequency_ranges(
    report: Dict[str, Any],
) -> List[str]:

    combined_parts: List[str] = []

    for section in [
        "displayed_metadata",
        "visible_findings",
        "impression",
        "reason",
        "pattern_interpretation",
    ]:
        for value in (
            report.get(section, [])
            or []
        ):
            value = text(value)

            if value:
                combined_parts.append(value)

    combined = " ".join(
        combined_parts
    )

    band_pattern = re.compile(
        r"\b"
        r"(Gamma|Beta|Alpha|Theta|Delta)"
        r"\s*\(\s*"
        r"(\d+(?:\.\d+)?)"
        r"\s*-\s*"
        r"(\d+(?:\.\d+)?)"
        r"\s*Hz\s*\)",
        flags=re.I,
    )

    found = []

    for match in band_pattern.finditer(
        combined
    ):
        band = match.group(1).capitalize()
        lo = match.group(2)
        hi = match.group(3)

        if not any(
            item[0].lower()
            == band.lower()
            for item in found
        ):
            found.append(
                (band, lo, hi)
            )

    order = {
        "gamma": 0,
        "beta": 1,
        "alpha": 2,
        "theta": 3,
        "delta": 4,
    }

    found.sort(
        key=lambda item: order.get(
            item[0].lower(),
            99,
        )
    )

    return [
        f"{band}: {lo}-{hi} Hz"
        for band, lo, hi
        in found
    ]


def _technical_metadata_available(
    report: Dict[str, Any],
) -> bool:

    metadata = " ".join(
        text(value)
        for value in (
            report.get(
                "displayed_metadata"
            )
            or []
        )
        if text(value)
    )

    if not metadata:
        return False

    technical_positive = re.compile(
        r"(time\s*scale|seconds?|secs?|"
        r"amplitude|calibration|sensitivity|"
        r"\buV\b|µV|mV|montage|channels?|"
        r"derivations?|electrodes?|10[- ]?20|"
        r"sampling\s*rate|sampling|filters?)",
        flags=re.I,
    )

    technical_negative = re.compile(
        r"\b(no|not|without|unavailable)\b.*?"
        r"(time\s*scale|amplitude|calibration|"
        r"sensitivity|montage|channels?|"
        r"derivations?|electrodes?|sampling|"
        r"filters?)",
        flags=re.I,
    )

    return bool(
        technical_positive.search(
            metadata
        )
        and not technical_negative.search(
            metadata
        )
    )


# ============================================================
# ENRICHMENT
# ============================================================

def _enrich_pattern_characteristics(
    report: Dict[str, Any],
) -> List[str]:

    existing = clean_language(
        clean_list(
            report.get(
                "pattern_characteristics"
            )
        )
    )

    if existing:
        return existing

    if _is_educational_or_reference_image(
        report
    ):

        ranges = _explicit_frequency_ranges(
            report
        )

        result = [
            (
                "The supplied image is an educational/reference EEG graphic "
                "rather than a patient EEG recording."
            )
        ]

        if ranges:
            result.append(
                (
                    "The graphic contains explicitly labeled EEG frequency "
                    "information; these labels are educational reference "
                    "values and are not patient measurements."
                )
            )

        return result

    return []


def _enrich_frequency_assessment(
    report: Dict[str, Any],
) -> List[str]:

    existing = clean_language(
        clean_list(
            report.get(
                "frequency_assessment"
            )
        )
    )

    if existing:
        return existing

    ranges = _explicit_frequency_ranges(
        report
    )

    educational = (
        _is_educational_or_reference_image(
            report
        )
    )

    content = _report_text(
        report
    ).lower()

    if ranges:
        result = list(ranges)

        if educational:
            result.append(
                (
                    "These frequency ranges are explicitly displayed by the "
                    "source image and are not measurements independently "
                    "obtained from the stylized waveform examples."
                )
            )

        return result

    if (
        "decrease in frequency"
        in content
        or "decreases from gamma to delta"
        in content
        or "progressively decreases"
        in content
    ):
        return [
            (
                "Qualitatively, the displayed waveforms progress from faster "
                "activity at the top of the chart to slower activity at the bottom."
            ),
            (
                "Exact frequency should not be estimated from waveform appearance "
                "alone when a recording time scale is not available."
            ),
        ]

    if educational:
        return [
            (
                "No independently measured patient EEG frequency can be established "
                "from this illustrative graphic."
            )
        ]

    return [
        (
            "No quantitative frequency measurement is established "
            "from the available image."
        )
    ]


def _enrich_reason(
    report: Dict[str, Any],
) -> List[str]:

    existing = clean_language(
        clean_list(
            report.get(
                "reason"
            )
        )
    )

    if existing:
        return existing

    if _is_educational_or_reference_image(
        report
    ):
        return [
            (
                "The interpretation is based only on the visible labels, "
                "waveforms, and descriptive content of the supplied "
                "educational/reference image."
            )
        ]

    return [
        (
            "The interpretation is based only on waveform features directly "
            "supported by the supplied image observations. Findings that "
            "cannot be established from the displayed EEG segment are not inferred."
        )
    ]


def _enrich_clinical_significance(
    report: Dict[str, Any],
) -> List[str]:

    educational = (
        _is_educational_or_reference_image(
            report
        )
    )

    existing = clean_language(
        clean_list(
            report.get(
                "clinical_significance"
            )
        )
    )

    if educational:
        content = _report_text(
            report
        ).lower()

        if (
            "depressed" in content
            and "normal" in content
        ):
            return [
                (
                    "The figure provides an educational comparison of two "
                    "source-labeled waveform examples, 'Depressed' and 'Normal'."
                ),
                (
                    "The labels describe the intended categories in the source "
                    "graphic and must not be interpreted as a patient's "
                    "psychiatric or neurological diagnosis."
                ),
                (
                    "No patient-specific clinical significance can be established "
                    "from this graphic."
                ),
            ]

        if existing:
            return existing

        return [
            (
                "The figure provides educational information about "
                "waveform characteristics."
            ),
            (
                "No patient-specific clinical significance can be established "
                "from this graphic."
            ),
        ]

    if existing:
        return existing

    return [
        (
            "Patient-specific clinical significance cannot be established "
            "from the supplied screenshot alone."
        ),
        (
            "Interpretation should be correlated with the complete EEG "
            "recording and relevant clinical context."
        ),
    ]


def _enrich_limitations(
    report: Dict[str, Any],
) -> List[str]:

    existing = clean_language(
        clean_list(
            report.get(
                "limitations"
            )
        )
    )

    if existing:
        return existing

    educational = (
        _is_educational_or_reference_image(
            report
        )
    )

    content = _report_text(
        report
    ).lower()

    if educational:
        return [
            (
                "The image is an educational/reference graphic rather than "
                "a patient EEG recording."
            ),
            (
                "No patient-specific montage, channel labels, time scale, "
                "amplitude calibration, or acquisition parameters are provided."
            ),
            (
                "The displayed frequency ranges are source labels and should "
                "not be treated as measurements from an original EEG recording."
            ),
        ]

    result = []

    if (
        "no time scale" in content
        or "time scale is not"
        in content
    ):
        result.append(
            (
                "A visible recording time scale is not available, limiting "
                "quantitative frequency assessment."
            )
        )

    if (
        "no amplitude calibration"
        in content
        or "amplitude calibration is not"
        in content
        or "calibration is unavailable"
        in content
    ):
        result.append(
            (
                "Visible amplitude calibration is not available, limiting "
                "quantitative amplitude assessment."
            )
        )

    if (
        "no channel labels"
        in content
        or "no montage"
        in content
    ):
        result.append(
            (
                "Channel and montage information is not available for assessing "
                "distribution or localization."
            )
        )

    result.extend([
        (
            "A static screenshot may not represent the complete EEG recording "
            "or unseen temporal evolution."
        ),
        (
            "Patient-specific interpretation requires the original EEG recording "
            "and relevant clinical context."
        ),
    ])

    return dedupe(result)


def _enrich_diagnostic_safety(
    report: Dict[str, Any],
) -> List[str]:

    existing = clean_language(
        clean_list(
            report.get(
                "diagnostic_safety"
            )
        )
    )

    required = (
        "This image-only analysis is not a definitive medical diagnosis. "
        "The original EEG recording and relevant clinical context should "
        "be reviewed by a qualified EEG professional."
    )

    if existing:
        if not any(
            re.search(
                r"not a definitive medical diagnosis",
                item,
                flags=re.I,
            )
            for item in existing
        ):
            existing.insert(
                0,
                required,
            )

        return dedupe(existing)

    return [required]


def enrich_report_sections(
    report: Dict[str, Any],
) -> Dict[str, Any]:

    report = dict(report)

    report[
        "pattern_characteristics"
    ] = _enrich_pattern_characteristics(
        report
    )

    report[
        "frequency_assessment"
    ] = _enrich_frequency_assessment(
        report
    )

    report[
        "reason"
    ] = _enrich_reason(
        report
    )

    report[
        "clinical_significance"
    ] = _enrich_clinical_significance(
        report
    )

    report[
        "limitations"
    ] = _enrich_limitations(
        report
    )

    report[
        "diagnostic_safety"
    ] = _enrich_diagnostic_safety(
        report
    )

    confidence = normalize_confidence(
        report.get(
            "confidence"
        )
    )

    if _is_educational_or_reference_image(
        report
    ):
        confidence[
            "image_observation"
        ] = "high"

        confidence[
            "technical_metadata"
        ] = "not_available"

        confidence[
            "pattern_interpretation"
        ] = "high"

        confidence[
            "clinical_interpretation"
        ] = "low"

    elif not _technical_metadata_available(
        report
    ):
        confidence[
            "technical_metadata"
        ] = "not_available"

    report["confidence"] = confidence

    return report


# ============================================================
# REPORT NORMALIZATION
# ============================================================

def normalize_report(
    report: Dict[str, Any],
) -> Dict[str, Any]:

    sections = [
        "impression",
        "visible_findings",
        "pattern_interpretation",
        "pattern_characteristics",
        "frequency_assessment",
        "clinical_significance",
        "limitations",
        "diagnostic_safety",
        "displayed_metadata",
        "reason",
        "clinical_advice",
    ]

    normalized: Dict[
        str,
        Any,
    ] = {}

    for section in sections:
        values = clean_list(
            report.get(section)
        )

        values = remove_dimension_claims(
            values
        )

        # Keep channel labels / technical location information when
        # Gemini explicitly places them in displayed_metadata.
        # Clinical interpretation sections remain protected from
        # unsupported channel/localization claims.
        if section != "displayed_metadata":

            values = remove_unsupported_channel_groups(
                values
            )

            values = remove_unsupported_localization(
                values
            )

        values = remove_generic_image_quality(
            values
        )

        values = normalize_distribution_language(
            values
        )

        values = neutralize_unsafe_claims(
            values
        )

        values = normalize_cannot_conclude(
            values
        )

        values = clean_language(
            values
        )

        normalized[
            section
        ] = values

    normalized[
        "confidence"
    ] = normalize_confidence(
        report.get(
            "confidence"
        )
    )

    is_eeg = report.get(
        "is_eeg"
    )

    if isinstance(
        is_eeg,
        bool,
    ):
        normalized[
            "is_eeg"
        ] = is_eeg
    else:
        normalized[
            "is_eeg"
        ] = True

    image_type = text(
        report.get(
            "image_type"
        )
    ).lower()

    allowed_types = {
        "eeg",
        "clinical_eeg",
        "educational_eeg",
        "ecg",
        "other",
        "uncertain",
    }

    if image_type not in allowed_types:
        image_type = "eeg"

    normalized[
        "image_type"
    ] = image_type

    educational = (
        _is_educational_or_reference_image(
            normalized
        )
    )

    declared_clinical = report.get(
        "is_clinical_recording"
    )

    if isinstance(
        declared_clinical,
        bool,
    ):
        is_clinical_recording = (
            declared_clinical
        )

    elif image_type == "educational_eeg":
        is_clinical_recording = False

    elif image_type == "clinical_eeg":
        is_clinical_recording = True

    elif image_type in {
        "ecg",
        "other",
    }:
        is_clinical_recording = False

    elif image_type == "eeg":
        is_clinical_recording = (
            not educational
        )

    else:
        is_clinical_recording = False

    if image_type in {
        "ecg",
        "other",
    }:
        image_classification = "non_eeg"

    elif image_type == "uncertain":
        image_classification = "uncertain"

    elif (
        educational
        or image_type
        == "educational_eeg"
    ):
        image_classification = (
            "educational_eeg_reference"
        )

        is_clinical_recording = False

        normalized[
            "image_type"
        ] = "educational_eeg"

    else:
        image_classification = (
            "clinical_eeg_recording"
        )

        if image_type == "eeg":
            normalized[
                "image_type"
            ] = "clinical_eeg"

    normalized[
        "is_clinical_recording"
    ] = is_clinical_recording

    normalized[
        "image_classification"
    ] = image_classification

    normalized[
        "complete_study_available"
    ] = bool(
        report.get(
            "complete_study_available",
            False,
        )
    )

    normalized[
        "clinical_review_status"
    ] = (
        "STUDY_LEVEL_REVIEW"
        if normalized[
            "complete_study_available"
        ]
        else "PRELIMINARY_IMAGE_REVIEW"
    )

    if not normalized[
        "diagnostic_safety"
    ]:
        normalized[
            "diagnostic_safety"
        ] = [
            (
                "This image-only analysis is not a definitive medical diagnosis. "
                "The original EEG recording and relevant clinical context should "
                "be reviewed by a qualified EEG professional."
            )
        ]
    # ========================================================
    # STRUCTURED ACTIVATION FIELDS
    # ========================================================

    eye_reactivity = text(
        report.get(
            "eye_reactivity",
            "not_established",
        )
    ).lower()

    if eye_reactivity not in {
        "reactive",
        "nonreactive",
        "not_established",
    }:
        eye_reactivity = "not_established"

    normalized[
        "eye_reactivity"
    ] = eye_reactivity

    # ========================================================
    # HYPERVERNTILATION RESPONSE
    # ========================================================

    hyperventilation_response = text(
        report.get(
            "hyperventilation_response",
            "not_established",
        )
    ).lower()

    if hyperventilation_response not in {
        "no_additional_abnormality",
        "abnormal_response",
        "not_established",
    }:
        hyperventilation_response = "not_established"

    normalized[
        "hyperventilation_response"
    ] = hyperventilation_response


    photic_response = text(
        report.get(
            "photic_response",
            "not_established",
        )
    ).lower()

    if photic_response not in {
        "no_additional_abnormality",
        "abnormal_response",
        "not_established",
    }:
        photic_response = "not_established"

    normalized[
        "photic_response"
    ] = photic_response

    return enrich_report_sections(
        normalized
    )


# ============================================================
# CONFIDENCE DISPLAY
# ============================================================

def build_confidence_display(
    confidence: Optional[
        Dict[str, str]
    ],
) -> str:

    confidence = normalize_confidence(
        confidence
    )

    def pretty(
        value: str,
    ) -> str:

        if value == "not_available":
            return "Not available"

        return value.capitalize()

    return "\n".join([
        (
            "Image observation: "
            + pretty(
                confidence.get(
                    "image_observation",
                    "low",
                )
            )
        ),
        (
            "Technical metadata: "
            + pretty(
                confidence.get(
                    "technical_metadata",
                    "not_available",
                )
            )
        ),
        (
            "Pattern interpretation: "
            + pretty(
                confidence.get(
                    "pattern_interpretation",
                    "low",
                )
            )
        ),
        (
            "Clinical interpretation: "
            + pretty(
                confidence.get(
                    "clinical_interpretation",
                    "low",
                )
            )
        ),
    ])


# ============================================================
# PHYSICIAN-FACING REPORT
# ============================================================

def build_clinical_report(
    report: Dict[str, Any],
) -> str:

    # ========================================================
    # HELPERS
    # ========================================================

    def clean(items: Any) -> List[str]:

        if items is None:
            return []

        if isinstance(items, str):
            items = [items]

        if not isinstance(items, list):
            items = [items]

        result = []
        seen = set()

        for item in items:

            if isinstance(item, dict):
                value = text(
                    item.get("description")
                    or item.get("finding")
                    or item.get("text")
                    or item.get("value")
                    or ""
                )
            else:
                value = text(item)

            value = re.sub(
                r"\s+",
                " ",
                value,
            ).strip(" -•")

            if not value:
                continue

            key = re.sub(
                r"[.!?]+$",
                "",
                value.lower(),
            ).strip()

            if key in seen:
                continue

            seen.add(key)
            result.append(value)

        return result


    def sentence(value: str) -> str:

        value = re.sub(
            r"\s+",
            " ",
            str(value or ""),
        ).strip(" -•")

        value = re.sub(
            r"^(BACKGROUND ACTIVITY|BACKGROUND|ACTIVATION RESPONSE|ACTIVATION|"
            r"ABNORMAL SLOWING|SLOWING|ARTIFACT)\s*:\s*",
            "",
            value,
            flags=re.I,
        ).strip()

        if not value:
            return ""

        if value.endswith((".", "!", "?")):
            return value

        return value + "."


    def first_matching(
        items: List[str],
        terms: List[str],
        exclude: Optional[List[str]] = None,
    ) -> str:

        exclude = exclude or []

        for item in items:

            low = item.lower()

            if not any(term in low for term in terms):
                continue

            if any(term in low for term in exclude):
                continue

            return sentence(item)

        return ""


    def unique_findings(
        findings: List[str],
    ) -> List[str]:

        result = []
        seen = set()

        for item in findings:

            item = sentence(item)

            if not item:
                continue

            key = re.sub(
                r"\s+",
                " ",
                item.lower(),
            ).strip()

            if key in seen:
                continue

            seen.add(key)
            result.append(item)

        return result


    def combined_has(terms: List[str]) -> bool:
        return any(term in all_text for term in terms)


    # ========================================================
    # REPORT DATA
    # ========================================================

    is_eeg = bool(
        report.get(
            "is_eeg",
            True,
        )
    )

    image_type = text(
        report.get(
            "image_type",
            "",
        )
    ).lower()

    complete_study_available = bool(
        report.get(
            "complete_study_available",
            False,
        )
    )

    visible = clean(
        report.get("visible_findings")
    )

    interpreted = clean(
        report.get("pattern_interpretation")
    )

    characteristics = clean(
        report.get("pattern_characteristics")
    )

    frequency = clean(
        report.get("frequency_assessment")
    )

    impression_items = clean(
        report.get("impression")
    )

    reason = clean(
        report.get("reason")
    )

    metadata = clean(
        report.get("displayed_metadata")
    )

    all_items = (
        visible
        + interpreted
        + characteristics
        + frequency
    )

    # Physician-facing findings must be derived only from image-grounded
    # observation sections. RAG-generated impression/reason text is intentionally
    # excluded here so reference knowledge cannot create or weaken image findings.
    all_text = " ".join(
        all_items
    ).lower()

    metadata_text = " ".join(
        metadata
    ).lower()


    # ========================================================
    # NON EEG
    # ========================================================

    if not is_eeg:

        reason_text = (
            reason[0]
            if reason
            else
            "The supplied image is not identified as an EEG recording."
        )

        return "\n".join([
            "EEG REPORT",
            "",
            "Procedure:",
            "The supplied image was reviewed for EEG waveform characteristics.",
            "",
            "Findings:",
            f"• {sentence(reason_text)}",
            "",
            "Impression:",
            (
                "EEG interpretation was not performed because the supplied "
                "image was not identified as an EEG recording."
            ),
        ])


    # ========================================================
    # EDUCATIONAL EEG
    # ========================================================

    if (
        image_type == "educational_eeg"
        or _is_educational_or_reference_image(report)
    ):

        findings = unique_findings(
            visible
            + characteristics
            + frequency
        )

        if not findings:
            findings = [
                "The supplied image appears to be an educational EEG reference graphic."
            ]

        return "\n".join([
            "EEG REPORT",
            "",
            "Procedure:",
            "The supplied EEG reference image was reviewed.",
            "",
            "Findings:",
            *[
                f"• {sentence(item)}"
                for item in findings[:4]
            ],
            "",
            "Impression:",
            (
                "Educational/reference EEG image; "
                "no patient-specific EEG diagnosis is made."
            ),
        ])


    # ========================================================
    # PROCEDURE
    # ========================================================

    # Use the doctor's procedure style only when 10-20 placement is
    # explicitly supported by visible metadata/observations.
    if (
        "10-20" in metadata_text
        or "10–20" in metadata_text
        or "10/20" in metadata_text
        or "international standard of electrode placement" in all_text
    ):
        procedure = (
            "10-20 international standard of electrode placement was utilized."
        )
    else:
        procedure = (
            "EEG waveform image reviewed from the supplied recording."
        )


    # ========================================================
    # FINDING 1 — BACKGROUND
    # ========================================================

    # Build one compact doctor-style background sentence from
    # supported features across the observation set.
    alpha_present = combined_has([
        "alpha",
        "8-9 hz",
        "08-09 hz",
        "8–9 hz",
        "08–09 hz",
    ])

    beta_present = combined_has(["beta"])
    theta_present = combined_has(["theta"])
    delta_present = combined_has(["delta"])

    symmetric_present = combined_has([
        "symmetric",
        "symmetrical",
        "symmetry is preserved",
        "appears generally symmetric",
    ])

    synchronous_present = combined_has([
        "synchronous",
        "synchrony",
    ])

    reactivity_positive = combined_has([
        "reactive to eye opening",
        "reactive to eyes opening",
        "reactive to eye closing",
        "reactive to eyes closing",
        "reactive to eye opening and closing",
        "reactive to eyes opening and closing",
        "reactivity to eye opening",
        "reactivity to eyes opening",
        "reactivity is present",
        "reactivity present",
    ])

    explicit_background = first_matching(
        all_items,
        [
            "background",
            "dominant rhythm",
            "dominant frequency",
            "alpha",
            "mixed frequency",
            "mixed-frequency",
        ],
        [
            "cannot be reliably characterized",
            "not established",
        ],
    )

    hz_match = None
    for item in all_items:
        match = re.search(
            r"\b(0?\d{1,2})\s*(?:[-–]|to)\s*(\d{1,2})\s*hz\b",
            item,
            flags=re.I,
        )
        if match:
            hz_match = f"{match.group(1)}-{match.group(2)} Hz"
            break

    if alpha_present:

        if hz_match:
            background = (
                f"Background activity is predominantly alpha activity at "
                f"approximately {hz_match}"
            )
        else:
            background = (
                "Background activity is predominantly alpha activity"
            )

        if symmetric_present and synchronous_present:
            background += ", appearing generally symmetric and synchronous"
        elif symmetric_present:
            background += ", appearing generally symmetric"
        elif synchronous_present:
            background += ", appearing generally synchronous"

        if reactivity_positive:
            background += " and reactive to eye opening and closing"

        background = sentence(background)

    elif explicit_background:

        background = sentence(explicit_background)

    elif beta_present or theta_present or delta_present:

        background = (
            "Mixed-frequency background activity is visible."
        )

    else:

        background = (
            "Background activity cannot be reliably characterized "
            "from the supplied image."
        )

    # ========================================================
    # FINDING 2 — ACTIVATION RESPONSE
    # ========================================================

    eye_status = text(
        report.get(
            "eye_reactivity",
            "not_established",
        )
    ).lower()

    hv_status = text(
        report.get(
            "hyperventilation_response",
            "not_established",
        )
    ).lower()

    photic_status = text(
        report.get(
            "photic_response",
            "not_established",
        )
    ).lower()

    # Eye reactivity text
    eye_text = ""

    if eye_status == "reactive":
        eye_text = (
            "Background activity is reactive to eye opening and closing."
        )

    elif eye_status == "nonreactive":
        eye_text = (
            "Background reactivity to eye opening and closing is not demonstrated."
        )

    # Hyperventilation + photic text
    activation_parts = []

    if hv_status == "no_additional_abnormality":
        activation_parts.append(
            "Hyperventilation did not produce any additional abnormality."
        )

    elif hv_status == "abnormal_response":
        activation_parts.append(
            "An abnormal response to hyperventilation is demonstrated "
            "in the supplied recording."
        )

    if photic_status == "no_additional_abnormality":
        activation_parts.append(
            "Photic stimulation did not produce any additional abnormality."
        )

    elif photic_status == "abnormal_response":
        activation_parts.append(
            "An abnormal response to photic stimulation is demonstrated "
            "in the supplied recording."
        )

    # Build final activation finding
    if eye_text and activation_parts:
        activation = (
            eye_text
            + " "
            + " ".join(activation_parts)
        )

    elif activation_parts:
        activation = " ".join(
            activation_parts
        )

    elif eye_text:
        activation = (
            eye_text
            + " Hyperventilation and photic stimulation responses are "
            "not established from the supplied image."
        )

    else:
        activation = (
            "Activation responses are not established from the supplied image."
        )

    # ========================================================
    # FINDING 3 — ABNORMAL SLOWING
    # ========================================================

    no_abnormal_slowing = combined_has([
        "no abnormal slowing",
        "no definite abnormal slowing",
        "without definite abnormal slowing",
        "abnormal slowing is not identified",
    ])

    definite_abnormal_slowing = combined_has([
        "definite abnormal slowing",
        "clear focal slowing",
        "clear generalized slowing",
        "focal slowing is present",
        "generalized slowing is present",
    ])

    if no_abnormal_slowing:
        slowing = (
            "No definite abnormal slowing is identified in the displayed segment."
        )

    elif definite_abnormal_slowing:
        slowing = first_matching(
            all_items,
            [
                "definite abnormal slowing",
                "clear focal slowing",
                "clear generalized slowing",
                "focal slowing",
                "generalized slowing",
            ],
        ) or (
            "Abnormal slowing is identified in the displayed segment."
        )

    else:
        slowing = (
            "Abnormal slowing is not reliably established from the supplied image."
        )


    # ========================================================
    # FINDING 4 — ARTIFACT / SHARP MORPHOLOGY
    # ========================================================

    explicit_no_artifact = combined_has([
        "no eye blink",
        "no eye-blink",
        "no movement artifact",
        "no movement artefact",
        "no eye blink and movement",
        "no eye-blink and movement",
        "no artifact identified",
        "no artefact identified",
    ])

    artifact = first_matching(
        all_items,
        [
            "artifact",
            "artefact",
            "eye blink",
            "eye-blink",
            "movement",
            "muscle",
            "myogenic",
            "electrode",
            "baseline shift",
            "broad slow transient",
            "broad slow transients",
            "broad slow deflection",
            "broad slow deflections",
            "slow transient",
            "slow transients",
            "slow deflection",
            "slow deflections",
        ],
    )

    frontal_polar_metadata = any(
        marker in metadata_text
        for marker in [
            "fp1",
            "fp2",
            "fp1-",
            "fp2-",
            "fp1 –",
            "fp2 –",
        ]
    )

    sharp_finding = first_matching(
        all_items,
        [
            "sharp-contoured",
            "sharp contoured",
            "sharply contoured",
            "sharp transient",
            "spike",
            "epileptiform",
        ],
    )

    definite_epileptiform = False

    if sharp_finding:

        sharp_low = sharp_finding.lower()

        definite_epileptiform = any(
            term in sharp_low
            for term in [
                "definite epileptiform",
                "clear epileptiform",
                "unequivocal epileptiform",
                "epileptiform discharge is present",
                "electrographic seizure",
            ]
        )

    if explicit_no_artifact:
        fourth = (
            "No definite eye-blink or movement artifact is identified "
            "in the displayed segment."
        )

    elif artifact:

        artifact_low = artifact.lower()

        broad_slow_artifact = any(
            phrase in artifact_low
            for phrase in [
                "broad slow transient",
                "broad slow transients",
                "broad slow deflection",
                "broad slow deflections",
                "slow transient",
                "slow transients",
                "slow deflection",
                "slow deflections",
            ]
        )

        explicit_eye_or_movement = any(
            phrase in artifact_low
            for phrase in [
                "eye blink",
                "eye-blink",
                "movement artifact",
                "movement artefact",
            ]
        )

        if explicit_eye_or_movement:
            fourth = sentence(artifact)

        elif broad_slow_artifact and frontal_polar_metadata:
            fourth = (
                "Broad slow transients are visible in frontal-polar "
                "derivations, compatible with eye-blink or movement artifact."
            )

        elif broad_slow_artifact:
            fourth = (
                "Broad slow artifact-like transients are visible; the exact "
                "artifact type is not established from the supplied image."
            )

        elif any(
            phrase in artifact_low
            for phrase in [
                "possibly",
                "possible",
                "may represent",
                "could represent",
                "not established",
                "cannot be established",
            ]
        ):
            fourth = (
                "Artifact-like activity is visible, but a specific artifact "
                "classification is not established from the supplied image."
            )

        else:
            fourth = sentence(artifact)

    elif sharp_finding:

        if definite_epileptiform:
            fourth = sentence(sharp_finding)
        else:
            fourth = (
                "Sharply contoured transients are visible, but epileptiform "
                "significance is not established from the supplied image."
            )

    else:
        fourth = (
            "No specific artifact or additional abnormal morphology "
            "is established from the supplied image."
        )


    # ========================================================
    # EXACTLY FOUR FINDINGS
    # ========================================================

    # Prefer concise doctor-style artifact wording when the
    # image-supported artifact classification is explicit.
    fourth_low = fourth.lower()

    eye_artifact_supported = any(
        term in fourth_low
        for term in [
            "eye blink",
            "eye-blink",
            "eye blinks",
            "eye-blinks",
            "ocular",
        ]
    )

    movement_artifact_supported = any(
        term in fourth_low
        for term in [
            "movement artifact",
            "movement artefact",
            "movement-related",
            "related to movement",
            "eye-blinks or movement",
            "eye-blink or movement",
        ]
    )

    if eye_artifact_supported and movement_artifact_supported:
        fourth = (
            "Eye-blink and movement artifact is present in the displayed segment."
        )
    elif eye_artifact_supported:
        fourth = (
            "Eye-blink artifact is present in the displayed segment."
        )
    elif movement_artifact_supported:
        fourth = (
            "Movement artifact is present in the displayed segment."
        )

    findings = unique_findings([
        background,
        activation,
        slowing,
        fourth,
    ])

    category_defaults = [
        background,
        activation,
        slowing,
        fourth,
    ]

    while len(findings) < 4:
        for candidate in category_defaults:
            candidate = sentence(candidate)
            if candidate.lower() not in {
                item.lower()
                for item in findings
            }:
                findings.append(candidate)

            if len(findings) >= 4:
                break

        if len(findings) >= 4:
            break

    findings = findings[:4]


    # ========================================================
    # IMPRESSION — WEIGHTED WHOLE-REPORT LOGIC
    # ========================================================

    combined_findings = " ".join(findings).lower()

    # Determine definite abnormality finding-by-finding so that phrases such as
    # "does not appear to represent definite abnormal slowing" cannot be
    # misread as positive abnormality merely because they contain the words
    # "definite abnormal slowing".
    def is_definite_abnormal_finding(item: str) -> bool:

        low = item.lower()

        negation_markers = [
            "no definite",
            "no abnormal",
            "not identified",
            "not established",
            "cannot be established",
            "not reliably established",
            "does not appear to represent",
            "does not represent",
            "without definite",
            "without evidence of",
            "epileptiform significance is not established",
            "compatible with artifact",
            "compatible with eye-blink",
            "compatible with eye blink",
            "movement artifact",
            "movement artefact",
            "artifactual activity",
            "artifact-like",
        ]

        if any(
            marker in low
            for marker in negation_markers
        ):
            return False

        definite_abnormal_markers = [
            "definite epileptiform",
            "clear epileptiform",
            "unequivocal epileptiform",
            "epileptiform discharge is present",
            "electrographic seizure",
            "definite abnormal slowing",
            "clear focal slowing",
            "clear generalized slowing",
            "periodic discharges",
        ]

        return any(
            marker in low
            for marker in definite_abnormal_markers
        )

    strong_abnormal_support = any(
        is_definite_abnormal_finding(item)
        for item in findings
    )

    cautious_sharp_only = (
        any(
            term in combined_findings
            for term in [
                "sharply contoured",
                "sharp transient",
                "epileptiform significance is not established",
                "artifact-like activity",
            ]
        )
        and not strong_abnormal_support
    )

    reassuring_support = any(
        phrase in combined_findings
        for phrase in [
            "no definite abnormal slowing",
            "no abnormal slowing",
            "abnormal slowing is not identified",
            "does not appear to represent definite abnormal slowing",
            "does not represent definite abnormal slowing",
            "no definite abnormality",
        ]
    )

    background_supportive = (
        "alpha activity" in background.lower()
        or "background activity is predominantly" in background.lower()
        or "mixed-frequency background activity is visible" in background.lower()
    )

    # Important: a cautious sharp-transient statement alone does NOT
    # make the displayed segment abnormal.
    if strong_abnormal_support:

        chosen_impression = (
            "The displayed EEG segment is abnormal, showing "
            "image-supported abnormal findings."
        )

    elif reassuring_support and background_supportive:

        chosen_impression = (
            "No definite abnormality is identified in the supplied complete EEG study."
            if complete_study_available
            else "No definite abnormality is identified in the displayed EEG segment."
        )

    elif cautious_sharp_only and background_supportive:

        chosen_impression = (
            "No definite abnormality is identified in the supplied complete EEG study."
            if complete_study_available
            else "No definite abnormality is identified in the displayed EEG segment."
        )

    elif (
        "cannot be reliably characterized" in background.lower()
        and "not reliably established" in slowing.lower()
    ):

        chosen_impression = (
            "The supplied EEG image is indeterminate for "
            "normal-versus-abnormal classification."
        )

    else:

        chosen_impression = (
            "No definite abnormality is identified in the supplied complete EEG study."
            if complete_study_available
            else "No definite abnormality is identified in the displayed EEG segment."
        )


    # ========================================================
    # FINAL REPORT
    # ========================================================

    lines = [
        "EEG REPORT",
        "",
        "Procedure:",
        procedure,
        "",
        "Findings:",
    ]

    for finding in findings:
        lines.append(
            f"• {sentence(finding)}"
        )

    scope_sentence = (
        "Correlation with clinical context and qualified EEG review is required "
        "for final interpretation."
        if complete_study_available
        else (
            "Correlation with the complete EEG recording and "
            "clinical context is required for final interpretation."
        )
    )

    lines.extend([
        "",
        "Impression:",
        sentence(chosen_impression),
        scope_sentence,
    ])

    return "\n".join(lines)


# ============================================================
# MIME TYPE
# ============================================================

def _mime_type(
    image_bytes: bytes,
) -> str:

    # PDF files begin with the %PDF signature.
    # Gemini can receive PDF bytes directly as application/pdf,
    # so no local PDF-to-image conversion is required.
    if image_bytes.lstrip().startswith(b"%PDF"):
        return "application/pdf"

    try:
        from PIL import Image

        image = Image.open(
            io.BytesIO(
                image_bytes
            )
        )

        fmt = (
            image.format
            or ""
        ).upper()

        mapping = {
            "PNG": "image/png",
            "JPEG": "image/jpeg",
            "JPG": "image/jpeg",
            "WEBP": "image/webp",
            "BMP": "image/bmp",
        }

        return mapping.get(
            fmt,
            "image/png",
        )

    except Exception:
        # Unknown non-PDF input: keep the previous conservative fallback.
        return "image/png"


# ============================================================
# GEMINI CLIENT
# ============================================================

def _gemini_client():

    try:
        from google import genai

        api_key = os.getenv(
            "GEMINI_API_KEY"
        )

        if not api_key:
            return None

        return genai.Client(
            api_key=api_key
        )

    except Exception as exc:
        print(
            f"Gemini client unavailable: {exc}"
        )

        return None


# ============================================================
# JSON PARSER
# ============================================================

def _parse_json_response(
    raw: str,
) -> Optional[
    Dict[str, Any]
]:

    if not raw:
        return None

    raw = raw.strip()

    try:
        parsed = json.loads(
            raw
        )

        if isinstance(
            parsed,
            dict,
        ):
            return parsed

    except json.JSONDecodeError:
        pass

    fenced = re.search(
        r"```(?:json)?\s*(\{.*\})\s*```",
        raw,
        flags=re.S | re.I,
    )

    if fenced:
        try:
            parsed = json.loads(
                fenced.group(1)
            )

            if isinstance(
                parsed,
                dict,
            ):
                return parsed

        except json.JSONDecodeError:
            pass

    start = raw.find(
        "{"
    )

    end = raw.rfind(
        "}"
    )

    if (
        start >= 0
        and end > start
    ):
        try:
            parsed = json.loads(
                raw[
                    start:
                    end + 1
                ]
            )

            if isinstance(
                parsed,
                dict,
            ):
                return parsed

        except json.JSONDecodeError:
            pass

    return None


# ============================================================
# GEMINI VISION
# ============================================================

def vision(
    image_bytes: bytes,
) -> Optional[
    Dict[str, Any]
]:

    try:
        client = _gemini_client()

        if client is None:
            return None

        from google.genai import types

        mime_type = _mime_type(
            image_bytes
        )

        image_part = (
            types.Part.from_bytes(
                data=image_bytes,
                mime_type=mime_type,
            )
        )

        source_scope = (
            "MULTI-PAGE PDF EEG DOCUMENT"
            if mime_type == "application/pdf"
            else "SINGLE EEG IMAGE / SCREENSHOT"
        )

        prompt = """
Analyze the supplied EEG source.

SOURCE FORMAT:
__SOURCE_SCOPE__

If the source is a PDF:
- inspect ALL supplied pages, not only the first page
- combine directly visible EEG observations across the document
- look for sequential recording pages, eye-opening/eye-closing sections,
  hyperventilation, photic stimulation, wake/sleep state, montage/electrode
  information, artifacts, slowing, and other explicitly displayed findings
- do not treat a PDF container by itself as proof that the full EEG study is
  present
- set complete_study_available=true ONLY when the supplied document itself
  clearly represents a complete EEG study or explicitly provides sufficient
  whole-study context
- otherwise set complete_study_available=false
- findings that occur on one page may be incorporated only when directly
  demonstrated on that page
- do not invent findings for pages or procedures that are absent

If the source is a single image/screenshot:
- set complete_study_available=false unless complete-study information is
  explicitly printed in that image
- restrict conclusions to the displayed segment

First determine whether the supplied source visibly contains EEG waveform traces.

Focus ONLY on visible information:

- whether it appears to contain EEG waveform traces
- whether another modality such as ECG is shown
- background organization
- relative slower activity
- relative faster activity
- waveform morphology
- sharp-contoured transients
- rhythmic activity
- periodic activity
- visible temporal evolution
- artifacts
- displayed time scale
- displayed amplitude calibration
- displayed montage
- visible channel labels
- other displayed technical information

For every technical item, only report it if it is actually visible.

IMPORTANT:

Do not diagnose epilepsy.

Do not diagnose a seizure disorder.

Do not diagnose a seizure from a static screenshot or from isolated PDF pages.

Do not infer localization.

Do not infer exact frequency without a visible time scale.

Do not infer exact amplitude without visible calibration.

Do not infer channel names that are not visible.

Do not infer montage that is not visible.

Do not infer unseen temporal evolution.

If sharp-contoured waveforms are visible, describe their morphology
without automatically calling them epileptiform.

Before reporting a sharp transient as clinically important, consider
alternative explanations visible in the supplied source, including:

- artifact
- electrode-related change
- movement
- muscle activity
- baseline shift
- benign transient
- normal sharply contoured activity
- image rendering artifact

Do not convert a sharp contour directly into an epileptiform finding.

If rhythmic activity is visible but convincing evolution is not visible,
describe the rhythmic morphology without calling it ictal.

For a CLINICAL EEG image, organize visible_findings into EXACTLY FOUR
concise observation categories whenever sufficient information is available.

The four categories should represent:

1. BACKGROUND ACTIVITY
   - dominant visible rhythm
   - symmetry
   - synchrony
   - organization
   - approximate dominant frequency ONLY when a visible time scale permits
     a reasonable measurement from clearly countable cycles
   - when a 1-second interval or equivalent visible time grid is available,
     count representative background cycles across more than one clean
     interval before estimating frequency
   - if frequency is estimated from the visible time scale, use cautious
     wording such as "approximately 8-9 Hz"; do not present it as an exact
     machine-measured value
   - do not estimate frequency from a single isolated sharp, slow, or
     artifact-contaminated deflection

2. ACTIVATION RESPONSE
   - eye opening / eye closing response
   - hyperventilation response
   - photic stimulation response
   - actively look for explicit event labels, annotations, markers, or a
     clearly demonstrated before/after waveform change associated with one
     of these activation procedures
   - report a response ONLY when the event itself and the corresponding
     waveform change are directly visible or explicitly labeled
   - do not infer eye reactivity merely because alpha activity is present
   - do not infer hyperventilation or photic stimulation merely because they
     are common parts of a routine EEG
   - otherwise state that activation responses are not established

3. ABNORMAL SLOWING
   - state whether definite abnormal slowing is visible, absent in the
     displayed segment, or not reliably established
   - do not call ordinary theta/delta content pathological solely because
     it is slow

4. ARTIFACT / OTHER TECHNICAL MORPHOLOGY
   - actively distinguish broad slow artifact-like deflections from genuinely
     sharp transients
   - if broad synchronous slow deflections are strongest in visibly labeled
     frontal-polar derivations such as Fp1/Fp2-containing channels, consider
     eye-blink or movement artifact as a cautious explanation
   - use wording such as "compatible with eye-blink or movement artifact"
     unless the artifact type is directly obvious
   - do not label a broad slow deflection as a sharp transient merely because
     it is prominent

Put exact visible channel labels, montage text, speed, sensitivity, filters,
time markers, and other acquisition text in displayed_metadata.

Do NOT infer that the international 10-20 electrode-placement standard was
used unless that standard is explicitly written in the supplied image.

If one category cannot be reliably evaluated, explicitly state:

"Not established from the supplied image."

Do NOT invent a positive or negative finding simply to fill a category.

============================================================
IMPRESSION RULE
============================================================

If complete_study_available=false, the impression applies ONLY to the
DISPLAYED EEG SEGMENT.

If complete_study_available=true, the impression may summarize the supplied
complete study, but only from findings actually demonstrated across the PDF.

Use one of these approaches:

1. If no definite abnormality is identified:

"No definite abnormality is identified in the displayed EEG segment."

2. If an abnormality is supported:

"The displayed EEG segment is abnormal, showing [supported abnormality]."

3. If the image is insufficient:

"The supplied EEG image is indeterminate for normal-versus-abnormal
classification."

Do NOT write:

"This EEG is normal."

"This EEG is abnormal."

"Normal EEG."

unless complete-study information has explicitly been supplied.

============================================================
ACTIVATION PROCEDURE RULE
============================================================

Do not claim:

- eye-opening reactivity
- eye-closing reactivity
- hyperventilation response
- photic stimulation response
- sleep stage
- drowsiness
- wakefulness

unless those events or labels are directly demonstrated in the supplied image.

Return JSON only using:

{
  "is_eeg": true,
  "image_type": "eeg|clinical_eeg|educational_eeg|ecg|other|uncertain",
  "is_clinical_recording": true,
  "image_classification": "clinical_eeg_recording|educational_eeg_reference|non_eeg|uncertain",
  "complete_study_available": false,
  "eye_reactivity": "reactive|nonreactive|not_established",
  "hyperventilation_response": "no_additional_abnormality|abnormal_response|not_established",
  "photic_response": "no_additional_abnormality|abnormal_response|not_established",
  "impression": [],
  "visible_findings": [],
  "pattern_interpretation": [],
  "pattern_characteristics": [],
  "frequency_assessment": [],
  "clinical_significance": [],
  "limitations": [],
  "diagnostic_safety": [],
  "displayed_metadata": [],
  "reason": [],
  "clinical_advice": [],
  "confidence": {
    "image_observation": "low|moderate|high",
    "technical_metadata": "low|moderate|high|not_available",
    "pattern_interpretation": "low|moderate|high",
    "clinical_interpretation": "low|moderate|high"
  }
}

For an educational/reference EEG chart:

- set image_type to "educational_eeg"
- set is_clinical_recording to false
- set image_classification to "educational_eeg_reference"
- populate pattern_characteristics with the number/type of stylized examples
  and their qualitative organization
- populate frequency_assessment only with frequency ranges explicitly printed
  in the image
- populate clinical_significance with educational/reference meaning rather
  than patient-specific meaning
- populate limitations with the lack of patient-specific recording context
- populate diagnostic_safety with a clear no-diagnosis statement
- do not leave these sections empty when the image visibly supports them
"""

        # Avoid using an f-string for this prompt because the JSON schema below
        # contains literal { } braces. Using f"""...""" makes Python interpret
        # those braces as formatting expressions and causes:
        # "Invalid format specifier ... for object of type 'str'".
        prompt = prompt.replace(
            "__SOURCE_SCOPE__",
            source_scope,
        )

        model = os.getenv(
            "GEMINI_MODEL",
            "gemini-2.5-flash",
        )

        # ----------------------------------------------------
        # GEMINI VISION REQUEST WITH TEMPORARY-503 RETRIES
        # ----------------------------------------------------
        #
        # Stage 1 should perform image observation only.
        # Doctor-report formatting belongs to the later reasoning
        # and report-building stages, so DOCTOR_REPORT_STRUCTURE
        # is intentionally NOT sent in this first Vision request.
        #
        # Retry only temporary service-capacity failures. Quota,
        # authentication, configuration, and other errors are
        # allowed to fall through to the existing outer handler.
        # ----------------------------------------------------

        response = None

        retry_delays = [
            2,
            4,
        ]

        max_attempts = 3

        for attempt in range(
            1,
            max_attempts + 1,
        ):

            try:

                response = (
                    client.models.generate_content(
                        model=model,
                        contents=[
                            SYSTEM,
                            prompt,
                            image_part,
                        ],
                        config={
                            "response_mime_type":
                                "application/json",
                            "temperature":
                                0.0,
                        },
                    )
                )

                break

            except Exception as exc:

                error_text = str(
                    exc
                ).lower()

                temporary_service_error = any(
                    marker in error_text
                    for marker in [
                        "503",
                        "unavailable",
                        "high demand",
                        "service unavailable",
                        "temporarily unavailable",
                    ]
                )

                if (
                    temporary_service_error
                    and attempt < max_attempts
                ):

                    delay = retry_delays[
                        attempt - 1
                    ]

                    print(
                        "Gemini Vision temporary service "
                        f"error on attempt {attempt}/{max_attempts}. "
                        f"Retrying in {delay} seconds..."
                    )

                    time.sleep(
                        delay
                    )

                    continue

                raise

        if response is None:
            return None

        raw = getattr(
            response,
            "text",
            None,
        )

        parsed = _parse_json_response(
            raw
            or ""
        )

        if not parsed:
            return None

        parsed = normalize_report(
            parsed
        )

        # Stage 1 Vision returns its own observations directly.
        # Stage 2 preservation is handled later by
        # merge_preserving_image_observations().
        return parsed

    except Exception as exc:

        error_text = str(
            exc
        ).lower()

        quota_errors = [
            "429",
            "resource_exhausted",
            "quota",
            "rate limit",
            "too many requests",
        ]

        if any(
            error in error_text
            for error in quota_errors
        ):
            print(
                "Gemini vision unavailable because "
                "the API quota/rate limit is exhausted."
            )

            return None

        print(
            f"Gemini vision unavailable: {exc}"
        )

        return None


# ============================================================
# RAG QUERY BUILDING
# ============================================================

def build_evidence_query(
    report: Dict[str, Any],
) -> str:

    parts = []

    for section in [
        "visible_findings",
        "pattern_interpretation",
        "displayed_metadata",
    ]:
        for value in report.get(
            section,
            [],
        ):
            value = text(
                value
            )

            if value:
                parts.append(
                    value
                )

    query = " ".join(
        parts
    ).strip()

    if not query:
        return (
            "EEG waveform screenshot "
            "interpretation limitations "
            "artifacts technical display "
            "quality clinical EEG review"
        )

    return query


def build_reporting_query(
    report: Dict[str, Any],
) -> str:

    parts = [
        (
            "EEG report findings impression "
            "uncertainty limitations"
        )
    ]

    for section in [
        "visible_findings",
        "pattern_interpretation",
    ]:
        for value in report.get(
            section,
            [],
        ):
            value = text(
                value
            )

            if value:
                parts.append(
                    value
                )

    return " ".join(
        parts
    )


# ============================================================
# RAG FORMATTER
# ============================================================

def format_rag_evidence(
    evidence: List[
        Dict[str, Any]
    ],
    max_items: int = 5,
) -> str:

    if not evidence:
        return (
            "No relevant RAG evidence was retrieved."
        )

    blocks = []

    for item in evidence[
        :max_items
    ]:

        source = text(
            item.get(
                "source"
            )
            or item.get(
                "file"
            )
            or item.get(
                "title"
            )
        )

        excerpt = text(
            item.get(
                "excerpt"
            )
            or item.get(
                "text"
            )
            or item.get(
                "content"
            )
        )

        category = text(
            item.get(
                "category"
            )
        )

        if not excerpt:
            continue

        block = []

        if category:
            block.append(
                f"Category: {category}"
            )

        if source:
            block.append(
                f"Reference: {source}"
            )

        block.append(
            f"Evidence: {excerpt}"
        )

        blocks.append(
            "\n".join(
                block
            )
        )

    if not blocks:
        return (
            "No usable RAG evidence was retrieved."
        )

    return "\n\n".join(
        blocks
    )


# ============================================================
# IMAGE-OBSERVATION PRESERVATION
# ============================================================

def merge_preserving_image_observations(
    image_report: Dict[str, Any],
    rag_report: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Merge second-stage RAG reasoning without allowing RAG to replace
    first-stage Gemini Vision observations.

    The first Gemini Vision result remains authoritative for anything
    that claims what is visible in the supplied EEG image.

    RAG may contribute interpretation, safety, limitations, explanatory
    reasoning, and confidence, but it cannot create, remove, or downgrade
    image observations.
    """

    image_report = dict(image_report or {})
    rag_report = dict(rag_report or {})

    # Begin with RAG so its non-observation fields remain available.
    merged: Dict[str, Any] = dict(rag_report)

    # These fields describe the image itself or structured observations.
    # Always preserve them from Stage 1, including empty/not_established
    # values. This prevents Stage 2 from creating a finding that Vision
    # did not support.
    locked_image_fields = [
        "is_eeg",
        "image_type",
        "is_clinical_recording",
        "image_classification",
        "complete_study_available",
        "visible_findings",
        "pattern_interpretation",
        "pattern_characteristics",
        "frequency_assessment",
        "displayed_metadata",
        "impression",
        "eye_reactivity",
        "hyperventilation_response",
        "photic_response",
    ]

    for key in locked_image_fields:
        if key in image_report:
            merged[key] = image_report[key]

    # RAG is allowed to enrich these non-observation sections.
    rag_enrichment_fields = [
        "clinical_significance",
        "limitations",
        "diagnostic_safety",
        "reason",
        "clinical_advice",
        "confidence",
    ]

    for key in rag_enrichment_fields:
        if key in rag_report:
            merged[key] = rag_report[key]
        elif key in image_report:
            merged[key] = image_report[key]

    return merged


# ============================================================
# SECOND-STAGE RAG REASONING
# ============================================================

def reason_with_rag(
    image_report: Dict[
        str,
        Any,
    ],
    eeg_evidence: List[
        Dict[str, Any]
    ],
    safety_evidence: List[
        Dict[str, Any]
    ],
    reporting_evidence: List[
        Dict[str, Any]
    ],
) -> Optional[
    Dict[str, Any]
]:

    try:
        client = _gemini_client()

        if client is None:
            return None

        observation_json = json.dumps(
            image_report,
            ensure_ascii=False,
            indent=2,
        )

        eeg_context = format_rag_evidence(
            eeg_evidence
        )

        safety_context = format_rag_evidence(
            safety_evidence
        )

        reporting_context = format_rag_evidence(
            reporting_evidence
        )

        reasoning_input = f"""
{REASONING_PROMPT}

============================================================
IMAGE OBSERVATIONS
============================================================

{observation_json}

============================================================
EEG REFERENCE EVIDENCE
============================================================

{eeg_context}

============================================================
SAFETY EVIDENCE
============================================================

{safety_context}

============================================================
REPORTING EVIDENCE
============================================================

{reporting_context}

============================================================
FINAL INSTRUCTIONS
============================================================

Now produce the final constrained EEG interpretation.

For a clinical EEG screenshot:

- visible_findings should contain exactly FOUR concise doctor-style findings.

The four findings must cover:

1. Background activity.
2. Activation response.
3. Abnormal slowing.
4. Artifact / other technical morphology.

BACKGROUND PRESERVATION:
If IMAGE OBSERVATIONS explicitly identify alpha activity, symmetry,
synchrony, or an approximate dominant frequency, preserve those supported
observations. Do not replace supported alpha activity with "mixed frequency"
or "dominant rhythm not established" unless the IMAGE OBSERVATIONS themselves
contain contradictory evidence.

ACTIVATION PRESERVATION:
Preserve the structured IMAGE OBSERVATION fields eye_reactivity,
hyperventilation_response, and photic_response when they contain supported
values. Do not downgrade a supported response to "not_established".
Do not create a positive activation response when the source field is
"not_established".

If a category cannot be reliably assessed, state that it is not
established from the supplied image.

Do not invent a positive or negative finding simply to fill a category.

The Impression must describe ONLY the displayed EEG segment.

Allowed conclusion styles:

"No definite abnormality is identified in the displayed EEG segment."

"The displayed EEG segment is abnormal, showing ..."

"The supplied EEG image is indeterminate for normal-versus-abnormal
classification."

Never produce a whole-study conclusion such as:

"This EEG is normal."

from a screenshot alone.

Do not claim:

- eye-opening reactivity
- eye-closing reactivity
- hyperventilation response
- photic stimulation response

unless directly supported by IMAGE OBSERVATIONS.

Remember:

The RAG evidence is NOT evidence that the image contains a finding.

Return JSON only using this schema:

{{
  "is_eeg": true,
  "image_type": "eeg|clinical_eeg|educational_eeg|ecg|other|uncertain",
  "is_clinical_recording": true,
  "image_classification": "clinical_eeg_recording|educational_eeg_reference|non_eeg|uncertain",
  "complete_study_available": false,
  "eye_reactivity": "reactive|nonreactive|not_established",
  "hyperventilation_response": "no_additional_abnormality|abnormal_response|not_established",
  "photic_response": "no_additional_abnormality|abnormal_response|not_established",
  "impression": [],
  "visible_findings": [],
  "pattern_interpretation": [],
  "pattern_characteristics": [],
  "frequency_assessment": [],
  "clinical_significance": [],
  "limitations": [],
  "diagnostic_safety": [],
  "displayed_metadata": [],
  "reason": [],
  "clinical_advice": [],
  "confidence": {{
    "image_observation": "low|moderate|high",
    "technical_metadata": "low|moderate|high|not_available",
    "pattern_interpretation": "low|moderate|high",
    "clinical_interpretation": "low|moderate|high"
  }}
}}

============================================================
DOCTOR-REPORT COMPONENT STRUCTURE
============================================================

{DOCTOR_REPORT_STRUCTURE}
"""

        model = os.getenv(
            "GEMINI_MODEL",
            "gemini-2.5-flash",
        )

        response = (
            client.models.generate_content(
                model=model,
                contents=[
                    SYSTEM,
                    reasoning_input,
                ],
                config={
                    "response_mime_type":
                        "application/json",
                    "temperature":
                        0.0,
                },
            )
        )

        raw = getattr(
            response,
            "text",
            None,
        )

        parsed = _parse_json_response(
            raw
            or ""
        )

        if not parsed:
            return None

        return normalize_report(
            parsed
        )

    except Exception as exc:
        print(
            f"RAG reasoning stage failed: {exc}"
        )

        return None


# ============================================================
# RAG CATEGORY
# ============================================================

def determine_rag_category(
    report: Dict[str, Any],
) -> str:

    combined = " ".join([
        *clean_list(
            report.get(
                "impression"
            )
        ),
        *clean_list(
            report.get(
                "visible_findings"
            )
        ),
        *clean_list(
            report.get(
                "pattern_interpretation"
            )
        ),
    ]).lower()

    if any(
        term in combined
        for term in [
            "artifact",
            "movement",
            "electrode",
            "muscle",
            "eye",
            "blink",
        ]
    ):
        return "artifacts"

    if any(
        term in combined
        for term in [
            "sharp",
            "spike",
            "transient",
            "epileptiform",
            "rhythmic",
            "periodic",
            "evolution",
            "sleep",
            "vertex",
            "spindle",
            "k-complex",
            "slowing",
            "slow activity",
            "slow wave",
            "delta slowing",
            "theta slowing",
        ]
    ):
        return "patterns"

    return "technical"


# ============================================================
# EEG IMAGE ANALYSIS PIPELINE — VISION ONLY
# ============================================================

def analyze_eeg_image(
    image_bytes: bytes,
) -> Dict[str, Any]:

    input_mime_type = _mime_type(
        image_bytes
    )

    input_is_pdf = (
        input_mime_type == "application/pdf"
    )

    # ========================================================
    # STAGE 1 — GEMINI VISION
    # ========================================================

    vision_report = vision(
        image_bytes
    )

    # ========================================================
    # VISION UNAVAILABLE
    # ========================================================

    if vision_report is None:

        confidence = (
            normalize_confidence(
                None
            )
        )

        formatted_report = "\n".join([
            "EEG REPORT",
            "",
            "Procedure:",
            (
                "EEG image analysis could not "
                "be completed."
            ),
            "",
            "Findings:",
            (
                "• No image-based EEG findings are available "
                "because the vision analysis was unavailable."
            ),
            "",
            "Impression:",
            (
                "No EEG interpretation was performed because "
                "the vision model was unavailable."
            ),
        ])

        return {
            "image_type":
                "uncertain",

            "is_eeg":
                False,

            "is_clinical_recording":
                False,

            "image_classification":
                "uncertain",

            "mode":
                "vision_unavailable",

            "reasoning_mode":
                "vision_unavailable",

            "clinical_review_status":
                "VISION_UNAVAILABLE",

            "complete_study_available":
                False,

            "confidence":
                build_confidence_display(
                    confidence
                ),

            "rag_category":
                "none",

            "impression": [
                (
                    "No EEG interpretation was performed because "
                    "the vision model was unavailable."
                )
            ],

            "visible_findings":
                [],

            "pattern_interpretation":
                [],

            "pattern_characteristics":
                [],

            "frequency_assessment":
                [],

            "clinical_significance":
                [],

            "limitations": [
                (
                    "The image-analysis stage did not return "
                    "a usable vision result."
                )
            ],

            "diagnostic_safety": [
                (
                    "No clinical EEG conclusion should be drawn "
                    "from this failed analysis."
                )
            ],

            "confidence_details":
                confidence,

            "confidence_display":
                build_confidence_display(
                    confidence
                ),

            "displayed_metadata":
                [],

            "reason": [
                (
                    "Gemini Vision did not return a usable "
                    "image analysis result."
                )
            ],

            "clinical_advice":
                [],

            "evidence":
                [],

            "safety_evidence":
                [],

            "reporting_evidence":
                [],

            "rag_query":
                "",

            "reporting_query":
                "",

            "formatted_report":
                formatted_report,

            "meta": {
                "rag_used":
                    False,

                "eeg_rag_used":
                    False,

                "safety_rag_used":
                    False,

                "reporting_rag_used":
                    False,

                "rag_category":
                    "none",

                "vision_available":
                    False,

                "rag_reasoning_used":
                    False,

                "physician_report_excludes_rag_sources":
                    True,

                "rag_image_observations_locked":
                    True,

                "hms_classifier_used":
                    False,

                "clinical_diagnosis_from_image":
                    False,

                "static_image_limitation":
                    True,
            },
        }

    observation_report = normalize_report(
        vision_report
    )

    mode = "pdf_vision" if input_is_pdf else "vision"

    # ========================================================
    # IMAGE TYPE VALIDATION
    # ========================================================

    is_eeg = observation_report.get(
        "is_eeg",
        True,
    )

    image_type = observation_report.get(
        "image_type",
        "eeg",
    )

    if (
        is_eeg is False
        or image_type in {
            "ecg",
            "other",
        }
    ):

        final_report = normalize_report(
            observation_report
        )

        formatted_report = build_clinical_report(
            final_report
        )

        return {
            "image_type":
                image_type,

            "is_eeg":
                False,

            "is_clinical_recording":
                final_report.get(
                    "is_clinical_recording",
                    False,
                ),

            "image_classification":
                final_report.get(
                    "image_classification",
                    "non_eeg",
                ),

            "mode":
                mode,

            "reasoning_mode":
                "image_rejected",

            "clinical_review_status":
                "IMAGE_REJECTED",

            "complete_study_available":
                False,

            "confidence":
                build_confidence_display(
                    final_report.get(
                        "confidence"
                    )
                ),

            "rag_category":
                "none",

            "impression":
                final_report.get(
                    "impression",
                    [],
                ),

            "visible_findings":
                final_report.get(
                    "visible_findings",
                    [],
                ),

            "pattern_interpretation":
                final_report.get(
                    "pattern_interpretation",
                    [],
                ),

            "pattern_characteristics":
                final_report.get(
                    "pattern_characteristics",
                    [],
                ),

            "frequency_assessment":
                final_report.get(
                    "frequency_assessment",
                    [],
                ),

            "clinical_significance":
                final_report.get(
                    "clinical_significance",
                    [],
                ),

            "limitations":
                final_report.get(
                    "limitations",
                    [],
                ),

            "diagnostic_safety":
                final_report.get(
                    "diagnostic_safety",
                    [],
                ),

            "confidence_details":
                final_report.get(
                    "confidence",
                    {},
                ),

            "confidence_display":
                build_confidence_display(
                    final_report.get(
                        "confidence"
                    )
                ),

            "displayed_metadata":
                final_report.get(
                    "displayed_metadata",
                    [],
                ),

            "reason":
                final_report.get(
                    "reason",
                    [],
                ),

            "clinical_advice":
                final_report.get(
                    "clinical_advice",
                    [],
                ),

            "evidence":
                [],

            "safety_evidence":
                [],

            "reporting_evidence":
                [],

            "rag_query":
                "",

            "reporting_query":
                "",

            "formatted_report":
                formatted_report,

            "meta": {
                "rag_used":
                    False,

                "eeg_rag_used":
                    False,

                "safety_rag_used":
                    False,

                "reporting_rag_used":
                    False,

                "rag_category":
                    "none",

                "vision_available":
                    True,

                "rag_reasoning_used":
                    False,

                "physician_report_excludes_rag_sources":
                    True,

                "rag_image_observations_locked":
                    True,

                "image_rejected_as_non_eeg":
                    True,

                "hms_classifier_used":
                    False,

                "clinical_diagnosis_from_image":
                    False,

                "static_image_limitation":
                    True,
            },
        }

    # ========================================================
    # BUILD RAG QUERIES
    # ========================================================

    rag_query = build_evidence_query(
        observation_report
    )

    reporting_query = build_reporting_query(
        observation_report
    )

    # ========================================================
    # EEG RAG
    # ========================================================

    try:
        eeg_result = (
            retrieve_eeg_evidence(
                rag_query,
                top_k=5,
            )
        )

        if isinstance(
            eeg_result,
            dict,
        ):
            eeg_evidence = (
                eeg_result.get(
                    "evidence",
                    [],
                )
                or []
            )
        else:
            eeg_evidence = (
                eeg_result
                or []
            )

    except Exception as exc:
        print(
            f"EEG RAG retrieval failed: {exc}"
        )

        eeg_evidence = []

    # ========================================================
    # SAFETY RAG
    # ========================================================

    try:
        safety_evidence = (
            retrieve_safety_evidence(
                rag_query,
                top_k=5,
            )
        )

    except Exception as exc:
        print(
            f"Safety RAG retrieval failed: {exc}"
        )

        safety_evidence = []

    # ========================================================
    # REPORTING RAG
    # ========================================================

    try:
        reporting_evidence = (
            retrieve_reporting_evidence(
                reporting_query,
                top_k=5,
            )
        )

    except Exception as exc:
        print(
            f"Reporting RAG retrieval failed: {exc}"
        )

        reporting_evidence = []

    # ========================================================
    # SECOND-STAGE GEMINI RAG REASONING
    # ========================================================

    rag_report = reason_with_rag(
        observation_report,
        eeg_evidence,
        safety_evidence,
        reporting_evidence,
    )

    reasoning_mode = (
        "vision_rag"
        if rag_report is not None
        else "vision_only"
    )

    # Stage 1 Gemini Vision remains authoritative for image observations.
    # Stage 2 RAG may enrich interpretation/safety/reporting context, but it
    # must not replace, create, remove, or downgrade visible EEG findings.
    if rag_report is not None:
        final_source = merge_preserving_image_observations(
            observation_report,
            rag_report,
        )
    else:
        final_source = observation_report

    final_report = normalize_report(
        final_source
    )

    # ========================================================
    # FINAL SAFETY NORMALIZATION
    # ========================================================

    final_report = normalize_report(
        final_report
    )

    confidence = final_report.get(
        "confidence",
        normalize_confidence(
            None
        ),
    )

    if (
        confidence.get(
            "clinical_interpretation"
        )
        == "high"
    ):
        confidence[
            "clinical_interpretation"
        ] = "moderate"

    final_report[
        "confidence"
    ] = confidence

    # ========================================================
    # RAG CATEGORY
    # ========================================================

    rag_category = determine_rag_category(
        final_report
    )

    # ========================================================
    # PHYSICIAN REPORT
    # ========================================================

    formatted_report = build_clinical_report(
        final_report
    )

    # ========================================================
    # API RESULT
    # ========================================================

    return {
        "image_type":
            final_report.get(
                "image_type",
                "clinical_eeg",
            ),

        "is_eeg":
            final_report.get(
                "is_eeg",
                True,
            ),

        "is_clinical_recording":
            final_report.get(
                "is_clinical_recording",
                True,
            ),

        "image_classification":
            final_report.get(
                "image_classification",
                "clinical_eeg_recording",
            ),

        "mode":
            mode,

        "reasoning_mode":
            reasoning_mode,

        "clinical_review_status":
            final_report.get(
                "clinical_review_status",
                "PRELIMINARY_IMAGE_REVIEW",
            ),

        "complete_study_available":
            final_report.get(
                "complete_study_available",
                False,
            ),

        "confidence":
            build_confidence_display(
                final_report.get(
                    "confidence"
                )
            ),

        "rag_category":
            rag_category,

        "impression":
            final_report.get(
                "impression",
                [],
            ),

        "visible_findings":
            final_report.get(
                "visible_findings",
                [],
            ),

        "pattern_interpretation":
            final_report.get(
                "pattern_interpretation",
                [],
            ),

        "pattern_characteristics":
            final_report.get(
                "pattern_characteristics",
                [],
            ),

        "frequency_assessment":
            final_report.get(
                "frequency_assessment",
                [],
            ),

        "clinical_significance":
            final_report.get(
                "clinical_significance",
                [],
            ),

        "limitations":
            final_report.get(
                "limitations",
                [],
            ),

        "diagnostic_safety":
            final_report.get(
                "diagnostic_safety",
                [],
            ),

        "confidence_details":
            final_report.get(
                "confidence",
                {},
            ),

        "confidence_display":
            build_confidence_display(
                final_report.get(
                    "confidence"
                )
            ),

        "displayed_metadata":
            final_report.get(
                "displayed_metadata",
                [],
            ),

        "reason":
            final_report.get(
                "reason",
                [],
            ),

        "clinical_advice":
            final_report.get(
                "clinical_advice",
                [],
            ),

        "evidence":
            eeg_evidence,

        "safety_evidence":
            safety_evidence,

        "reporting_evidence":
            reporting_evidence,

        "rag_query":
            rag_query,

        "reporting_query":
            reporting_query,

        "formatted_report":
            formatted_report,

        "meta": {
            "rag_used":
                bool(
                    eeg_evidence
                    or safety_evidence
                    or reporting_evidence
                ),

            "eeg_rag_used":
                bool(
                    eeg_evidence
                ),

            "safety_rag_used":
                bool(
                    safety_evidence
                ),

            "reporting_rag_used":
                bool(
                    reporting_evidence
                ),

            "rag_category":
                rag_category,

            "vision_available":
                True,

            "input_format":
                "pdf" if input_is_pdf else "image",

            "pdf_multi_page_analysis":
                input_is_pdf,

            "rag_reasoning_used":
                (
                    rag_report
                    is not None
                ),

            "physician_report_excludes_rag_sources":
                True,

            "hms_classifier_used":
                False,

            "clinical_diagnosis_from_image":
                False,

            "static_image_limitation":
                True,
        },
    }
