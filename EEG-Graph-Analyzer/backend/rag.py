# backend/rag.py
# EEG Clinical RAG v6.1
#
# Finding-aware retrieval for EEG image analysis.
#
# Design:
#   - TF-IDF remains the base retrieval signal
#   - EEG finding-aware positive matching
#   - Explicit negative/absence evidence handling
#   - Contradiction penalties
#   - EEG-specific routing
#   - Safety retrieval kept separate
#   - Reporting retrieval kept separate
#   - HMS dataset isolation
#   - Source diversity
#   - Frequency terminology separated from EEG slowing
#
# IMPORTANT:
#   This module retrieves knowledge-base evidence.
#   It does NOT diagnose a patient and does NOT infer findings
#   that are not present in the supplied image/query.

from pathlib import Path
from functools import lru_cache
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =====================================================================
# CONFIGURATION
# =====================================================================

KB = Path(__file__).resolve().parent / "knowledge_base"

CHUNK_SIZE = 1400
CHUNK_OVERLAP = 250

VALID_CATEGORIES = {
    "guardrails",
    "technical",
    "terminology",
    "patterns",
    "artifacts",
    "reporting",
    "clinical",
    "dataset",
}


# =====================================================================
# SOURCE AUTHORITY
# =====================================================================

SOURCE_AUTHORITY = {

    # Pattern sources
    "04_epilepsy_ictal_patterns.md": 0.18,
    "03_abnormal_eeg_patterns.md": 0.14,
    "05_rhythmic_and_periodic_patterns.md": 0.14,
    "06_sleep_patterns.md": 0.12,
    "01_normal_eeg_patterns.md": 0.12,
    "02_benign_variants.md": 0.10,

    "spikes_and_sharp_waves.md": 0.16,
    "epileptiform_activity.md": 0.18,
    "background_and_slowing.md": 0.16,
    "lpd_gpd_lrda_grda.md": 0.16,
    "electrographic_seizure.md": 0.18,
    "benign_variants.md": 0.10,
    "sleep.md": 0.12,

    # Terminology
    "01_ACNS_Standardized_EEG_Terminology_2021.md": 0.10,
    "acns_background.md": 0.10,
    "acns_rhythmic_periodic.md": 0.12,
    "acns_modifiers.md": 0.10,
    "acns_seizure_birds_iic.md": 0.12,
    "distribution_terms.md": 0.10,

    # Clinical
    "04_ILAEEpilepsy_Classification_2025.md": 0.08,
    "ilae_2025.md": 0.08,
    "epilepsy_vs_eeg_finding.md": 0.08,
    "clinical_context.md": 0.06,

    # Guardrails
    "05_RAG_Clinical_Guardrails.md": 0.10,
    "evidence_hierarchy.md": 0.10,
    "image_observability.md": 0.10,
    "no_invented_information.md": 0.10,
    "no_unsupported_diagnosis.md": 0.12,
    "seizure_claim_requirements.md": 0.12,
    "no_false_localization.md": 0.10,
    "clinical_safety.md": 0.12,
    "image_only_workflow.md": 0.12,

    # Artifacts
    "03_EEG_Artifact_and_Image_Limitations.md": 0.08,
    "05_artifacts.md": 0.08,
    "common_artifacts.md": 0.12,
    "eye_muscle_movement.md": 0.10,
    "electrode_cardiac_electrical.md": 0.10,
    "artifact_vs_cerebral_activity.md": 0.12,

    # Technical
    "02_ACNS_EEG_Recording_and_Display_Quality.md": 0.08,
    "ACNS_Guideline_1_Minimum_Technical_Requirements.md": 0.06,
    "recording_requirements.md": 0.08,
    "electrodes_and_10_20.md": 0.08,
    "montage.md": 0.08,
    "calibration_sensitivity.md": 0.08,
    "time_scale_frequency.md": 0.08,
    "frequency_bands.md": 0.10,
    "display_quality.md": 0.08,
    "filters_and_sampling.md": 0.08,

    # Reporting
    "eeg_report_structure.md": 0.10,
    "technical_description.md": 0.08,
    "findings.md": 0.10,
    "impression.md": 0.12,
    "uncertainty_language.md": 0.12,
    "limitations.md": 0.12,

    # Dataset
    "HMS_Dataset_Reference.md": 0.08,
}


# =====================================================================
# IMAGE TYPE ROUTING
# =====================================================================

IMAGE_TYPE_SOURCES = {

    "clinical_eeg_trace": {

        "background_and_slowing.md": 0.14,
        "spikes_and_sharp_waves.md": 0.16,
        "epileptiform_activity.md": 0.20,
        "lpd_gpd_lrda_grda.md": 0.16,
        "electrographic_seizure.md": 0.18,
        "benign_variants.md": 0.08,
        "sleep.md": 0.10,

        "acns_background.md": 0.10,
        "acns_rhythmic_periodic.md": 0.12,
        "acns_modifiers.md": 0.08,
        "acns_seizure_birds_iic.md": 0.12,
        "distribution_terms.md": 0.08,

        "01_normal_eeg_patterns.md": 0.10,
        "03_abnormal_eeg_patterns.md": 0.16,
        "04_epilepsy_ictal_patterns.md": 0.22,
        "05_rhythmic_and_periodic_patterns.md": 0.16,
        "06_sleep_patterns.md": 0.12,
        "01_ACNS_Standardized_EEG_Terminology_2021.md": 0.12,

        "05_RAG_Clinical_Guardrails.md": 0.08,
        "image_observability.md": 0.08,
        "no_unsupported_diagnosis.md": 0.08,
        "seizure_claim_requirements.md": 0.08,
        "03_EEG_Artifact_and_Image_Limitations.md": 0.06,
        "common_artifacts.md": 0.08,

        "recording_requirements.md": 0.04,
        "montage.md": 0.05,
        "calibration_sensitivity.md": 0.05,
        "time_scale_frequency.md": 0.06,
    },

    "educational_illustration": {

        "frequency_bands.md": 0.35,
        "background_and_slowing.md": 0.10,
        "01_normal_eeg_patterns.md": 0.20,
        "acns_background.md": 0.10,

        "01_ACNS_Standardized_EEG_Terminology_2021.md": 0.08,
        "image_observability.md": 0.08,
        "clinical_safety.md": 0.06,

        "03_abnormal_eeg_patterns.md": -0.12,
        "04_epilepsy_ictal_patterns.md": -0.14,
        "05_rhythmic_and_periodic_patterns.md": -0.12,
        "06_sleep_patterns.md": -0.08,

        "epileptiform_activity.md": -0.08,
        "electrographic_seizure.md": -0.10,
        "lpd_gpd_lrda_grda.md": -0.08,
    },

    "dataset_plot": {

        "HMS_Dataset_Reference.md": 0.35,

        "frequency_bands.md": 0.08,
        "01_normal_eeg_patterns.md": 0.08,
        "image_observability.md": 0.08,
        "clinical_safety.md": 0.08,

        "03_abnormal_eeg_patterns.md": -0.08,
        "04_epilepsy_ictal_patterns.md": -0.10,
        "05_rhythmic_and_periodic_patterns.md": -0.08,
        "06_sleep_patterns.md": -0.06,
    },

    "metadata_table": {

        "recording_requirements.md": 0.18,
        "electrodes_and_10_20.md": 0.15,
        "montage.md": 0.18,
        "calibration_sensitivity.md": 0.18,
        "time_scale_frequency.md": 0.16,
        "frequency_bands.md": 0.12,
        "display_quality.md": 0.14,
        "filters_and_sampling.md": 0.14,

        "02_ACNS_EEG_Recording_and_Display_Quality.md": 0.18,
        "ACNS_Guideline_1_Minimum_Technical_Requirements.md": 0.18,

        "electrographic_seizure.md": -0.08,
        "epileptiform_activity.md": -0.06,
    },

    "mixed_or_unknown": {

        "image_observability.md": 0.12,
        "clinical_safety.md": 0.10,
        "recording_requirements.md": 0.08,
        "display_quality.md": 0.08,

        "01_ACNS_Standardized_EEG_Terminology_2021.md": 0.08,
        "02_ACNS_EEG_Recording_and_Display_Quality.md": 0.10,
        "03_EEG_Artifact_and_Image_Limitations.md": 0.10,
    },
}


# =====================================================================
# CATEGORY-SPECIFIC SOURCE ROUTING
# =====================================================================

CATEGORY_SOURCES = {

    "normal": {
        "01_normal_eeg_patterns.md": 0.18,
        "acns_background.md": 0.10,
        "01_ACNS_Standardized_EEG_Terminology_2021.md": 0.06,
    },

    "slowing": {
        "background_and_slowing.md": 0.24,
        "03_abnormal_eeg_patterns.md": 0.20,
        "acns_background.md": 0.10,
        "lpd_gpd_lrda_grda.md": 0.04,
    },

    "epileptiform": {
        "epileptiform_activity.md": 0.26,
        "spikes_and_sharp_waves.md": 0.24,
        "electrographic_seizure.md": 0.18,
        "04_epilepsy_ictal_patterns.md": 0.30,
        "01_ACNS_Standardized_EEG_Terminology_2021.md": 0.12,
        "acns_seizure_birds_iic.md": 0.14,
        "02_benign_variants.md": 0.08,
        "benign_variants.md": 0.08,
        "03_abnormal_eeg_patterns.md": 0.05,
        "seizure_claim_requirements.md": 0.08,
        "no_unsupported_diagnosis.md": 0.05,
    },

    "rhythmic_periodic": {
        "lpd_gpd_lrda_grda.md": 0.28,
        "acns_rhythmic_periodic.md": 0.24,
        "05_rhythmic_and_periodic_patterns.md": 0.22,
        "01_ACNS_Standardized_EEG_Terminology_2021.md": 0.12,
        "electrographic_seizure.md": 0.08,
        "acns_seizure_birds_iic.md": 0.08,
    },

    "sleep": {
        "sleep.md": 0.24,
        "06_sleep_patterns.md": 0.20,
        "01_normal_eeg_patterns.md": 0.07,
    },

    "artifact": {
        "common_artifacts.md": 0.24,
        "artifact_vs_cerebral_activity.md": 0.22,
        "eye_muscle_movement.md": 0.18,
        "electrode_cardiac_electrical.md": 0.18,
        "05_artifacts.md": 0.20,
        "03_EEG_Artifact_and_Image_Limitations.md": 0.18,
        "02_ACNS_EEG_Recording_and_Display_Quality.md": 0.08,
    },

    "technical": {
        "recording_requirements.md": 0.18,
        "electrodes_and_10_20.md": 0.16,
        "montage.md": 0.18,
        "calibration_sensitivity.md": 0.18,
        "time_scale_frequency.md": 0.18,
        "frequency_bands.md": 0.18,
        "display_quality.md": 0.16,
        "filters_and_sampling.md": 0.14,
        "ACNS_Guideline_1_Minimum_Technical_Requirements.md": 0.18,
        "02_ACNS_EEG_Recording_and_Display_Quality.md": 0.16,
        "01_ACNS_Standardized_EEG_Terminology_2021.md": 0.06,
    },

    "clinical": {
        "ilae_2025.md": 0.16,
        "epilepsy_vs_eeg_finding.md": 0.16,
        "clinical_context.md": 0.12,
        "04_ILAEEpilepsy_Classification_2025.md": 0.16,
        "01_ACNS_Standardized_EEG_Terminology_2021.md": 0.10,
    },

    "reporting": {
        "eeg_report_structure.md": 0.20,
        "technical_description.md": 0.16,
        "findings.md": 0.20,
        "impression.md": 0.24,
        "uncertainty_language.md": 0.22,
        "limitations.md": 0.20,
    },
}


# =====================================================================
# FINDING GROUPS
# =====================================================================

FINDING_KEYWORDS = {

    "sharp": [
        "sharp",
        "sharp wave",
        "sharp waves",
        "sharp transient",
        "sharp transients",
        "spike",
        "spikes",
        "spike-and-wave",
        "spike wave",
        "polyspike",
        "polyspikes",
        "epileptiform",
        "interictal",
        "ied",
    ],

    # IMPORTANT:
    # Bare "delta" and "theta" are intentionally NOT included here.
    # They can describe frequency bands without indicating slowing.
    "slowing": [
        "slowing",
        "slow activity",
        "slow wave",
        "slow waves",
        "background slowing",
        "generalized slowing",
        "focal slowing",
        "diffuse slowing",
        "polymorphic slowing",
        "delta slowing",
        "theta slowing",
        "slow background",
    ],

    "rhythmic_periodic": [
        "rhythmic",
        "periodic",
        "rda",
        "lpd",
        "gpd",
        "lrda",
        "grda",
        "periodic discharges",
        "rhythmic delta",
        "triphasic",
        "triphasic waves",
    ],

    "seizure": [
        "seizure",
        "ictal",
        "electrographic seizure",
        "electroclinical seizure",
        "evolution",
        "evolving",
    ],

    "artifact": [
        "artifact",
        "artifacts",
        "blink",
        "eye movement",
        "muscle artifact",
        "movement artifact",
        "electrode artifact",
        "electrode pop",
        "ekg",
        "ecg",
        "electrical artifact",
    ],

    "sleep": [
        "sleep",
        "sleep spindle",
        "spindles",
        "vertex",
        "k complex",
        "k-complex",
        "nrem",
        "rem",
    ],

    "technical": [
        "channel",
        "channels",
        "montage",
        "bipolar",
        "referential",
        "reference",
        "electrode",
        "electrodes",
        "calibration",
        "sensitivity",
        "filter",
        "filters",
        "sampling rate",
        "time scale",
        "amplitude scale",
        "frequency scale",
        "microvolt",
        "microvolts",
        "µv",
    ],
}


# =====================================================================
# NEGATIVE / ABSENCE PATTERNS
# =====================================================================

NEGATIVE_PATTERNS = {

    "rhythmic_periodic": [
        r"\bno\s+(?:sustained\s+)?rhythmic",
        r"\bno\s+(?:clearly\s+)?periodic",
        r"\bwithout\s+(?:sustained\s+)?rhythmic",
        r"\bwithout\s+(?:clearly\s+)?periodic",
        r"\bnot\s+rhythmic",
        r"\bnot\s+periodic",
        r"\babsence\s+of\s+rhythmic",
        r"\babsence\s+of\s+periodic",
    ],

    "seizure": [
        r"\bno\s+(?:clear\s+)?seizure",
        r"\bno\s+ictal",
        r"\bwithout\s+(?:clear\s+)?seizure",
        r"\bwithout\s+ictal",
        r"\bnot\s+ictal",
        r"\bno\s+electrographic\s+seizure",
    ],

    "evolution": [
        r"\bno\s+clear\s+temporal\s+evolution",
        r"\bno\s+temporal\s+evolution",
        r"\bno\s+evolution",
        r"\bwithout\s+temporal\s+evolution",
        r"\bwithout\s+evolution",
        r"\bnot\s+evolving",
    ],

    "artifact": [
        r"\bno\s+artifact",
        r"\bwithout\s+artifact",
        r"\bartifact\s+not\s+seen",
    ],

    "sleep": [
        r"\bno\s+sleep",
        r"\bawake",
        r"\bnot\s+sleep",
    ],
}


# =====================================================================
# SOURCE FINDING AFFINITY
# =====================================================================

SOURCE_FINDING_AFFINITY = {

    # Sharp / epileptiform
    "spikes_and_sharp_waves.md": {
        "sharp": 0.24,
        "seizure": 0.03,
    },

    "epileptiform_activity.md": {
        "sharp": 0.24,
        "seizure": 0.08,
    },

    "04_epilepsy_ictal_patterns.md": {
        "sharp": 0.14,
        "seizure": 0.20,
    },

    # Slowing
    "background_and_slowing.md": {
        "slowing": 0.24,
    },

    "acns_background.md": {
        "slowing": 0.14,
    },

    "03_abnormal_eeg_patterns.md": {
        "slowing": 0.12,
        "sharp": 0.06,
    },

    # Rhythmic / periodic
    "lpd_gpd_lrda_grda.md": {
        "rhythmic_periodic": 0.26,
    },

    "acns_rhythmic_periodic.md": {
        "rhythmic_periodic": 0.20,
    },

    "05_rhythmic_and_periodic_patterns.md": {
        "rhythmic_periodic": 0.22,
    },

    # Seizure
    "electrographic_seizure.md": {
        "seizure": 0.22,
        "rhythmic_periodic": 0.04,
    },

    "acns_seizure_birds_iic.md": {
        "seizure": 0.16,
        "rhythmic_periodic": 0.08,
    },

    # Artifact
    "common_artifacts.md": {
        "artifact": 0.24,
    },

    "artifact_vs_cerebral_activity.md": {
        "artifact": 0.24,
    },

    "eye_muscle_movement.md": {
        "artifact": 0.20,
    },

    "electrode_cardiac_electrical.md": {
        "artifact": 0.20,
    },

    "05_artifacts.md": {
        "artifact": 0.20,
    },

    # Sleep
    "sleep.md": {
        "sleep": 0.24,
    },

    "06_sleep_patterns.md": {
        "sleep": 0.20,
    },

    # Technical
    "recording_requirements.md": {
        "technical": 0.20,
    },

    "electrodes_and_10_20.md": {
        "technical": 0.20,
    },

    "montage.md": {
        "technical": 0.20,
    },

    "calibration_sensitivity.md": {
        "technical": 0.22,
    },

    "time_scale_frequency.md": {
        "technical": 0.20,
    },

    "frequency_bands.md": {
        "technical": 0.18,
    },

    "display_quality.md": {
        "technical": 0.18,
    },

    "filters_and_sampling.md": {
        "technical": 0.18,
    },

    # Reporting
    "eeg_report_structure.md": {
        "technical": 0.04,
    },

    "technical_description.md": {
        "technical": 0.10,
    },

    "findings.md": {
        "sharp": 0.04,
        "slowing": 0.04,
        "artifact": 0.04,
    },

    "impression.md": {
        "sharp": 0.04,
        "slowing": 0.04,
        "seizure": 0.04,
    },

    "uncertainty_language.md": {
        "seizure": 0.04,
        "sharp": 0.03,
    },

    "limitations.md": {
        "technical": 0.06,
    },
}


# =====================================================================
# QUERY KEYWORDS
# =====================================================================

CATEGORY_KEYWORDS = {

    "epileptiform": FINDING_KEYWORDS["sharp"],

    "slowing": FINDING_KEYWORDS["slowing"],

    "rhythmic_periodic": FINDING_KEYWORDS["rhythmic_periodic"],

    "sleep": FINDING_KEYWORDS["sleep"],

    "artifact": FINDING_KEYWORDS["artifact"],

    "technical": FINDING_KEYWORDS["technical"],

    "normal": [
        "normal eeg",
        "normal background",
        "posterior dominant rhythm",
        "pdr",
        "alpha rhythm",
        "alpha activity",
        "mu rhythm",
        "beta activity",
        "normal waking",
        "normal awake",
        "symmetric background",
    ],

    "clinical": [
        "epilepsy",
        "epileptic",
        "encephalopathy",
        "clinical",
        "diagnosis",
        "classification",
        "seizure disorder",
    ],

    "reporting": [
        "report",
        "reporting",
        "impression",
        "findings",
        "finding",
        "how to write",
        "report structure",
        "report format",
        "uncertainty language",
        "limitations",
        "clinical impression",
        "eeg impression",
    ],

    "frequency": [
        "frequency band",
        "frequency bands",
        "delta band",
        "theta band",
        "alpha band",
        "beta band",
        "gamma band",
        "delta",
        "theta",
        "alpha",
        "beta",
        "gamma",
        "hz",
        "hertz",
    ],

    "dataset": [
        "hms dataset",
        "hms",
        "harmful brain activity",
        "expert consensus",
        "vote",
        "votes",
        "label",
        "labels",
        "seizure vote",
        "lpd vote",
        "gpd vote",
        "lrda vote",
        "grda vote",
        "dataset",
        "training dataset",
    ],
}


# =====================================================================
# TECHNICAL / GUARDRAIL SOURCES
# =====================================================================

TECHNICAL_SOURCES = {
    "ACNS_Guideline_1_Minimum_Technical_Requirements.md",
    "02_ACNS_EEG_Recording_and_Display_Quality.md",
    "recording_requirements.md",
    "electrodes_and_10_20.md",
    "montage.md",
    "calibration_sensitivity.md",
    "time_scale_frequency.md",
    "frequency_bands.md",
    "display_quality.md",
    "filters_and_sampling.md",
}


GUARDRAIL_SOURCES = {
    "05_RAG_Clinical_Guardrails.md",
    "evidence_hierarchy.md",
    "image_observability.md",
    "no_invented_information.md",
    "no_unsupported_diagnosis.md",
    "seizure_claim_requirements.md",
    "no_false_localization.md",
    "clinical_safety.md",
    "image_only_workflow.md",
}


# =====================================================================
# NORMALIZATION
# =====================================================================

def _normalize(text):
    return re.sub(
        r"\s+",
        " ",
        str(text or "")
    ).strip().lower()


def _source_name(path):
    return Path(path).name


# =====================================================================
# CATEGORY FROM FOLDER
# =====================================================================

def _category_from_path(path):

    try:
        relative = path.relative_to(KB)
    except ValueError:
        return "root"

    parent_parts = relative.parts[:-1]

    for part in reversed(parent_parts):

        part_lower = part.lower()

        if part_lower in VALID_CATEGORIES:
            return part_lower

    return "root"


# =====================================================================
# MARKDOWN PARSING
# =====================================================================

def _split_markdown(text):

    lines = text.splitlines()

    sections = []

    current_heading = "Document"
    current_lines = []

    for line in lines:

        stripped = line.strip()

        if re.match(
            r"^#{1,6}\s+",
            stripped
        ):

            if current_lines:

                sections.append(
                    (
                        current_heading,
                        "\n".join(
                            current_lines
                        ).strip()
                    )
                )

            current_heading = re.sub(
                r"^#{1,6}\s+",
                "",
                stripped
            )

            current_lines = []

        else:

            current_lines.append(line)

    if current_lines:

        sections.append(
            (
                current_heading,
                "\n".join(
                    current_lines
                ).strip()
            )
        )

    return sections


# =====================================================================
# CHUNKING
# =====================================================================

def _chunk_text(text):

    if len(text) <= CHUNK_SIZE:
        return [text]

    chunks = []

    start = 0

    while start < len(text):

        end = min(
            start + CHUNK_SIZE,
            len(text)
        )

        chunks.append(
            text[start:end]
        )

        if end >= len(text):
            break

        start = max(
            end - CHUNK_OVERLAP,
            start + 1
        )

    return chunks


# =====================================================================
# INDEX
# =====================================================================

@lru_cache(maxsize=1)
def index():

    documents = []

    if not KB.exists():

        return {
            "vectorizer": None,
            "matrix": None,
            "documents": [],
        }


    for path in sorted(
        KB.rglob("*")
    ):

        if not path.is_file():
            continue

        if path.suffix.lower() not in {
            ".md",
            ".txt",
        }:
            continue

        try:

            text = path.read_text(
                encoding="utf-8",
                errors="ignore",
            )

        except Exception:

            continue


        source = _source_name(path)

        source_category = _category_from_path(
            path
        )


        for section, section_text in _split_markdown(
            text
        ):

            section_text = section_text.strip()

            if not section_text:
                continue


            for chunk in _chunk_text(
                section_text
            ):

                chunk = chunk.strip()

                if len(chunk) < 30:
                    continue


                documents.append(
                    {
                        "source": source,
                        "category": source_category,
                        "section": section,
                        "text": chunk,
                    }
                )


    if not documents:

        return {
            "vectorizer": None,
            "matrix": None,
            "documents": [],
        }


    # Remove exact duplicate chunks.

    seen = set()

    unique_documents = []

    for doc in documents:

        key = (
            doc["source"],
            doc["category"],
            doc["section"],
            _normalize(
                doc["text"]
            ),
        )

        if key in seen:
            continue

        seen.add(key)

        unique_documents.append(doc)


    documents = unique_documents


    corpus = [
        doc["text"]
        for doc in documents
    ]


    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=20000,
        min_df=1,
        max_df=1.0,
        sublinear_tf=True,
    )


    try:

        matrix = vectorizer.fit_transform(
            corpus
        )

    except ValueError:

        vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 1),
            max_features=20000,
            min_df=1,
            max_df=1.0,
            sublinear_tf=True,
        )

        matrix = vectorizer.fit_transform(
            corpus
        )


    return {
        "vectorizer": vectorizer,
        "matrix": matrix,
        "documents": documents,
    }


# =====================================================================
# CATEGORY DETECTION
# =====================================================================

def detect_categories(query):

    q = _normalize(query)

    categories = set()

    # -------------------------------------------------------------
    # Explicit frequency context.
    #
    # This separates:
    #
    #   "delta/theta frequency bands"
    #
    # from:
    #
    #   "delta/theta slowing"
    # -------------------------------------------------------------

    frequency_context = any(
        phrase in q
        for phrase in [
            "frequency band",
            "frequency bands",
            "frequency range",
            "frequency ranges",
            "hz",
            "hertz",
            "oscillation frequency",
            "frequency content",
            "frequency components",
            "spectral",
            "spectrum",
            "band activity",
        ]
    )

    slowing_context = any(
        phrase in q
        for phrase in [
            "slowing",
            "slow activity",
            "slow wave",
            "slow waves",
            "background slowing",
            "generalized slowing",
            "focal slowing",
            "diffuse slowing",
            "polymorphic slowing",
            "delta slowing",
            "theta slowing",
            "slow background",
        ]
    )


    # -------------------------------------------------------------
    # Normal category matching.
    # -------------------------------------------------------------

    for category, keywords in CATEGORY_KEYWORDS.items():

        for keyword in keywords:

            if keyword not in q:
                continue


            # -----------------------------------------------------
            # Slowing special case.
            #
            # Bare delta/theta should not create a slowing category
            # when the query is clearly asking about frequency.
            # -----------------------------------------------------

            if category == "slowing":

                if keyword in {
                    "delta",
                    "theta",
                }:

                    if frequency_context and not slowing_context:
                        continue


            # -----------------------------------------------------
            # Frequency special case.
            #
            # If delta/theta occur explicitly in a slowing context,
            # they should not independently create frequency routing.
            # -----------------------------------------------------

            if category == "frequency":

                if keyword in {
                    "delta",
                    "theta",
                }:

                    if slowing_context and not frequency_context:
                        continue


            categories.add(
                category
            )

            break


    # -------------------------------------------------------------
    # Explicit frequency context always activates frequency.
    # -------------------------------------------------------------

    if frequency_context:

        categories.add(
            "frequency"
        )

        categories.add(
            "technical"
        )


    # -------------------------------------------------------------
    # Explicit slowing context always activates slowing.
    # -------------------------------------------------------------

    if slowing_context:

        categories.add(
            "slowing"
        )


    # -------------------------------------------------------------
    # Reporting routing.
    # -------------------------------------------------------------

    if _is_reporting_query(
        q,
        categories
    ):

        categories.add(
            "reporting"
        )


    # -------------------------------------------------------------
    # Dataset routing.
    # -------------------------------------------------------------

    if _is_dataset_query(q):

        categories.add(
            "dataset"
        )


    return list(categories)


# =====================================================================
# QUERY HELPERS
# =====================================================================

def _is_reporting_query(
    query,
    categories=None
):

    q = _normalize(query)

    if categories and "reporting" in categories:
        return True

    reporting_terms = [
        "report",
        "reporting",
        "impression",
        "findings",
        "finding",
        "how to write",
        "report structure",
        "report format",
        "uncertainty language",
        "limitations",
        "clinical impression",
        "eeg impression",
    ]

    return any(
        term in q
        for term in reporting_terms
    )


def _is_frequency_query(
    query,
    categories=None
):

    q = _normalize(query)

    if categories and "frequency" in categories:
        return True

    frequency_terms = [
        "frequency band",
        "frequency bands",
        "frequency range",
        "frequency ranges",
        "delta band",
        "theta band",
        "alpha band",
        "beta band",
        "gamma band",
        "frequency content",
        "frequency components",
        "hz",
        "hertz",
    ]

    return any(
        term in q
        for term in frequency_terms
    )


def _is_dataset_query(query):

    q = _normalize(query)

    dataset_terms = [
        "hms dataset",
        "hms",
        "harmful brain activity",
        "expert consensus",
        "seizure vote",
        "lpd vote",
        "gpd vote",
        "lrda vote",
        "grda vote",
        "training dataset",
    ]

    return any(
        term in q
        for term in dataset_terms
    )


def _is_high_risk_clinical_query(query):

    q = _normalize(query)

    high_risk_terms = [
        "seizure",
        "ictal",
        "epileptiform",
        "epilepsy",
        "diagnosis",
        "diagnose",
        "localize",
        "localization",
        "lpd",
        "gpd",
        "lrda",
        "grda",
        "periodic discharge",
        "electrographic seizure",
        "clinical impression",
    ]

    return any(
        term in q
        for term in high_risk_terms
    )


# =====================================================================
# NEGATIVE EVIDENCE DETECTION
# =====================================================================

def detect_negative_findings(query):

    q = _normalize(query)

    negative = set()

    for finding, patterns in NEGATIVE_PATTERNS.items():

        for pattern in patterns:

            if re.search(
                pattern,
                q,
                flags=re.IGNORECASE
            ):

                negative.add(
                    finding
                )

                break


    # "no evolution" is particularly important
    # for seizure retrieval.

    if "evolution" in negative:
        negative.add("seizure")


    return negative


# =====================================================================
# POSITIVE FINDING DETECTION
# =====================================================================

def detect_positive_findings(query):

    q = _normalize(query)

    positive = set()

    negative = detect_negative_findings(
        q
    )


    for finding, keywords in FINDING_KEYWORDS.items():

        for keyword in keywords:

            if keyword in q:

                positive.add(
                    finding
                )

                break


    # Explicit negative statements override
    # positive detection.

    positive -= negative


    return positive


# =====================================================================
# FINDING SCORE
# =====================================================================

def _finding_score(
    source,
    query,
):

    positive = detect_positive_findings(
        query
    )

    if not positive:
        return 0.0, []


    affinities = SOURCE_FINDING_AFFINITY.get(
        source,
        {}
    )


    score = 0.0
    matched = []


    for finding in positive:

        value = affinities.get(
            finding,
            0.0
        )

        if value > 0:

            score += value

            matched.append(
                finding
            )


    # Prevent finding routing from completely
    # overwhelming actual TF-IDF similarity.

    score = min(
        score,
        0.30
    )


    return score, matched


# =====================================================================
# CONTRADICTION PENALTY
# =====================================================================

def _contradiction_penalty(
    source,
    query,
):

    negative = detect_negative_findings(
        query
    )

    if not negative:
        return 0.0, []


    penalty = 0.0
    contradictions = []


    source_affinity = SOURCE_FINDING_AFFINITY.get(
        source,
        {}
    )


    for finding in negative:

        affinity = source_affinity.get(
            finding,
            0.0
        )

        if affinity <= 0:
            continue


        if finding == "rhythmic_periodic":

            value = 0.24 * (
                affinity / 0.26
            )

        elif finding == "seizure":

            value = 0.24 * (
                affinity / 0.22
            )

        elif finding == "evolution":

            value = 0.20 * (
                affinity / 0.22
            )

        else:

            value = 0.16 * (
                affinity / 0.24
            )


        value = min(
            value,
            0.28
        )


        penalty += value

        contradictions.append(
            finding
        )


    return min(
        penalty,
        0.35
    ), contradictions


# =====================================================================
# CATEGORY BONUS
# =====================================================================

def _category_bonus(
    source,
    categories
):

    bonus = 0.0

    matched_categories = []


    for category in categories:

        source_scores = CATEGORY_SOURCES.get(
            category,
            {}
        )

        if source in source_scores:

            bonus += source_scores[source]

            matched_categories.append(
                category
            )


    return bonus, matched_categories


# =====================================================================
# IMAGE TYPE BONUS
# =====================================================================

def _image_type_bonus(
    source,
    image_type
):

    mapping = IMAGE_TYPE_SOURCES.get(
        image_type,
        IMAGE_TYPE_SOURCES[
            "mixed_or_unknown"
        ],
    )


    return mapping.get(
        source,
        0.0
    )


# =====================================================================
# EVIDENCE ROLE
# =====================================================================

def _evidence_role(
    source,
    source_category,
    categories,
    image_type,
    negative_findings=None,
):

    negative_findings = (
        negative_findings or set()
    )


    if source_category == "reporting":

        if source == "impression.md":
            return "primary_reporting_evidence"

        if source == "findings.md":
            return "findings_reporting_evidence"

        if source == "uncertainty_language.md":
            return "uncertainty_reporting_evidence"

        if source == "limitations.md":
            return "limitations_reporting_evidence"

        return "reporting_structure_evidence"


    if source == "frequency_bands.md":

        return "frequency_band_reference"


    if source_category == "technical":

        return "technical_context"


    if (
        "rhythmic_periodic"
        in negative_findings
        and source in {
            "lpd_gpd_lrda_grda.md",
            "acns_rhythmic_periodic.md",
            "05_rhythmic_and_periodic_patterns.md",
        }
    ):

        return "cautionary_contrast_evidence"


    if (
        "seizure" in negative_findings
        and source in {
            "electrographic_seizure.md",
            "acns_seizure_birds_iic.md",
            "04_epilepsy_ictal_patterns.md",
        }
    ):

        return "seizure_guardrail_evidence"


    if "epileptiform" in categories:

        if source in {
            "epileptiform_activity.md",
            "spikes_and_sharp_waves.md",
            "04_epilepsy_ictal_patterns.md",
        }:

            return "primary_clinical_pattern_evidence"


        if source in {
            "acns_seizure_birds_iic.md",
            "01_ACNS_Standardized_EEG_Terminology_2021.md",
            "acns_rhythmic_periodic.md",
        }:

            return "terminology_support"


    if "rhythmic_periodic" in categories:

        if source in {
            "lpd_gpd_lrda_grda.md",
            "05_rhythmic_and_periodic_patterns.md",
        }:

            return "primary_rhythmic_periodic_evidence"


    if "slowing" in categories:

        if source in {
            "background_and_slowing.md",
            "03_abnormal_eeg_patterns.md",
        }:

            return "primary_slowing_evidence"


    if "artifact" in categories:

        if source in {
            "common_artifacts.md",
            "artifact_vs_cerebral_activity.md",
            "05_artifacts.md",
        }:

            return "primary_artifact_evidence"


        if source in {
            "eye_muscle_movement.md",
            "electrode_cardiac_electrical.md",
        }:

            return "artifact_support"


    if source in GUARDRAIL_SOURCES:

        return "safety_and_evidence_constraint"


    if source == "03_EEG_Artifact_and_Image_Limitations.md":

        return "image_limitation"


    if source_category == "dataset":

        return "dataset_context"


    if image_type in {
        "educational_illustration",
        "dataset_plot",
    }:

        return "educational_or_context_support"


    return "support"


# =====================================================================
# SOURCE RELEVANCE
# =====================================================================

def _is_relevant_source(
    source,
    source_category,
    image_type,
    categories,
    query,
):

    if source_category == "dataset":

        if not _is_dataset_query(query):

            return False


    if image_type == "educational_illustration":

        blocked = {
            "03_abnormal_eeg_patterns.md",
            "04_epilepsy_ictal_patterns.md",
            "05_rhythmic_and_periodic_patterns.md",
            "06_sleep_patterns.md",
            "04_ILAEEpilepsy_Classification_2025.md",
            "epileptiform_activity.md",
            "electrographic_seizure.md",
            "lpd_gpd_lrda_grda.md",
        }


        if (
            source in blocked
            and not categories.intersection(
                {
                    "epileptiform",
                    "rhythmic_periodic",
                    "sleep",
                    "clinical",
                }
            )
        ):

            return False


    if image_type == "dataset_plot":

        clinical_pattern_sources = {
            "04_epilepsy_ictal_patterns.md",
            "03_abnormal_eeg_patterns.md",
            "05_rhythmic_and_periodic_patterns.md",
            "04_ILAEEpilepsy_Classification_2025.md",
            "epileptiform_activity.md",
            "electrographic_seizure.md",
            "lpd_gpd_lrda_grda.md",
        }


        if (
            source in clinical_pattern_sources
            and "dataset" not in categories
        ):

            return False


    if _is_reporting_query(
        query,
        categories
    ):

        if source_category == "dataset":

            return False


    return True


# =====================================================================
# EXCERPT
# =====================================================================

def _best_excerpt(
    text,
    max_chars=700
):

    text = str(
        text or ""
    ).strip()


    if len(text) <= max_chars:

        return text


    shortened = text[
        :max_chars
    ]


    last_period = shortened.rfind(
        "."
    )


    if last_period > 300:

        return shortened[
            :last_period + 1
        ]


    return shortened + "..."


# =====================================================================
# RETRIEVE EVIDENCE
# =====================================================================

def retrieve_evidence(
    query,
    top_k=5,
    image_type="mixed_or_unknown",
):

    data = index()


    vectorizer = data[
        "vectorizer"
    ]

    matrix = data[
        "matrix"
    ]

    documents = data[
        "documents"
    ]


    if (
        not documents
        or vectorizer is None
        or matrix is None
    ):

        return []


    query = str(
        query or ""
    ).strip()


    if not query:

        return []


    categories = set(
        detect_categories(query)
    )


    positive_findings = (
        detect_positive_findings(
            query
        )
    )


    negative_findings = (
        detect_negative_findings(
            query
        )
    )


    query_vector = vectorizer.transform(
        [query]
    )


    similarities = cosine_similarity(
        query_vector,
        matrix,
    )[0]


    candidates = []


    reporting_query = _is_reporting_query(
        query,
        categories
    )


    frequency_query = _is_frequency_query(
        query,
        categories
    )


    dataset_query = _is_dataset_query(
        query
    )


    high_risk = _is_high_risk_clinical_query(
        query
    )


    # =================================================================
    # SCORE EACH DOCUMENT
    # =================================================================

    for idx, tfidf_score in enumerate(
        similarities
    ):

        doc = documents[idx]

        source = doc[
            "source"
        ]

        source_category = doc[
            "category"
        ]


        tfidf_score = float(
            tfidf_score
        )


        if tfidf_score <= 0.0:

            continue


        if not _is_relevant_source(
            source,
            source_category,
            image_type,
            categories,
            query,
        ):

            continue


        authority_bonus = SOURCE_AUTHORITY.get(
            source,
            0.0
        )


        image_bonus = _image_type_bonus(
            source,
            image_type,
        )


        category_bonus, matched_categories = (
            _category_bonus(
                source,
                categories,
            )
        )


        finding_bonus, matched_findings = (
            _finding_score(
                source,
                query,
            )
        )


        contradiction_penalty, contradictions = (
            _contradiction_penalty(
                source,
                query,
            )
        )


        final_score = (
            tfidf_score
            + authority_bonus
            + image_bonus
            + category_bonus
            + finding_bonus
            - contradiction_penalty
        )


        # =============================================================
        # REPORTING
        # =============================================================

        if reporting_query:

            if source_category == "reporting":

                final_score += 0.18

            elif source_category == "guardrails":

                final_score += 0.05

            elif source_category in {
                "patterns",
                "terminology",
                "clinical",
            }:

                final_score -= 0.04


            if source == "impression.md":

                final_score += 0.12

            elif source == "findings.md":

                final_score += 0.10

            elif source == "uncertainty_language.md":

                final_score += 0.09

            elif source == "limitations.md":

                final_score += 0.09

            elif source == "eeg_report_structure.md":

                final_score += 0.08

            elif source == "technical_description.md":

                final_score += 0.06


        # =============================================================
        # FREQUENCY
        # =============================================================

        if frequency_query:

            if source == "frequency_bands.md":

                final_score += 0.25

            elif source == "time_scale_frequency.md":

                final_score += 0.14

            elif source == "display_quality.md":

                final_score += 0.06

            elif source == "calibration_sensitivity.md":

                final_score += 0.05

            elif source_category == "technical":

                final_score += 0.08

            elif source == "background_and_slowing.md":

                # Frequency terminology should not automatically
                # become a slowing finding.
                final_score -= 0.12

            elif source == "acns_background.md":

                final_score -= 0.08

            elif source == "03_abnormal_eeg_patterns.md":

                final_score -= 0.08

            elif source_category == "dataset":

                final_score -= 0.15


        # =============================================================
        # DATASET
        # =============================================================

        if dataset_query:

            if source == "HMS_Dataset_Reference.md":

                final_score += 0.25

            elif source_category == "dataset":

                final_score += 0.15

            elif source_category in {
                "patterns",
                "clinical",
            }:

                final_score -= 0.03


        # =============================================================
        # EPILEPTIFORM
        # =============================================================

        if "epileptiform" in categories:

            if source in {
                "epileptiform_activity.md",
                "spikes_and_sharp_waves.md",
            }:

                final_score += 0.12


            elif source == "04_epilepsy_ictal_patterns.md":

                final_score += 0.20


            elif source == "electrographic_seizure.md":

                final_score += 0.08


            elif source in {
                "ACNS_Guideline_1_Minimum_Technical_Requirements.md",
                "recording_requirements.md",
            }:

                final_score -= 0.08


            elif source in {
                "02_ACNS_EEG_Recording_and_Display_Quality.md",
                "display_quality.md",
            }:

                final_score -= 0.03


        # =============================================================
        # SLOWING
        # =============================================================

        if "slowing" in categories:

            if source == "background_and_slowing.md":

                final_score += 0.16

            elif source == "03_abnormal_eeg_patterns.md":

                final_score += 0.10

            elif source == "acns_background.md":

                final_score += 0.08


        # =============================================================
        # RHYTHMIC / PERIODIC
        # =============================================================

        if "rhythmic_periodic" in categories:

            if source == "lpd_gpd_lrda_grda.md":

                final_score += 0.16

            elif source == "acns_rhythmic_periodic.md":

                final_score += 0.12

            elif source == "05_rhythmic_and_periodic_patterns.md":

                final_score += 0.10


        # =============================================================
        # ARTIFACT
        # =============================================================

        if "artifact" in categories:

            if source in {
                "common_artifacts.md",
                "artifact_vs_cerebral_activity.md",
                "05_artifacts.md",
            }:

                final_score += 0.16


            elif source in {
                "eye_muscle_movement.md",
                "electrode_cardiac_electrical.md",
            }:

                final_score += 0.10


            elif source in TECHNICAL_SOURCES:

                final_score -= 0.04


        # =============================================================
        # TECHNICAL
        # =============================================================

        if "technical" in categories:

            if source_category == "technical":

                final_score += 0.10


        # =============================================================
        # HIGH-RISK SAFETY
        # =============================================================

        if high_risk:

            if source_category == "guardrails":

                final_score += 0.05


            if source in {
                "seizure_claim_requirements.md",
                "no_unsupported_diagnosis.md",
                "no_false_localization.md",
                "image_observability.md",
                "clinical_safety.md",
            }:

                final_score += 0.04


        # =============================================================
        # NEGATIVE-EVIDENCE ROUTING
        # =============================================================

        if "rhythmic_periodic" in negative_findings:

            if source in {
                "lpd_gpd_lrda_grda.md",
                "acns_rhythmic_periodic.md",
                "05_rhythmic_and_periodic_patterns.md",
            }:

                final_score -= 0.10


        if "seizure" in negative_findings:

            if source in {
                "electrographic_seizure.md",
                "acns_seizure_birds_iic.md",
                "04_epilepsy_ictal_patterns.md",
            }:

                final_score -= 0.10


        if "evolution" in negative_findings:

            if source in {
                "electrographic_seizure.md",
                "04_epilepsy_ictal_patterns.md",
            }:

                final_score -= 0.08


        # =============================================================
        # ROOT DOCUMENT PENALTY
        # =============================================================

        if source_category == "root":

            final_score -= 0.05


        # =============================================================
        # FINAL DIAGNOSTIC INFORMATION
        # =============================================================

        evidence_role = _evidence_role(
            source,
            source_category,
            categories,
            image_type,
            negative_findings,
        )


        candidates.append(
            {
                "source": source,

                "category": source_category,

                "section": doc[
                    "section"
                ],

                "score": round(
                    final_score,
                    4
                ),

                "tfidf_score": round(
                    tfidf_score,
                    4
                ),

                "authority_bonus": round(
                    authority_bonus,
                    4
                ),

                "image_type_bonus": round(
                    image_bonus,
                    4
                ),

                "category_bonus": round(
                    category_bonus,
                    4
                ),

                "finding_bonus": round(
                    finding_bonus,
                    4
                ),

                "contradiction_penalty": round(
                    contradiction_penalty,
                    4
                ),

                "matched_categories": (
                    matched_categories
                ),

                "matched_findings": (
                    matched_findings
                ),

                "negative_findings": sorted(
                    negative_findings
                ),

                "contradictions": (
                    contradictions
                ),

                "evidence_role": evidence_role,

                "excerpt": _best_excerpt(
                    doc[
                        "text"
                    ]
                ),
            }
        )


    # =================================================================
    # SORT
    # =================================================================

    candidates.sort(
        key=lambda x: x["score"],
        reverse=True
    )


    # =================================================================
    # SOURCE DIVERSITY
    # =================================================================

    selected = []

    used_sources = set()


    for item in candidates:

        if len(selected) >= top_k:

            break


        source = item[
            "source"
        ]


        if source in used_sources:

            continue


        selected.append(
            item
        )

        used_sources.add(
            source
        )


    if len(selected) < top_k:

        for item in candidates:

            if len(selected) >= top_k:

                break


            if item in selected:

                continue


            selected.append(
                item
            )


    return selected[
        :top_k
    ]


# =====================================================================
# SAFETY RETRIEVAL
# =====================================================================

def retrieve_safety_evidence(
    query,
    top_k=5,
):

    data = index()


    vectorizer = data[
        "vectorizer"
    ]

    matrix = data[
        "matrix"
    ]

    documents = data[
        "documents"
    ]


    if (
        not documents
        or vectorizer is None
        or matrix is None
    ):

        return []


    query = str(
        query or ""
    ).strip()


    if not query:

        return []


    query_vector = vectorizer.transform(
        [query]
    )


    similarities = cosine_similarity(
        query_vector,
        matrix,
    )[0]


    results = []


    for idx, similarity in enumerate(
        similarities
    ):

        doc = documents[idx]

        source = doc[
            "source"
        ]

        category = doc[
            "category"
        ]


        if category != "guardrails":

            continue


        similarity = float(
            similarity
        )


        if similarity <= 0.0:

            continue


        score = (
            similarity
            + SOURCE_AUTHORITY.get(
                source,
                0.0
            )
        )


        results.append(
            {
                "source": source,

                "category": category,

                "section": doc[
                    "section"
                ],

                "score": round(
                    score,
                    4
                ),

                "tfidf_score": round(
                    similarity,
                    4
                ),

                "authority_bonus": round(
                    SOURCE_AUTHORITY.get(
                        source,
                        0.0
                    ),
                    4
                ),

                "image_type_bonus": 0.0,

                "category_bonus": 0.0,

                "finding_bonus": 0.0,

                "contradiction_penalty": 0.0,

                "matched_categories": [
                    "guardrails"
                ],

                "matched_findings": [],

                "negative_findings": [],

                "contradictions": [],

                "evidence_role":
                    "safety_and_evidence_constraint",

                "excerpt": _best_excerpt(
                    doc[
                        "text"
                    ]
                ),
            }
        )


    results.sort(
        key=lambda x: x[
            "score"
        ],
        reverse=True
    )


    return results[
        :top_k
    ]


# =====================================================================
# REPORTING RETRIEVAL
# =====================================================================

def retrieve_reporting_evidence(
    query,
    top_k=5,
):

    query = str(
        query or ""
    ).strip()


    if not query:

        return []


    return retrieve_evidence(
        query,
        top_k=top_k,
        image_type="clinical_eeg_trace",
    )


# =====================================================================
# EEG RETRIEVAL
# =====================================================================

def retrieve_eeg_evidence(
    query,
    top_k=5,
    image_type="clinical_eeg_trace",
):

    return retrieve_evidence(
        query=query,
        top_k=top_k,
        image_type=image_type,
    )


# =====================================================================
# INDEX MANAGEMENT
# =====================================================================

def clear_index():

    index.cache_clear()


def rebuild_index():

    clear_index()

    return index()


def get_index_info():

    data = index()

    documents = data[
        "documents"
    ]


    distribution = {}


    for doc in documents:

        category = doc.get(
            "category",
            "root"
        )


        distribution[
            category
        ] = (
            distribution.get(
                category,
                0
            ) + 1
        )


    return {
        "knowledge_base": str(
            KB
        ),

        "exists": KB.exists(),

        "indexed_chunks": len(
            documents
        ),

        "category_distribution":
            distribution,
    }


# =====================================================================
# CLI HELPERS
# =====================================================================

def _print_results(
    results
):

    if not results:

        print(
            "No evidence found."
        )

        return


    for i, result in enumerate(
        results,
        1
    ):

        print()

        print(
            f"{i}. "
            f"{result['source']}"
        )

        print(
            f"   Category : "
            f"{result.get('category')}"
        )

        print(
            f"   Section  : "
            f"{result['section']}"
        )

        print(
            f"   Role     : "
            f"{result['evidence_role']}"
        )

        print(
            f"   Score    : "
            f"{result['score']:.4f}"
        )

        print(
            f"   TF-IDF   : "
            f"{result['tfidf_score']:.4f}"
        )

        print(
            f"   Finding  : "
            f"{result.get('finding_bonus', 0.0):.4f}"
        )

        print(
            f"   Conflict : "
            f"{result.get('contradiction_penalty', 0.0):.4f}"
        )

        print(
            f"   Matched  : "
            f"{result.get('matched_findings', [])}"
        )

        print(
            f"   Excerpt  : "
            f"{result['excerpt']}"
        )


def _print_header(title):

    print()

    print(
        "=" * 80
    )

    print(title)

    print(
        "=" * 80
    )


# =====================================================================
# COMMAND-LINE TESTS
# =====================================================================

if __name__ == "__main__":

    print(
        "=" * 80
    )

    print(
        "EEG CLINICAL RAG v6.1"
    )

    print(
        "=" * 80
    )

    print(
        "Knowledge base:",
        KB
    )

    print(
        "Exists:",
        KB.exists()
    )


    info = get_index_info()


    print(
        "Indexed chunks:",
        info[
            "indexed_chunks"
        ]
    )


    print()

    print(
        "Category distribution:"
    )


    for category, count in sorted(
        info[
            "category_distribution"
        ].items()
    ):

        print(
            f"  {category}: {count}"
        )


    # ===============================================================
    # TEST 1
    # ===============================================================

    _print_header(
        "TEST 1 — SHARP / EPILEPTIFORM"
    )


    results = retrieve_evidence(
        (
            "sharp transients potentially "
            "epileptiform discharges"
        ),
        top_k=5,
        image_type="clinical_eeg_trace",
    )


    _print_results(
        results
    )


    # ===============================================================
    # TEST 2
    # ===============================================================

    _print_header(
        "TEST 2 — SLOWING"
    )


    results = retrieve_evidence(
        (
            "generalized slowing "
            "background slow activity"
        ),
        top_k=5,
        image_type="clinical_eeg_trace",
    )


    _print_results(
        results
    )


    # ===============================================================
    # TEST 3
    # ===============================================================

    _print_header(
        "TEST 3 — RHYTHMIC / PERIODIC"
    )


    results = retrieve_evidence(
        (
            "lateralized periodic discharges "
            "LPD and rhythmic periodic activity"
        ),
        top_k=5,
        image_type="clinical_eeg_trace",
    )


    _print_results(
        results
    )


    # ===============================================================
    # TEST 4
    # ===============================================================

    _print_header(
        "TEST 4 — ARTIFACT"
    )


    results = retrieve_evidence(
        (
            "EEG artifact caused by blink "
            "muscle movement or electrode artifact"
        ),
        top_k=5,
        image_type="clinical_eeg_trace",
    )


    _print_results(
        results
    )


    # ===============================================================
    # TEST 5
    # ===============================================================

    _print_header(
        "TEST 5 — TECHNICAL"
    )


    results = retrieve_evidence(
        (
            "EEG montage calibration sensitivity "
            "time scale and channel information"
        ),
        top_k=5,
        image_type="metadata_table",
    )


    _print_results(
        results
    )


    # ===============================================================
    # TEST 6
    # ===============================================================

    _print_header(
        "TEST 6 — REPORTING"
    )


    results = retrieve_reporting_evidence(
        (
            "how to write an EEG impression "
            "when image evidence is limited"
        ),
        top_k=5,
    )


    _print_results(
        results
    )


    # ===============================================================
    # TEST 7
    # ===============================================================

    _print_header(
        "TEST 7 — FREQUENCY BANDS"
    )


    results = retrieve_evidence(
        (
            "educational illustration of EEG "
            "frequency bands alpha beta theta delta"
        ),
        top_k=5,
        image_type="educational_illustration",
    )


    _print_results(
        results
    )


    # ===============================================================
    # TEST 8
    # ===============================================================

    _print_header(
        "TEST 8 — HMS DATASET"
    )


    results = retrieve_evidence(
        (
            "HMS dataset expert consensus "
            "seizure LPD GPD LRDA GRDA labels"
        ),
        top_k=5,
        image_type="dataset_plot",
    )


    _print_results(
        results
    )


    # ===============================================================
    # TEST 9
    # ===============================================================

    _print_header(
        "TEST 9 — SAFETY / GUARDRAILS"
    )


    results = retrieve_safety_evidence(
        (
            "image-only EEG analysis should not "
            "make unsupported diagnosis or seizure claim"
        ),
        top_k=5,
    )


    _print_results(
        results
    )


    # ===============================================================
    # TEST 10
    # ===============================================================

    _print_header(
        "TEST 10 — NEGATIVE EVIDENCE / FINDING-AWARE"
    )


    results = retrieve_evidence(
        (
            "EEG waveform with sharp-contoured "
            "transients and mixed background. "
            "No sustained rhythmic or clearly periodic "
            "activity is observed. "
            "No clear temporal evolution is visible. "
            "Possible epileptiform morphology."
        ),
        top_k=5,
        image_type="clinical_eeg_trace",
    )


    _print_results(
        results
    )


    # ===============================================================
    # TEST 11
    # ===============================================================

    _print_header(
        "TEST 11 — NO SEIZURE / NO EVOLUTION"
    )


    results = retrieve_evidence(
        (
            "sharp transients are visible, "
            "but there is no clear seizure, "
            "no ictal pattern, and no temporal evolution"
        ),
        top_k=5,
        image_type="clinical_eeg_trace",
    )


    _print_results(
        results
    )


    # ===============================================================
    # TEST 12
    # ===============================================================

    _print_header(
        "TEST 12 — FREQUENCY VS SLOWING"
    )


    print()
    print(
        "A. Frequency terminology query"
    )


    results = retrieve_evidence(
        (
            "EEG frequency bands "
            "alpha beta theta delta"
        ),
        top_k=5,
        image_type="educational_illustration",
    )


    _print_results(
        results
    )


    print()
    print(
        "B. Actual slowing finding query"
    )


    results = retrieve_evidence(
        (
            "prominent generalized delta "
            "and theta slowing in the EEG background"
        ),
        top_k=5,
        image_type="clinical_eeg_trace",
    )


    _print_results(
        results
    )


    # ===============================================================
    # TEST 13
    # ===============================================================

    _print_header(
        "TEST 13 — FREQUENCY ESTIMATION"
    )


    results = retrieve_evidence(
        (
            "estimate EEG frequency from "
            "visible time scale and waveform"
        ),
        top_k=5,
        image_type="clinical_eeg_trace",
    )


    _print_results(
        results
    )


    # ===============================================================
    # TEST 14
    # ===============================================================

    _print_header(
        "TEST 14 — INDEX DIAGNOSTICS"
    )


    print(
        "Knowledge base:",
        info[
            "knowledge_base"
        ]
    )


    print(
        "Exists:",
        info[
            "exists"
        ]
    )


    print(
        "Indexed chunks:",
        info[
            "indexed_chunks"
        ]
    )


    print(
        "Categories:"
    )


    for category, count in sorted(
        info[
            "category_distribution"
        ].items()
    ):

        print(
            f"  {category}: {count}"
        )


    print()

    print(
        "=" * 80
    )

    print(
        "RAG v6.1 TEST COMPLETE"
    )

    print(
        "=" * 80
    )