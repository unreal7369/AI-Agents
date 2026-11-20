
"""
Doc AI - Live Doctor/Patient Consultation Dashboard

Features:
- Continuous local microphone recording
- Whisper transcription through audio_engine.py
- Solo and two-person consultation modes
- Provisional Doctor/Patient role inference
- Manual role correction
- Editable and rejectable transcripts
- Manual transcript verification
- Live symptom extraction in TABLE format
- Symptom details and conflict warnings
- Verified patient symptom table
- Consultation JSON export

Requires:
    streamlit
    audio_engine.py
    clinical.py

For simulated consultations only.
Not a validated medical diagnostic system.
"""

import json
import queue
import re
from datetime import datetime

import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Doc AI - Live Consultation",
    page_icon="🩺",
    layout="wide",
)

st.title("🩺 Doc AI - Live Consultation")

st.warning(
    "Prototype for simulated consultations only. "
    "Transcription, speaker identification, and symptom "
    "extraction may be incorrect. "
    "Do not use for clinical decisions."
)


# ============================================================
# IMPORT PROJECT MODULES
# ============================================================

try:
    from audio_engine import LiveTranscriber

    from clinical import (
        analyze_live,
        aggregate,
        summarize_consultation,
    )

except Exception as exc:
    st.error("Unable to import project modules.")
    st.exception(exc)
    st.stop()


# ============================================================
# SESSION STATE
# ============================================================

DEFAULTS = {
    "lines": [],
    "status": "Idle",
    "warnings": [],
    "events": 0,
    "consent": False,
    "speaker1_role": "Unassigned",
    "speaker2_role": "Unassigned",
    "session_number": 0,
    "consultation_mode": "Solo testing (one voice)",
    "solo_role": "Patient",
    "show_rejected": True,
    "pending_speaker_updates": {},
    "last_role_signature": None,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

if "engine" not in st.session_state:
    try:
        st.session_state.engine = LiveTranscriber()
    except Exception as exc:
        st.error("Unable to initialize audio engine.")
        st.exception(exc)
        st.stop()

engine = st.session_state.engine


# ============================================================
# ROLE INFERENCE
# ============================================================

LEADING_FILLER_RE = re.compile(
    r"^\s*(?:(?:so|okay|ok|well|now|alright)"
    r"\s*[,.:]?\s*)+",
    re.I,
)

DOCTOR_PROMPT_RE = re.compile(
    r"^\s*(?:"
    r"what(?:'s| is)?\b|"
    r"how\b|"
    r"for how many\b|"
    r"do you\b|"
    r"did you\b|"
    r"have you\b|"
    r"are you\b|"
    r"can you\b|"
    r"could you\b|"
    r"tell me\b|"
    r"please tell me\b|"
    r"where\b|"
    r"when\b|"
    r"which\b|"
    r"is your\b|"
    r"does your\b|"
    r"does it\b|"
    r"would you\b|"
    r"since when\b|"
    r"any\b"
    r")",
    re.I,
)

PATIENT_STATEMENT_RE = re.compile(
    r"^\s*(?:"
    r"i\b|"
    r"my\b|"
    r"yes\b|"
    r"yeah\b|"
    r"yep\b|"
    r"no\b|"
    r"nope\b|"
    r"it's\b|"
    r"it is\b|"
    r"around\b|"
    r"about\b|"
    r"approximately\b"
    r")",
    re.I,
)

SHORT_ANSWER_RE = re.compile(
    r"^\s*(?:"
    r"yes|yeah|yep|no|nope|nah|"
    r"left|right|front|back|"
    r"left side|right side|"
    r"front side|back side"
    r")[.! ]*$",
    re.I,
)

NUMBER_ANSWER_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)?|"
    r"one|two|three|four|five|six|seven|eight|"
    r"nine|ten|eleven|twelve)"
    r"(?:\s+(?:days?|weeks?|months?|years?|"
    r"times?|episodes?|degrees?|"
    r"per day|a day))?"
    r"[.! ]*$",
    re.I,
)

ROLE_OPTIONS = [
    "Use speaker mapping",
    "Unverified",
    "Doctor",
    "Patient",
]

ROLE_NAMES = ("Doctor", "Patient")


def normalize_question(text):
    return LEADING_FILLER_RE.sub(
        "",
        str(text or ""),
    ).strip()


def infer_solo_roles(lines):
    """
    Provisional wording-based role inference.

    This is not acoustic speaker identification.
    Manual role correction takes precedence.
    """
    inferred = []

    for index, line in enumerate(lines):
        manual = line.get(
            "manual_role",
            "Use speaker mapping",
        )

        if manual in (
            "Doctor",
            "Patient",
            "Unverified",
        ):
            inferred.append(manual)
            continue

        text = normalize_question(
            line.get("text", "")
        )

        captured = line.get("role_at_capture")

        if not text:
            inferred.append(
                captured
                if captured in ROLE_NAMES
                else "Unassigned"
            )
            continue

        previous = (
            inferred[index - 1]
            if index > 0
            else None
        )

        if DOCTOR_PROMPT_RE.search(text):
            role = "Doctor"

        elif (
            previous == "Doctor"
            and (
                SHORT_ANSWER_RE.fullmatch(text)
                or NUMBER_ANSWER_RE.fullmatch(text)
                or len(text.split()) <= 8
            )
        ):
            role = "Patient"

        elif PATIENT_STATEMENT_RE.search(text):
            role = "Patient"

        elif text.endswith("?"):
            role = "Doctor"

        elif captured in ROLE_NAMES:
            role = captured

        else:
            role = st.session_state.solo_role

        inferred.append(role)

    return inferred


# ============================================================
# GENERAL HELPERS
# ============================================================

def widget_key(prefix, index):
    return (
        f"{prefix}_"
        f"{st.session_state.session_number}_"
        f"{index}"
    )


def get_role_mapping():
    return {
        "Speaker 1": st.session_state.speaker1_role,
        "Speaker 2": st.session_state.speaker2_role,
    }


def roles_conflict():
    role1 = st.session_state.speaker1_role
    role2 = st.session_state.speaker2_role

    return (
        role1 != "Unassigned"
        and role1 == role2
    )


def current_role_signature():
    return (
        st.session_state.consultation_mode,
        st.session_state.solo_role,
        st.session_state.speaker1_role,
        st.session_state.speaker2_role,
    )


def clear_transcript():
    st.session_state.lines = []
    st.session_state.warnings = []
    st.session_state.events = 0
    st.session_state.status = "Idle"
    st.session_state.pending_speaker_updates = {}
    st.session_state.session_number += 1
    st.session_state.last_role_signature = None

    while True:
        try:
            engine.events.get_nowait()
        except queue.Empty:
            break


def invalidate_verifications_for_mapping_change():
    signature = current_role_signature()
    previous = st.session_state.last_role_signature

    if previous is not None and signature != previous:
        for index, line in enumerate(
            st.session_state.lines
        ):
            if (
                line.get("manual_role")
                == "Use speaker mapping"
            ):
                line["verified"] = False

                key = widget_key("verify", index)

                if key in st.session_state:
                    st.session_state[key] = False

    st.session_state.last_role_signature = signature


def get_effective_role(
    line,
    mapping,
    index=None,
    inferred_roles=None,
):
    manual = line.get(
        "manual_role",
        "Use speaker mapping",
    )

    if manual != "Use speaker mapping":
        return manual

    if (
        st.session_state.consultation_mode
        == "Solo testing (one voice)"
    ):
        if inferred_roles is None:
            inferred_roles = infer_solo_roles(
                st.session_state.lines
            )

        if (
            index is not None
            and 0 <= index < len(inferred_roles)
        ):
            return inferred_roles[index]

        return "Unassigned"

    if roles_conflict():
        return "Unassigned"

    return mapping.get(
        line.get("speaker"),
        "Unassigned",
    )


# ============================================================
# TRANSCRIPT CALLBACKS
# ============================================================

def invalidate_line_verification(index):
    line = st.session_state.lines[index]
    line["verified"] = False

    key = widget_key("verify", index)

    if key in st.session_state:
        st.session_state[key] = False


def on_transcript_change(index):
    line = st.session_state.lines[index]
    key = widget_key("transcript", index)

    new_text = st.session_state[key]

    if new_text != line["text"]:
        line["text"] = new_text
        invalidate_line_verification(index)


def on_manual_role_change(index):
    line = st.session_state.lines[index]
    key = widget_key("manual", index)

    new_role = st.session_state[key]

    if new_role != line.get("manual_role"):
        line["manual_role"] = new_role
        invalidate_line_verification(index)


def on_reject_change(index):
    line = st.session_state.lines[index]
    key = widget_key("reject", index)

    rejected = bool(st.session_state[key])
    line["rejected"] = rejected

    if rejected:
        invalidate_line_verification(index)


def on_verify_change(index):
    line = st.session_state.lines[index]
    key = widget_key("verify", index)

    if line.get("rejected", False):
        line["verified"] = False
        st.session_state[key] = False
    else:
        line["verified"] = bool(
            st.session_state[key]
        )


# ============================================================
# CLINICAL INPUT
# ============================================================

def build_clinical_lines(mapping):
    lines = st.session_state.lines
    inferred = infer_solo_roles(lines)

    result = []

    for index, line in enumerate(lines):
        if line.get("rejected", False):
            continue

        text = str(
            line.get("text", "") or ""
        ).strip()

        if not text:
            continue

        role = get_effective_role(
            line,
            mapping,
            index,
            inferred,
        )

        result.append({
            "text": text,
            "role": role,
            "verified": bool(
                line.get("verified", False)
            ),
            "rejected": False,
        })

    return result


# ============================================================
# AUDIO EVENTS
# ============================================================

def receive_events():
    while True:
        try:
            kind, value = engine.events.get_nowait()
        except queue.Empty:
            break

        st.session_state.events += 1

        if kind == "transcript":
            if not isinstance(value, dict):
                continue

            text = str(
                value.get("text", "") or ""
            ).strip()

            if not text:
                continue

            turn_id = value.get("turn_id")

            if turn_id is not None:
                duplicate = any(
                    line.get("turn_id") == turn_id
                    for line in st.session_state.lines
                )

                if duplicate:
                    continue

            speaker = value.get(
                "speaker",
                "Unverified",
            )

            if speaker not in (
                "Speaker 1",
                "Speaker 2",
                "Unverified",
            ):
                speaker = "Unverified"

            speaker_note = str(
                value.get("speaker_note", "")
            )

            pending = (
                st.session_state.pending_speaker_updates.pop(
                    turn_id,
                    None,
                )
                if turn_id is not None
                else None
            )

            if pending:
                speaker = pending.get(
                    "speaker",
                    speaker,
                )
                speaker_note = str(
                    pending.get(
                        "speaker_note",
                        speaker_note,
                    )
                )

            st.session_state.lines.append({
                "turn_id": turn_id,
                "at": datetime.now().strftime(
                    "%H:%M:%S"
                ),
                "text": text,
                "original_text": text,
                "speaker": speaker,
                "speaker_note": speaker_note,
                "verified": False,
                "rejected": False,
                "manual_role": "Use speaker mapping",
                "role_at_capture": value.get(
                    "role_at_capture"
                ),
                "asr_seconds": value.get(
                    "asr_seconds"
                ),
            })

        elif kind == "speaker_update":
            if not isinstance(value, dict):
                continue

            turn_id = value.get("turn_id")

            if turn_id is None:
                continue

            found = False

            for index, line in enumerate(
                st.session_state.lines
            ):
                if line.get("turn_id") != turn_id:
                    continue

                found = True

                new_speaker = value.get(
                    "speaker",
                    "Unverified",
                )

                if new_speaker not in (
                    "Speaker 1",
                    "Speaker 2",
                    "Unverified",
                ):
                    new_speaker = "Unverified"

                changed = (
                    line.get("speaker") != new_speaker
                )

                line["speaker"] = new_speaker
                line["speaker_note"] = str(
                    value.get("speaker_note", "")
                )

                if (
                    changed
                    and line.get("manual_role")
                    == "Use speaker mapping"
                ):
                    invalidate_line_verification(index)

                break

            if not found:
                st.session_state.pending_speaker_updates[
                    turn_id
                ] = value

        elif kind in ("warning", "error"):
            message = f"{kind.upper()}: {value}"

            st.session_state.warnings.append(
                message
            )

            st.session_state.warnings = (
                st.session_state.warnings[-50:]
            )

            if kind == "error":
                st.session_state.status = str(value)

        elif kind == "status":
            st.session_state.status = str(value)


# ============================================================
# SYMPTOM TABLE HELPERS
# ============================================================

TABLE_COLUMNS = [
    "Symptom",
    "Status",
    "Location",
    "Duration",
    "Frequency",
    "Severity",
    "Temperature",
    "Quality",
    "Verified",
    "Needs Review",
]


def clean_table_value(value):
    """Keep table cells readable."""
    if value is None:
        return ""

    if isinstance(value, (list, tuple, dict)):
        return json.dumps(
            value,
            ensure_ascii=False,
            default=str,
        )

    return str(value)


def finding_table(findings):
    """
    Convert clinical.py findings to table rows.

    The original findings remain unchanged for JSON export.
    """
    rows = []

    for finding in findings:
        if not isinstance(finding, dict):
            continue

        rows.append({
            "Symptom": clean_table_value(
                finding.get("symptom", "")
            ).title(),

            "Status": clean_table_value(
                finding.get("status", "")
            ),

            "Location": clean_table_value(
                finding.get("location", "")
            ),

            "Duration": clean_table_value(
                finding.get("duration", "")
            ),

            "Frequency": clean_table_value(
                finding.get("frequency", "")
            ),

            "Severity": clean_table_value(
                finding.get("severity", "")
            ),

            "Temperature": clean_table_value(
                finding.get("temperature", "")
            ),

            "Quality": clean_table_value(
                finding.get("quality", "")
            ),

            "Verified": bool(
                finding.get("verified", False)
            ),

            "Needs Review": bool(
                finding.get("needs_review", False)
            ),
        })

    return rows


def render_symptom_table(findings, height=350):
    """
    Display all symptoms in a single Streamlit table.
    """
    rows = finding_table(findings)

    if not rows:
        st.info("No symptom findings available.")
        return

    st.dataframe(
        rows,
        hide_index=True,
        use_container_width=True,
        height=height,
        column_order=TABLE_COLUMNS,
        column_config={
            "Symptom": st.column_config.TextColumn(
                "Symptom",
                width="medium",
            ),
            "Status": st.column_config.TextColumn(
                "Status",
            ),
            "Location": st.column_config.TextColumn(
                "Location",
            ),
            "Duration": st.column_config.TextColumn(
                "Duration",
            ),
            "Frequency": st.column_config.TextColumn(
                "Frequency",
            ),
            "Severity": st.column_config.TextColumn(
                "Severity",
            ),
            "Temperature": st.column_config.TextColumn(
                "Temperature",
            ),
            "Quality": st.column_config.TextColumn(
                "Quality",
            ),
            "Verified": st.column_config.CheckboxColumn(
                "Verified",
            ),
            "Needs Review": st.column_config.CheckboxColumn(
                "Needs Review",
            ),
        },
    )


def render_conflicts(findings):
    """Display conflicts without creating symptom cards."""
    conflicts = [
        finding
        for finding in findings
        if finding.get("needs_review", False)
    ]

    if not conflicts:
        return

    st.warning(
        f"{len(conflicts)} finding(s) require manual review."
    )

    with st.expander("⚠ View conflicting findings"):
        for finding in conflicts:
            symptom = str(
                finding.get("symptom", "Unknown")
            ).title()

            st.markdown(f"**{symptom}**")

            for conflict in finding.get("conflicts", []):
                if isinstance(conflict, dict):
                    field = conflict.get("field", "Detail")
                    previous = conflict.get("previous", "")
                    new = conflict.get("new", "")

                    st.write(
                        f"{field}: {previous} vs {new}"
                    )
                else:
                    st.write(str(conflict))


def render_evidence(findings):
    """
    Supporting transcript remains accessible,
    but no individual symptom cards are displayed.
    """
    evidence = [
        finding
        for finding in findings
        if finding.get("source_text")
    ]

    if not evidence:
        return

    with st.expander(
        "📋 View supporting transcript evidence"
    ):
        for finding in evidence:
            symptom = str(
                finding.get("symptom", "Unknown")
            ).title()

            st.markdown(f"**{symptom}**")
            st.write(finding.get("source_text", ""))


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("🎙 Consultation Controls")

    st.checkbox(
        "All participants consented to audio processing",
        key="consent",
    )

    st.caption(
        "Audio is captured from the microphone "
        "of the computer running this application."
    )

    st.radio(
        "Consultation mode",
        [
            "Solo testing (one voice)",
            "Two-person automatic voices",
        ],
        key="consultation_mode",
        disabled=engine.active,
    )

    if (
        st.session_state.consultation_mode
        == "Solo testing (one voice)"
    ):
        st.radio(
            "Default role when wording is unclear",
            ["Doctor", "Patient"],
            key="solo_role",
            horizontal=True,
            disabled=engine.active,
        )

        st.caption(
            "Solo mode guesses roles from wording. "
            "It does not identify different voices. "
            "Correct roles manually when necessary."
        )

    st.divider()

    if st.button(
        "▶ Start Consultation",
        disabled=(
            engine.active
            or not st.session_state.consent
        ),
        use_container_width=True,
        type="primary",
    ):
        try:
            clear_transcript()

            if (
                st.session_state.consultation_mode
                == "Solo testing (one voice)"
            ):
                engine.configure_mode(
                    "solo",
                    st.session_state.solo_role,
                )
            else:
                engine.configure_mode(
                    "two_person"
                )

                st.session_state.speaker1_role = (
                    "Unassigned"
                )
                st.session_state.speaker2_role = (
                    "Unassigned"
                )

            engine.start()

            st.session_state.status = (
                "Starting microphone..."
            )

            st.rerun()

        except Exception as exc:
            st.error(
                f"Cannot start consultation: {exc}"
            )

    if st.button(
        "■ Stop Consultation",
        disabled=(
            not engine.active
            or engine.stop_event.is_set()
        ),
        use_container_width=True,
    ):
        engine.stop()

        st.session_state.status = (
            "Stopping microphone and processing "
            "remaining speech..."
        )

        st.rerun()

    if st.button(
        "🗑 Clear Transcript",
        disabled=engine.active,
        use_container_width=True,
    ):
        clear_transcript()
        st.rerun()

    st.divider()

    if (
        st.session_state.consultation_mode
        == "Two-person automatic voices"
    ):
        st.header("Assign Voices to Roles")

        st.selectbox(
            "Speaker 1 is",
            ["Unassigned", "Doctor", "Patient"],
            key="speaker1_role",
        )

        st.selectbox(
            "Speaker 2 is",
            ["Unassigned", "Doctor", "Patient"],
            key="speaker2_role",
        )

        if roles_conflict():
            st.error(
                "Speaker 1 and Speaker 2 cannot "
                "have the same clinical role."
            )

        st.caption(
            "Confirm speaker mapping before "
            "verifying transcript entries."
        )

    st.divider()

    st.checkbox(
        "Show rejected transcript entries",
        key="show_rejected",
    )

    st.info(
        "Recording continues until Stop Consultation. "
        "Closing the browser does not guarantee "
        "that microphone capture stops."
    )


# ============================================================
# LIVE DASHBOARD
# ============================================================
@st.cache_data(show_spinner=False, max_entries=32)
def cached_analyze_live(lines):
    return analyze_live(lines) or []


@st.cache_data(show_spinner=False, max_entries=32)
def cached_summarize_consultation(lines):
    return summarize_consultation(lines) or []


@st.cache_data(show_spinner=False, max_entries=32)
def cached_aggregate(lines):
    return aggregate(lines) or []
@st.fragment(run_every="300ms")

def live_dashboard():

    receive_events()
    invalidate_verifications_for_mapping_change()

    st.divider()

    if (
        engine.active
        and not engine.stop_event.is_set()
    ):
        st.success(
            "🔴 Recording ON - Continuous listening"
        )

    elif engine.active:
        st.warning(
            "🟡 Finishing queued transcription..."
        )

    else:
        st.info("⚪ Recording OFF")

    st.write(
        "**Status:**",
        st.session_state.status,
    )

    total_lines = len(
        st.session_state.lines
    )

    rejected_count = sum(
        bool(line.get("rejected", False))
        for line in st.session_state.lines
    )

    verified_count = sum(
        bool(line.get("verified", False))
        for line in st.session_state.lines
        if not line.get("rejected", False)
    )

    st.caption(
        f"Events: {st.session_state.events} | "
        f"Speech turns: {total_lines} | "
        f"Verified: {verified_count} | "
        f"Rejected: {rejected_count}"
    )

    if st.session_state.warnings:
        with st.expander(
            "⚠ Audio warnings "
            f"({len(st.session_state.warnings)})"
        ):
            for warning in (
                st.session_state.warnings[-10:]
            ):
                st.warning(warning)

    mapping = get_role_mapping()

    left, right = st.columns([3, 2])

    # ========================================================
    # LIVE TRANSCRIPT
    # ========================================================

    with left:
        st.subheader("📝 Live Transcript")

        if not st.session_state.lines:
            st.info(
                "Start Consultation and speak naturally. "
                "Transcribed speech will appear here."
            )

        inferred_roles = infer_solo_roles(
            st.session_state.lines
        )

        for index, line in enumerate(
            st.session_state.lines
        ):
            if (
                line.get("rejected", False)
                and not st.session_state.show_rejected
            ):
                continue

            with st.container(border=True):

                role = get_effective_role(
                    line,
                    mapping,
                    index,
                    inferred_roles,
                )

                st.caption(
                    f"{line.get('at', '')} | "
                    f"{line.get('speaker', 'Unverified')} | "
                    f"Role: {role}"
                )

                if line.get("speaker_note"):
                    st.caption(
                        line["speaker_note"]
                    )

                if line.get("rejected", False):
                    st.warning(
                        "Rejected transcript. "
                        "Excluded from clinical extraction."
                    )

                # Transcript text editor
                transcript_key = widget_key(
                    "transcript",
                    index,
                )

                if transcript_key not in st.session_state:
                    st.session_state[transcript_key] = (
                        line["text"]
                    )

                st.text_area(
                    "Transcript text",
                    key=transcript_key,
                    height=75,
                    on_change=on_transcript_change,
                    args=(index,),
                )

                # Manual speaker role correction
                manual_key = widget_key(
                    "manual",
                    index,
                )

                if manual_key not in st.session_state:
                    st.session_state[manual_key] = (
                        line.get(
                            "manual_role",
                            "Use speaker mapping",
                        )
                    )

                st.selectbox(
                    "Correct role if needed",
                    ROLE_OPTIONS,
                    key=manual_key,
                    on_change=on_manual_role_change,
                    args=(index,),
                )

                # Reject inaccurate transcript
                reject_key = widget_key(
                    "reject",
                    index,
                )

                if reject_key not in st.session_state:
                    st.session_state[reject_key] = bool(
                        line.get("rejected", False)
                    )

                st.checkbox(
                    "Reject transcript "
                    "(not actually spoken / inaccurate)",
                    key=reject_key,
                    on_change=on_reject_change,
                    args=(index,),
                )

                # Manual verification
                verify_key = widget_key(
                    "verify",
                    index,
                )

                if verify_key not in st.session_state:
                    st.session_state[verify_key] = bool(
                        line.get("verified", False)
                    )

                st.checkbox(
                    "I verified the speaker "
                    "and transcript text",
                    key=verify_key,
                    disabled=line.get(
                        "rejected",
                        False,
                    ),
                    on_change=on_verify_change,
                    args=(index,),
                )

                if line.get("rejected", False):
                    st.caption(
                        "🚫 Excluded from extraction"
                    )

                elif line.get("verified", False):
                    st.caption(
                        "✅ Manually verified"
                    )

                else:
                    st.caption(
                        "⚠ Not verified"
                    )

    # ========================================================
    # CLINICAL EXTRACTION
    # ========================================================

    clinical_lines = build_clinical_lines(
    mapping
    )

    with st.expander("🔧 Debug: Live clinical input"):
        st.json([
            {
                "role": line["role"],
                "text": line["text"]
            }
            for line in clinical_lines
        ])

    try:
        provisional_findings = cached_analyze_live(
            clinical_lines
        )

    except Exception as exc:
        provisional_findings = []
        st.error(
            f"Live symptom extraction error: {exc}"
        )

    try:
        provisional_summary = cached_summarize_consultation(
            clinical_lines
        )

    except Exception as exc:
        provisional_summary = []
        st.error(
            f"Consultation summary error: {exc}"
        )

    try:
        verified_findings = cached_aggregate(
            clinical_lines
        )

    except Exception as exc:
        verified_findings = []
        st.error(
            f"Verified symptom extraction error: {exc}"
        )

    # ========================================================
    # SYMPTOM DASHBOARD - TABLE FORMAT
    # ========================================================

    with right:

        st.subheader(
            "🔎 Live Symptom Extraction"
        )

        reported = [
            item
            for item in provisional_findings
            if (
                item.get("speaker_role") == "Patient"
                and item.get("status") == "Reported"
            )
        ]

        denied = [
            item
            for item in provisional_findings
            if (
                item.get("speaker_role") == "Patient"
                and item.get("status") == "Denied"
            )
        ]

        conflicts = [
            item
            for item in provisional_findings
            if item.get("needs_review", False)
        ]

        # Summary metrics
        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Reported",
            len(reported),
        )

        col2.metric(
            "Denied",
            len(denied),
        )

        col3.metric(
            "Needs review",
            len(conflicts),
        )

        # --------------------------------------------
        # MAIN LIVE SYMPTOM TABLE
        # --------------------------------------------

        st.markdown("**Live Symptom Table**")

        if provisional_findings:

            render_symptom_table(
                provisional_findings,
                height=350,
            )

            # Conflict details remain available.
            render_conflicts(
                provisional_findings
            )

            # Evidence remains available without cards.
            render_evidence(
                provisional_findings
            )

        else:
            st.info(
                "Waiting for symptom mentions "
                "in the transcript..."
            )

        st.caption(
            "Provisional results can include unverified "
            "information. Doctor questions alone do not "
            "establish that a symptom is present."
        )

        st.divider()

        # --------------------------------------------
        # VERIFIED SYMPTOM TABLE
        # --------------------------------------------

        st.subheader(
            "🧠 Verified Patient Symptoms"
        )

        if verified_findings:

            render_symptom_table(
                verified_findings,
                height=250,
            )

        else:
            st.info(
                "No verified Patient findings yet. "
                "Review transcript text, correct "
                "speaker roles, and verify the "
                "supporting Patient statements."
            )

        st.caption(
            "Verification confirms manual review "
            "of transcript evidence, not a diagnosis."
        )

    # ========================================================
    # CONSULTATION EXPORT
    # ========================================================

    if st.session_state.lines:
        st.divider()

        st.subheader(
            "📁 Consultation Export"
        )

        export_data = {
            "application": "Doc AI",
            "type": "Simulated consultation",
            "exported_at": datetime.now().isoformat(),
            "consultation_mode": (
                st.session_state.consultation_mode
            ),
            "role_mapping": mapping,
            "transcript": st.session_state.lines,
            "clinical_input": clinical_lines,
            "provisional_symptom_mentions": (
                provisional_findings
            ),
            "provisional_patient_detail_summary": (
                provisional_summary
            ),
            "verified_patient_findings_demo": (
                verified_findings
            ),
            "clinical_review_required": any(
                item.get("needs_review", False)
                for item in provisional_findings
            ),
        }

        st.download_button(
            "⬇ Download Consultation JSON",
            data=json.dumps(
                export_data,
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            file_name="doc_ai_consultation.json",
            mime="application/json",
            use_container_width=True,
        )


# ============================================================
# START DASHBOARD
# ============================================================

live_dashboard()
