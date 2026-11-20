
"""
Doc AI - Hybrid Clinical Symptom Extraction
===========================================

Public functions used by app.py:
    analyze_live(lines)
    aggregate(lines)
    summarize_consultation(lines)

Features:
- Broad symptom terminology
- Doctor / Patient conversation context
- Yes / No question-answer linking
- Follow-up duration and location linking
- Explicit patient corrections
- Corrections without doctor follow-up questions
- Temperature extraction without guessing units
- Evidence and transcript verification
- Review flags for corrections and ambiguous measurements
- Optional OpenAI-assisted extraction

Optional environment variables:
    DOC_AI_ENABLE_LLM=1
    OPENAI_API_KEY=your_api_key
    DOC_AI_MODEL=gpt-4o-mini

Clinical documentation prototype only.
All findings require appropriate clinician review.
"""

import copy
import json
import os
import re
import urllib.error
import urllib.request


# ============================================================
# CONFIGURATION
# ============================================================

AI_ENABLED = os.getenv(
    "DOC_AI_ENABLE_LLM", "0"
).lower() in ("1", "true", "yes")

AI_MODEL = os.getenv(
    "DOC_AI_MODEL", "gpt-4o-mini"
)

AI_TIMEOUT = 25
MAX_AI_TURNS = 100
MAX_AI_CHARACTERS = 20000


# ============================================================
# MEDICAL SYMPTOM VOCABULARY
# ============================================================

SYMPTOM_TERMS = {
    "fever": [
        "fever", "febrile", "feverish",
        "high temperature", "running a temperature",
    ],
    "headache": [
        "headache", "head pain", "head hurts",
        "pain in my head", "migraine",
    ],
    "vomiting": [
        "vomit", "vomiting", "vomited",
        "throwing up", "threw up", "puking",
    ],
    "nausea": [
        "nausea", "nauseous", "feel like vomiting",
        "feeling sick to my stomach",
    ],
    "diarrhea": [
        "diarrhea", "diarrhoea", "loose motion",
        "loose motions", "loose stool",
        "loose stools", "watery stool",
    ],
    "stomach pain": [
        "stomach pain", "stomachache", "stomach ache",
        "stomach hurts", "abdominal pain",
        "belly pain", "bellyache", "tummy pain",
    ],
    "body pain": [
        "body pain", "body ache", "body aches",
        "muscle pain", "muscle aches", "myalgia",
    ],
    "joint pain": [
        "joint pain", "joint pains",
        "joint ache", "arthralgia",
    ],
    "leg pain": [
        "leg pain", "leg hurts", "legs hurt",
        "thigh pain", "knee pain",
        "calf pain", "ankle pain",
    ],
    "arm pain": [
        "arm pain", "arm hurts",
        "forearm pain", "elbow pain",
    ],
    "shoulder pain": [
        "shoulder pain", "shoulder hurts",
        "pain in my shoulder",
    ],
    "back pain": [
        "back pain", "backache", "back hurts",
        "lower back pain", "upper back pain",
    ],
    "neck pain": [
        "neck pain", "neck hurts",
        "stiff neck", "neck stiffness",
    ],
    "chest pain": [
        "chest pain", "chest hurts",
        "chest discomfort", "chest tightness",
    ],
    "cough": [
        "cough", "coughing", "dry cough",
        "wet cough", "productive cough",
    ],
    "sore throat": [
        "sore throat", "throat pain",
        "throat hurts", "painful swallowing",
    ],
    "runny nose": [
        "runny nose", "running nose",
        "nasal discharge",
    ],
    "nasal congestion": [
        "blocked nose", "stuffy nose",
        "nasal congestion",
    ],
    "shortness of breath": [
        "shortness of breath", "breathlessness",
        "difficulty breathing", "trouble breathing",
        "breathless", "cannot breathe",
        "can't breathe",
    ],
    "wheezing": [
        "wheezing", "wheeze",
        "whistling breathing",
    ],
    "dizziness": [
        "dizziness", "dizzy",
        "lightheaded", "light headed",
        "feeling faint",
    ],
    "vertigo": [
        "vertigo", "room is spinning",
        "everything is spinning",
        "spinning sensation",
    ],
    "weakness": [
        "weakness", "feel weak",
        "feeling weak", "loss of strength",
    ],
    "fatigue": [
        "fatigue", "tiredness",
        "exhaustion", "always tired",
        "extremely tired",
    ],
    "chills": [
        "chills", "shivering", "rigors",
    ],
    "sweating": [
        "excessive sweating", "night sweats",
        "sweating a lot",
    ],
    "constipation": [
        "constipation", "constipated",
        "difficulty passing stool",
    ],
    "bloating": [
        "bloating", "bloated",
        "abdominal distension",
    ],
    "heartburn": [
        "heartburn", "acid reflux",
        "acidity", "burning in my chest",
    ],
    "loss of appetite": [
        "loss of appetite", "poor appetite",
        "not feeling hungry", "reduced appetite",
    ],
    "difficulty swallowing": [
        "difficulty swallowing",
        "trouble swallowing",
        "food gets stuck", "dysphagia",
    ],
    "painful urination": [
        "painful urination", "burning urination",
        "burning while urinating", "dysuria",
    ],
    "frequent urination": [
        "frequent urination",
        "urinating frequently",
        "peeing frequently",
    ],
    "blood in urine": [
        "blood in urine", "bloody urine",
        "hematuria", "red urine",
    ],
    "rash": [
        "rash", "skin rash", "skin rashes",
    ],
    "itching": [
        "itching", "itchiness",
        "itchy skin", "itchy",
    ],
    "swelling": [
        "swelling", "swollen", "puffiness",
    ],
    "palpitations": [
        "palpitations", "heart racing",
        "racing heartbeat", "heart pounding",
    ],
    "fainting": [
        "fainting", "fainted",
        "syncope", "passed out",
    ],
    "tingling": [
        "tingling", "pins and needles",
        "tingling sensation",
    ],
    "numbness": [
        "numbness", "numb",
        "loss of sensation",
    ],
    "burning sensation": [
        "burning sensation", "burning feeling",
        "burning in my feet",
    ],
    "blurred vision": [
        "blurred vision", "blurry vision",
        "vision is blurry",
    ],
    "double vision": [
        "double vision", "seeing double",
        "diplopia",
    ],
    "hearing loss": [
        "hearing loss", "difficulty hearing",
        "can't hear properly",
    ],
    "tinnitus": [
        "tinnitus", "ringing in ears",
        "ringing in my ears",
    ],
    "sleep disturbance": [
        "insomnia", "cannot sleep",
        "can't sleep", "trouble sleeping",
        "difficulty sleeping",
    ],
    "anxiety": [
        "anxiety", "feeling anxious",
    ],
    "low mood": [
        "low mood", "feeling depressed",
        "feeling very sad",
    ],
    "memory difficulty": [
        "memory problems", "forgetfulness",
        "difficulty remembering",
    ],
    "heavy menstrual bleeding": [
        "heavy menstrual bleeding",
        "heavy periods", "menorrhagia",
    ],
    "menstrual pain": [
        "period pain", "menstrual cramps",
        "painful periods", "dysmenorrhea",
    ],
    "weight loss": [
        "unintentional weight loss",
        "losing weight", "weight loss",
    ],
    "weight gain": [
        "weight gain", "gaining weight",
    ],
    "excessive thirst": [
        "excessive thirst", "always thirsty",
        "very thirsty", "polydipsia",
    ],
}


TERM_PATTERNS = {}

for symptom, phrases in SYMPTOM_TERMS.items():
    alternatives = sorted(
        set(phrases),
        key=len,
        reverse=True,
    )

    TERM_PATTERNS[symptom] = re.compile(
        r"(?<!\w)(?:"
        + "|".join(
            re.escape(phrase)
            for phrase in alternatives
        )
        + r")(?!\w)",
        re.I,
    )


# ============================================================
# NUMBER AND ATTRIBUTE EXTRACTION
# ============================================================

NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}

NUMBER_RE = (
    r"(?:\d+(?:\.\d+)?|"
    + "|".join(
        sorted(NUMBER_WORDS, key=len, reverse=True)
    )
    + r")"
)

DURATION_RE = re.compile(
    rf"\b(?P<count>{NUMBER_RE})\s*"
    r"(?P<unit>days?|weeks?|months?|"
    r"years?|hours?|minutes?)\b",
    re.I,
)

FREQUENCY_RE = re.compile(
    rf"\b(?P<count>{NUMBER_RE})\s*"
    r"(?:times?|episodes?)"
    r"(?:\s*(?:per|a|each|every)\s*"
    r"(?P<period>day|week|hour|month))?\b",
    re.I,
)

SEVERITY_RE = re.compile(
    r"\b(?:very severe|mild|moderate|severe|"
    r"unbearable|intense|extreme)\b|"
    r"\b(?:10|[0-9])\s*(?:/|out of)\s*10\b",
    re.I,
)

QUALITY_RE = re.compile(
    r"\b(?:sharp|dull|throbbing|stabbing|"
    r"burning|aching|cramping|shooting|"
    r"constant|intermittent|pulsating|"
    r"dry|productive|hard)\b",
    re.I,
)

LOCATION_RE = re.compile(
    r"\b(?:left|right|both)\s+"
    r"(?:side|shoulder|arm|hand|wrist|"
    r"leg|knee|thigh|calf|ankle|foot|"
    r"hip|chest|abdomen|stomach|"
    r"temple|eye|ear|elbow)\b|"
    r"\b(?:upper|lower|middle)\s+"
    r"(?:back|abdomen|stomach)\b|"
    r"\b(?:forehead|behind the eyes|"
    r"back of head)\b",
    re.I,
)

TEMPERATURE_RE = re.compile(
    r"(?<![\w.])"
    r"(?P<value>\d{2,3}(?:\.\d+)?)"
    r"\s*(?:°\s*|degrees?\s*)?"
    r"(?P<unit>fahrenheit|celsius|[fFcC])?"
    r"(?!\w)",
    re.I,
)

TEMPERATURE_CONTEXT_RE = re.compile(
    r"\b(?:body temperature|temperature|temp|"
    r"degrees?|fahrenheit|celsius)\b|°",
    re.I,
)

TEMPERATURE_SPOKEN_RE = re.compile(
    r"\b(?:"
    r"one\s+hundred(?:\s+and)?(?:\s+"
    r"(?:twenty|thirty|forty|fifty|sixty|"
    r"seventy|eighty|ninety)(?:[-\s]+"
    r"(?:one|two|three|four|five|six|"
    r"seven|eight|nine))?"
    r"|\s+(?:one|two|three|four|five|"
    r"six|seven|eight|nine|ten|eleven|"
    r"twelve|thirteen|fourteen|fifteen|"
    r"sixteen|seventeen|eighteen|nineteen))?"
    r"|(?:thirty|forty|fifty|sixty|"
    r"seventy|eighty|ninety)(?:[-\s]+"
    r"(?:one|two|three|four|five|six|"
    r"seven|eight|nine))?"
    r"|(?:one|two|three|four|five|six|"
    r"seven|eight|nine|zero|oh)"
    r"(?:\s+(?:one|two|three|four|five|"
    r"six|seven|eight|nine|zero|oh)){1,2}"
    r")\b",
    re.I,
)


def spoken_temperature_number(text):
    """Convert spoken temperature values to numbers."""

    match = TEMPERATURE_SPOKEN_RE.search(
        str(text)
    )

    if not match:
        return None

    words = (
        match.group(0)
        .lower()
        .replace("-", " ")
        .replace(" and ", " ")
        .split()
    )

    digits = {
        "zero": 0, "oh": 0,
        "one": 1, "two": 2,
        "three": 3, "four": 4,
        "five": 5, "six": 6,
        "seven": 7, "eight": 8,
        "nine": 9,
    }

    tens = {
        "twenty": 20, "thirty": 30,
        "forty": 40, "fifty": 50,
        "sixty": 60, "seventy": 70,
        "eighty": 80, "ninety": 90,
    }

    teens = {
        "ten": 10, "eleven": 11,
        "twelve": 12, "thirteen": 13,
        "fourteen": 14, "fifteen": 15,
        "sixteen": 16, "seventeen": 17,
        "eighteen": 18, "nineteen": 19,
    }

    if "hundred" in words:
        index = words.index("hundred")
        before = words[:index]
        after = words[index + 1:]

        if before != ["one"]:
            return None

        value = 100

        if after:
            if after[0] in tens:
                value += tens[after[0]]
                if len(after) > 1:
                    value += digits.get(after[1], 0)
            else:
                value += teens.get(
                    after[0],
                    digits.get(after[0], 0),
                )

        return value

    if words[0] in tens:
        value = tens[words[0]]
        if len(words) > 1:
            value += digits.get(words[1], 0)
        return value

    if (
        len(words) in (2, 3)
        and all(word in digits for word in words)
    ):
        return int(
            "".join(str(digits[word]) for word in words)
        )

    return None


CORRECTION_RE = re.compile(
    r"\b(?:no[, ]+|not\s+|"
    r"i mean\b|actually\b|"
    r"correction\b|sorry\b|"
    r"instead\b|rather\b)",
    re.I,
)

DETAIL_FIELDS = (
    "duration",
    "location",
    "severity",
    "frequency",
    "temperature",
    "quality",
    "onset",
)


def number_value(value):
    value = str(value).lower().strip()

    if value in NUMBER_WORDS:
        return NUMBER_WORDS[value]

    try:
        result = float(value)
        return (
            int(result)
            if result.is_integer()
            else result
        )
    except ValueError:
        return None


def extract_duration(text):
    match = DURATION_RE.search(str(text))

    if not match:
        return ""

    count = number_value(match.group("count"))
    unit = match.group("unit").lower()

    if count is None:
        return ""

    singular = unit.rstrip("s")

    return (
        f"{count} {singular}"
        if count == 1
        else f"{count} {singular}s"
    )


def extract_frequency(text):
    match = FREQUENCY_RE.search(str(text))

    if not match:
        return ""

    count = number_value(match.group("count"))

    if count is None:
        return ""

    period = match.group("period")

    return (
        f"{count} times/{period.lower()}"
        if period
        else f"{count} times"
    )


def extract_location(text, symptom=""):
    match = LOCATION_RE.search(str(text))

    if match:
        return match.group(0).lower()

    if re.fullmatch(r"\s*(?:the\s+)?chest[.!]?\s*", str(text), re.I):
        return "chest"

    return ""

def extract_severity(text):
    match = SEVERITY_RE.search(str(text))

    return (
        match.group(0).lower()
        if match
        else ""
    )


def extract_quality(text):
    match = QUALITY_RE.search(str(text))

    return (
        match.group(0).lower()
        if match
        else ""
    )



def extract_temperature(text, allow_bare=False):
    """
    Examples:
        "102 degrees" -> "102°"
        "102" -> "102°" (when allow_bare=True)
        "one hundred and two degrees" -> "102°"
        "102 Fahrenheit" -> "102°F"
        "39 Celsius" -> "39°C"

    Do not assume Fahrenheit or Celsius.
    """
    text = str(text or "").strip()

    if (
        not allow_bare
        and not TEMPERATURE_CONTEXT_RE.search(text)
    ):
        return ""

    match = TEMPERATURE_RE.search(text)

    if match:
        value = float(match.group("value"))
        unit = (match.group("unit") or "").lower()
    else:
        spoken = spoken_temperature_number(text)

        if spoken is None:
            return ""

        value = float(spoken)
        unit = ""

    if not 30 <= value <= 115:
        return ""

    number = (
        str(int(value))
        if value.is_integer()
        else str(value)
    )

    if not unit:
        if re.search(r"\b(?:fahrenheit)\b|°\s*f\b", text, re.I):
            unit = "f"
        elif re.search(r"\b(?:celsius|centigrade)\b|°\s*c\b", text, re.I):
            unit = "c"

    if unit in ("f", "fahrenheit"):
        return f"{number}°F"

    if unit in ("c", "celsius"):
        return f"{number}°C"

    return f"{number}°"



def temperature_needs_review(value):
    if not value:
        return False

    match = re.search(
        r"\d+(?:\.\d+)?",
        value,
    )

    if not match:
        return True

    number = float(match.group(0))

    if "°C" in value:
        return not (30 <= number <= 45)

    if "°F" in value:
        return not (86 <= number <= 113)

    # A degree symbol alone does not establish
    # whether the scale is Celsius or Fahrenheit.
    return True


# ============================================================
# TRANSCRIPT NORMALIZATION
# ============================================================

def normalize_lines(lines):
    normalized = []

    for index, item in enumerate(lines):
        if isinstance(item, str):
            item = {
                "text": item,
                "role": "Unassigned",
                "verified": False,
            }

        if not isinstance(item, dict):
            continue

        if item.get("rejected", False):
            continue

        text = str(
            item.get("text", "") or ""
        ).strip()

        if not text:
            continue

        role = item.get(
            "role",
            item.get(
                "speaker_role",
                "Unassigned",
            ),
        )

        if role not in ("Doctor", "Patient"):
            role = "Unassigned"

        normalized.append({
            "id": index,
            "text": text,
            "role": role,
            "verified": bool(
                item.get("verified", False)
            ),
        })

    return normalized


def find_symptoms(text):
    matches = []

    for symptom, pattern in TERM_PATTERNS.items():
        for match in pattern.finditer(text):
            matches.append((
                match.start(),
                match.end(),
                symptom,
            ))

    matches.sort(
        key=lambda item: (
            item[0],
            -(item[1] - item[0]),
        )
    )

    results = []
    occupied = []

    for start, end, symptom in matches:
        overlap = any(
            start < other_end
            and end > other_start
            for other_start, other_end in occupied
        )

        if overlap:
            continue

        occupied.append((start, end))

        if symptom not in results:
            results.append(symptom)

    return results


def is_question(text):
    return bool(
        text.rstrip().endswith("?")
        or re.search(
            r"^\s*(?:do you|did you|have you|"
            r"are you|what|where|which|how|"
            r"since when|for how many|"
            r"does it|is your|can you)\b",
            text,
            re.I,
        )
    )


def answer_polarity(text):
    clean = str(text).lower().strip(" .!?")

    if re.fullmatch(
        r"(?:yes|yeah|yep|yes i do|"
        r"yes i have|correct)",
        clean,
    ):
        return "Reported"

    if re.fullmatch(
        r"(?:no|nope|nah|no i don't|"
        r"i do not|not at all)",
        clean,
    ):
        return "Denied"

    return None


def symptom_polarity(text, symptom):
    pattern = TERM_PATTERNS.get(symptom)

    if not pattern:
        return "Reported"

    match = pattern.search(text)

    if not match:
        return "Reported"

    prefix = text[
        max(0, match.start() - 60):
        match.start()
    ]

    prefix = re.split(
        r"[,;.]|\bbut\b|\band\b",
        prefix,
        flags=re.I,
    )[-1]

    if re.search(
        r"\b(?:no|not|never|without|"
        r"don't have|do not have|"
        r"haven't had|didn't have)\b",
        prefix,
        re.I,
    ):
        return "Denied"

    if re.search(
        r"\b(?:maybe|perhaps|possibly|"
        r"might|not sure|i think)\b",
        prefix,
        re.I,
    ):
        return "Uncertain"

    return "Reported"


# ============================================================
# FINDING MANAGEMENT
# ============================================================

def new_finding(symptom):
    return {
        "symptom": symptom,
        "status": "Question",
        "reported": "Unknown",
        "speaker_role": "Doctor",
        "duration": "",
        "measurement": "",
        "location": "",
        "severity": "",
        "frequency": "",
        "temperature": "",
        "verified": False,
        "source_text": "",
        "associated_symptoms": [],
        "quality": "",
        "onset": "",
        "conflicts": [],
        "evidence": [],
        "needs_review": False,
    }


def update_measurement(row):
    if row["symptom"] == "temperature":
        row["measurement"] = row["temperature"]
        return

    row["measurement"] = "; ".join(
        row[field]
        for field in (
            "location",
            "severity",
            "frequency",
            "quality",
        )
        if row.get(field)
    )


def add_evidence(row, line):
    evidence = {
        "turn_id": line["id"],
        "text": line["text"],
        "role": line["role"],
        "verified": line["verified"],
    }

    if not any(
        item["turn_id"] == line["id"]
        for item in row["evidence"]
    ):
        row["evidence"].append(evidence)

    patient_evidence = [
        item
        for item in row["evidence"]
        if item["role"] == "Patient"
    ]

    row["verified"] = (
        bool(patient_evidence)
        and all(
            item["verified"]
            for item in patient_evidence
        )
    )

    row["source_text"] = " ".join(
        item["text"]
        for item in patient_evidence
    )


def set_detail(
    row,
    field,
    value,
    source,
    correction=False,
):
    if field not in DETAIL_FIELDS or not value:
        return

    value = str(value).strip()
    previous = row.get(field, "")

    if previous == value:
        return

    if previous and previous != value:
        history = {
            "field": field,
            "previous": previous,
            "new": value,
            "source": source,
            "explicit_correction": bool(correction),
        }

        if history not in row["conflicts"]:
            row["conflicts"].append(history)

        if not correction:
            row["needs_review"] = True
            return

        # Explicit correction:
        # Use the corrected value and retain history.
        row["needs_review"] = True

    row[field] = value

    if field == "temperature":
        if temperature_needs_review(value):
            row["needs_review"] = True

    update_measurement(row)


def update_finding(
    rows,
    symptom,
    status,
    line,
    details=None,
):
    symptom = str(symptom).lower().strip()

    if not symptom:
        return

    if symptom not in rows:
        rows[symptom] = new_finding(symptom)

    row = rows[symptom]
    role = line["role"]

    if role == "Doctor":
        if row["speaker_role"] != "Patient":
            row["status"] = "Question"
            row["speaker_role"] = "Doctor"
        return

    if role != "Patient":
        return

    if (
        row["speaker_role"] == "Patient"
        and row["status"] in ("Reported", "Denied")
        and status in ("Reported", "Denied")
        and row["status"] != status
    ):
        row["conflicts"].append({
            "field": "status",
            "previous": row["status"],
            "new": status,
            "source": line["text"],
        })
        row["needs_review"] = True

    row["speaker_role"] = "Patient"
    row["status"] = status

    row["reported"] = {
        "Reported": "Yes",
        "Denied": "No",
        "Uncertain": "Unknown",
    }.get(status, "Unknown")

    add_evidence(row, line)

    if status == "Reported":
        for field, value in (details or {}).items():
            set_detail(
                row,
                field,
                value,
                line["text"],
            )


# ============================================================
# QUESTION AND FOLLOW-UP CONTEXT
# ============================================================

def question_field(text):
    text = text.lower()

    if re.search(
        r"\b(?:temperature|temp|how high)\b",
        text,
    ):
        return "temperature"

    if re.search(
        r"\b(?:where|which part|which side|"
        r"what part|which body part)\b",
        text,
    ):
        return "location"

    if re.search(
        r"\b(?:how long|since when|"
        r"how many days|how many weeks|"
        r"when did it start)\b",
        text,
    ):
        return "duration"

    if re.search(
        r"\b(?:how many times|how often|"
        r"how frequently|how many episodes)\b",
        text,
    ):
        return "frequency"

    if re.search(
        r"\b(?:how severe|how bad|"
        r"rate your pain|pain scale)\b",
        text,
    ):
        return "severity"

    return "yes_no"


def question_target(text, symptoms, last_symptom):
    if len(symptoms) == 1:
        return symptoms[0]

    if len(symptoms) > 1:
        return None

    lower = text.lower()

    if "head" in lower:
        return "headache"

    if re.search(
        r"\b(?:stomach|abdomen|belly)\b",
        lower,
    ):
        return "stomach pain"

    if "shoulder" in lower:
        return "shoulder pain"

    if "vomit" in lower:
        return "vomiting"

    return last_symptom


def answer_detail(text, field, symptom):
    if field == "duration":
        return extract_duration(text)

    if field == "frequency":
        return extract_frequency(text)

    if field == "severity":
        return extract_severity(text)

    if field == "location":
        return extract_location(text, symptom)

    if field == "temperature":
        return extract_temperature(
            text,
            allow_bare=True,
        )

    return ""


def explicit_correction(text):
    return bool(
        CORRECTION_RE.search(text)
    )


def correction_detail(text, field):
    """
    Select the final explicitly stated value.

    Examples:
        "Not one week, two days." -> "2 days"
        "No, five days." -> "5 days"
        "Not 2 days, 5 days." -> "5 days"
    """
    if field == "duration":
        matches = list(
            DURATION_RE.finditer(text)
        )

        if not matches:
            return ""

        return extract_duration(
            matches[-1].group(0)
        )

    if field == "frequency":
        matches = list(
            FREQUENCY_RE.finditer(text)
        )

        if not matches:
            return ""

        return extract_frequency(
            matches[-1].group(0)
        )

    return ""


# ============================================================
# NEW: PATIENT CORRECTION HANDLER
# ============================================================

def apply_patient_correction(
    rows,
    line,
    last_patient_symptom,
    last_detail_target=None,
):
    """
    Handle explicit corrections without requiring
    a preceding doctor question.

    Examples:

        Patient: I have fever for one week.
        Patient: Not one week, two days.

        Patient: Two days.
        Patient: No, five days.

    Returns True if a correction was applied.
    """
    text = line["text"]

    if not explicit_correction(text):
        return False

    if last_detail_target:
        target, field = last_detail_target
    else:
        target = last_patient_symptom
        field = None

    if not target or target not in rows:
        return False

    row = rows[target]

    if (
        row["speaker_role"] != "Patient"
        or row["status"] != "Reported"
    ):
        return False

    for candidate_field in (
        "duration",
        "frequency",
    ):
        if field not in (None, candidate_field):
            continue

        corrected = correction_detail(
            text,
            candidate_field,
        )

        if not corrected:
            continue

        set_detail(
            row,
            candidate_field,
            corrected,
            text,
            correction=True,
        )

        add_evidence(row, line)

        return True

    return False


# ============================================================
# LOCAL EXTRACTION
# ============================================================

def local_extract(transcript):
    rows = {}

    pending = None
    last_patient_symptom = None
    last_detail_target = None
    pending_duration_correction = None

    for line in transcript:
        text = line["text"]
        role = line["role"]

        symptoms = find_symptoms(text)

        # ----------------------------------------
        # DOCTOR
        # ----------------------------------------

        if role == "Doctor":
            if is_question(text):
                field = question_field(text)

                target = question_target(
                    text,
                    symptoms,
                    last_patient_symptom,
                )

                if field == "temperature":
                    pending = (
                        "temperature",
                        "temperature",
                    )

                elif target:
                    pending = (target, field)

                else:
                    pending = None

                for symptom in symptoms:
                    update_finding(
                        rows,
                        symptom,
                        "Question",
                        line,
                    )
            else:
                pending = None

            last_detail_target = None
            continue

        # If the current line is not from a patient, reset the pending state.
        if role != "Patient":
            pending = None
            last_detail_target = None
            pending_duration_correction = None
            continue

        # Complete a correction split across transcript turns.
        if pending_duration_correction is not None:
            target = pending_duration_correction
            corrected_duration = extract_duration(text)

            if (
                target in rows
                and rows[target]["speaker_role"] == "Patient"
                and rows[target]["status"] == "Reported"
                and corrected_duration
                and not symptoms
            ):
                set_detail(
                    rows[target],
                    "duration",
                    corrected_duration,
                    text,
                    correction=True,
                )
                add_evidence(rows[target], line)
                pending_duration_correction = None
                pending = None
                last_detail_target = (target, "duration")
                continue

            pending_duration_correction = None

        # Remember an incomplete duration correction.
        if (
            last_patient_symptom
            and last_patient_symptom in rows
            and rows[last_patient_symptom]["speaker_role"] == "Patient"
            and rows[last_patient_symptom]["status"] == "Reported"
            and re.match(r"^\s*not\s+", text, re.I)
            and len(list(DURATION_RE.finditer(text))) == 1
            and not symptoms
        ):
            pending_duration_correction = last_patient_symptom
            add_evidence(rows[last_patient_symptom], line)
            continue

        # ----------------------------------------
        # NEW: EXPLICIT PATIENT CORRECTIONS
        # ----------------------------------------

        # This runs BEFORE ordinary symptom
        # extraction and follow-up processing.
        #
        # It fixes:
        # "I am having fever for one week."
        # "Not one week, two days."
        #
        # The fever duration becomes "2 days".
        if apply_patient_correction(
            rows,
            line,
            last_patient_symptom,
            last_detail_target,
        ):
            pending = None
            continue

        # ----------------------------------------
        # PATIENT EXPLICIT SYMPTOMS
        # ----------------------------------------

        if symptoms and not is_question(text):
            for symptom in symptoms:
                status = symptom_polarity(
                    text,
                    symptom,
                )

                details = {}

                if (
                    status == "Reported"
                    and len(symptoms) == 1
                ):
                    details = {
                        "duration": extract_duration(text),
                        "frequency": extract_frequency(text),
                        "location": extract_location(
                            text,
                            symptom,
                        ),
                        "severity": extract_severity(text),
                        "quality": extract_quality(text),
                    }

                    previous_duration = (
                    rows[symptom]["duration"]
                    if symptom in rows
                    else ""
                )

                new_duration = details.get("duration", "")

                update_finding(
                    rows,
                    symptom,
                    status,
                    line,
                    details,
                )

                if (
                    status == "Reported"
                    and previous_duration
                    and new_duration
                    and previous_duration != new_duration
                ):
                    set_detail(
                        rows[symptom],
                        "duration",
                        new_duration,
                        text,
                        correction=True,
                    )

                last_patient_symptom = symptom

            temperature_value = extract_temperature(text)

            if temperature_value:
                update_finding(
                    rows,
                    "temperature",
                    "Reported",
                    line,
                    {"temperature": temperature_value},
                )

            pending = None
            last_detail_target = None
            continue

        # ----------------------------------------
        # PATIENT YES / NO ANSWERS
        # ----------------------------------------

        if pending and pending[1] == "yes_no":
            status = answer_polarity(text)

            if status:
                target = pending[0]

                update_finding(
                    rows,
                    target,
                    status,
                    line,
                )

                last_patient_symptom = target
                pending = None
                last_detail_target = None
                continue

        # ----------------------------------------
        # FOLLOW-UP ANSWERS
        # ----------------------------------------

        if pending:
            target, field = pending

            value = answer_detail(
                text,
                field,
                target,
            )

            if value:
                if field == "temperature":
                    update_finding(
                        rows,
                        "temperature",
                        "Reported",
                        line,
                        {"temperature": value},
                    )

                elif (
                    target in rows
                    and rows[target]["status"] == "Reported"
                    and rows[target]["speaker_role"] == "Patient"
                ):
                    set_detail(
                        rows[target],
                        field,
                        value,
                        text,
                    )

                    add_evidence(
                        rows[target],
                        line,
                    )

                    last_detail_target = (
                        target,
                        field,
                    )

                    last_patient_symptom = target

                pending = None
                continue

            if len(text.split()) > 6:
                pending = None

        # ----------------------------------------
        # STANDALONE TEMPERATURE
        # ----------------------------------------

        temperature_value = extract_temperature(text)

        if temperature_value:
            update_finding(
                rows,
                "temperature",
                "Reported",
                line,
                {
                    "temperature": temperature_value
                },
            )

            continue

        # ----------------------------------------
        # STANDALONE QUALITY
        # ----------------------------------------

        # Example:
        # Doctor: Do you have any cough?
        # Patient: Yes.
        # Patient: Hard cough.
        #
        # The explicit symptom block above already
        # handles "Hard cough", but short quality-only
        # answers can also be linked to the latest
        # patient-reported symptom.

        if (
            last_patient_symptom
            and last_patient_symptom in rows
            and rows[last_patient_symptom]["status"]
            == "Reported"
        ):
            quality = extract_quality(text)

            if (
                quality
                and len(text.split()) <= 4
            ):
                set_detail(
                    rows[last_patient_symptom],
                    "quality",
                    quality,
                    text,
                )

                add_evidence(
                    rows[last_patient_symptom],
                    line,
                )

    return rows


# ============================================================
# OPTIONAL AI EXTRACTION
# ============================================================

AI_SYSTEM_PROMPT = """
You extract patient-reported symptoms from
doctor-patient consultation transcripts.

This is documentation, not diagnosis.

Rules:
1. Extract common and unfamiliar symptoms.
2. Never infer a diagnosis.
3. Doctor questions are not patient reports.
4. Link follow-up answers only when clear.
5. Preserve negation and uncertainty.
6. Do not transfer details between symptoms.
7. Do not guess temperature units.
8. Treat "102 degrees" as a temperature with
   an unspecified unit.
9. Preserve explicit corrections.
10. Every finding must cite Patient turn IDs.
11. Never invent symptoms or evidence.
12. Leave ambiguous details blank.
13. When a patient explicitly corrects a value,
    the corrected value takes precedence.

Return JSON:

{
  "findings": [
    {
      "symptom": "short symptom name",
      "status": "Reported",
      "duration": "",
      "location": "",
      "severity": "",
      "frequency": "",
      "temperature": "",
      "quality": "",
      "onset": "",
      "evidence_turn_ids": [0],
      "needs_review": false
    }
  ]
}

Allowed statuses:
Reported, Denied, Uncertain.

Do not include doctor-only questions.
"""


def call_ai_extractor(transcript):
    if not AI_ENABLED:
        return None

    api_key = os.getenv(
        "OPENAI_API_KEY", ""
    ).strip()

    if not api_key:
        return None

    selected = transcript[-MAX_AI_TURNS:]

    payload_lines = [
        {
            "id": line["id"],
            "role": line["role"],
            "text": line["text"],
        }
        for line in selected
    ]

    content = json.dumps(
        payload_lines,
        ensure_ascii=False,
    )

    if len(content) > MAX_AI_CHARACTERS:
        return None

    request_body = {
        "model": AI_MODEL,
        "temperature": 0,
        "response_format": {
            "type": "json_object",
        },
        "messages": [
            {
                "role": "system",
                "content": AI_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": content,
            },
        ],
    }

    request = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(
            request_body
        ).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=AI_TIMEOUT,
        ) as response:
            result = json.loads(
                response.read().decode("utf-8")
            )

        message = result["choices"][0]["message"]
        parsed = json.loads(message["content"])

        findings = parsed.get(
            "findings", []
        )

        return (
            findings
            if isinstance(findings, list)
            else None
        )

    except (
        urllib.error.URLError,
        TimeoutError,
        KeyError,
        ValueError,
        TypeError,
        IndexError,
    ):
        return None


# ============================================================
# AI FINDING MERGE
# ============================================================

def normalize_symptom_name(name):
    name = re.sub(
        r"\s+",
        " ",
        str(name or "").lower().strip(),
    )

    if not name or len(name) > 100:
        return ""

    for symptom, pattern in TERM_PATTERNS.items():
        if pattern.fullmatch(name):
            return symptom

    return name


def merge_ai_findings(rows, ai_findings, transcript):
    by_id = {
        line["id"]: line
        for line in transcript
    }

    for finding in ai_findings:
        if not isinstance(finding, dict):
            continue

        symptom = normalize_symptom_name(
            finding.get("symptom")
        )

        status = finding.get("status")

        if (
            not symptom
            or status not in (
                "Reported",
                "Denied",
                "Uncertain",
            )
        ):
            continue

        raw_ids = finding.get(
            "evidence_turn_ids", []
        )

        if not isinstance(raw_ids, list):
            continue

        evidence_lines = []

        for turn_id in raw_ids:
            if (
                isinstance(turn_id, bool)
                or not isinstance(turn_id, int)
            ):
                continue

            line = by_id.get(turn_id)

            if line and line["role"] == "Patient":
                evidence_lines.append(line)

        if not evidence_lines:
            continue

        details = {}

        for field in DETAIL_FIELDS:
            value = finding.get(field, "")

            if isinstance(
                value,
                (str, int, float),
            ):
                details[field] = str(
                    value
                ).strip()[:200]

        # IMPORTANT:
        # Do not let AI restore an old duration
        # after an explicit patient correction.

        existing = rows.get(symptom)

        if existing:
            for conflict in existing["conflicts"]:
                if conflict.get(
                    "explicit_correction"
                ):
                    field = conflict["field"]
                    details.pop(field, None)

        for index, line in enumerate(evidence_lines):
            update_finding(
                rows,
                symptom,
                status,
                line,
                details if index == 0 else {},
            )

        if finding.get("needs_review") is True:
            rows[symptom]["needs_review"] = True

    return rows


# ============================================================
# PUBLIC API
# ============================================================

def analyze_live(lines):
    """
    Return consolidated symptom findings.
    """
    transcript = normalize_lines(lines)

    if not transcript:
        return []

    rows = local_extract(transcript)

    ai_findings = call_ai_extractor(transcript)

    if ai_findings is not None:
        rows = merge_ai_findings(
            rows,
            ai_findings,
            transcript,
        )

    reported_symptoms = [
        symptom
        for symptom, row in rows.items()
        if (
            row["speaker_role"] == "Patient"
            and row["status"] == "Reported"
            and symptom != "temperature"
        )
    ]

    for symptom, row in rows.items():
        if (
            row["speaker_role"] == "Patient"
            and row["status"] == "Reported"
        ):
            row["associated_symptoms"] = [
                other
                for other in reported_symptoms
                if other != symptom
            ]

        update_measurement(row)

    return list(rows.values())


def aggregate(lines):
    """
    Only verified, review-free,
    patient-reported findings.
    """
    findings = analyze_live(lines)

    return [
        copy.deepcopy(row)
        for row in findings
        if (
            row["speaker_role"] == "Patient"
            and row["status"] == "Reported"
            and row["reported"] == "Yes"
            and row["verified"]
            and not row["needs_review"]
        )
    ]


def summarize_consultation(lines):
    """
    Provisional consultation summary.
    """
    findings = analyze_live(lines)
    summaries = []

    for row in findings:
        if row["speaker_role"] != "Patient":
            continue

        if row["status"] not in (
            "Reported",
            "Denied",
            "Uncertain",
        ):
            continue

        symptom = row["symptom"]

        if row["status"] == "Denied":
            sentence = (
                f"Patient denies {symptom}."
            )

        elif row["status"] == "Uncertain":
            sentence = (
                f"Patient is uncertain about {symptom}."
            )

        elif symptom == "temperature":
            sentence = (
                "Patient-reported temperature: "
                f"{row['temperature'] or 'not captured'}."
            )

        else:
            sentence = (
                f"Patient reports {symptom}."
            )

            for field, label in (
                ("location", "Location"),
                ("duration", "Duration"),
                ("frequency", "Frequency"),
                ("severity", "Severity"),
                ("quality", "Quality"),
                ("onset", "Onset"),
            ):
                value = row.get(field)

                if value:
                    sentence += (
                        f" {label}: {value}."
                    )

        if row["needs_review"]:
            sentence += (
                " Requires clinician review."
            )

        if not row["verified"]:
            sentence += " (Unverified)"

        summaries.append(sentence)

    return summaries


# ============================================================
# TEST CASES
# ============================================================

if __name__ == "__main__":
    sample = [
        ("Patient", "Thank you for watching."),
        ("Patient", "I am having fever for one week."),
        ("Patient", "Not one week, two days."),
        ("Doctor", "Do you have headache?"),
        ("Patient", "Yes."),
        ("Doctor", "Which part of your head hurts?"),
        ("Patient", "The right side."),
        ("Doctor", "Do you have any body pain?"),
        ("Patient", "Yes."),
        ("Doctor", "Which part of her body hurts?"),
        ("Patient", "Right time."),
        ("Patient", "Right thigh."),
    ]

    lines = [
        {
            "role": role,
            "text": text,
            "verified": False,
        }
        for role, text in sample
    ]

    print("\nLIVE FINDINGS\n")

    for row in analyze_live(lines):
        print({
            "symptom": row["symptom"],
            "status": row["status"],
            "duration": row["duration"],
            "location": row["location"],
            "needs_review": row["needs_review"],
            "verified": row["verified"],
        })

    print("\nSUMMARY\n")

    for item in summarize_consultation(lines):
        print("-", item)

    print("\nCORRECTION HISTORY\n")

    for row in analyze_live(lines):
        if row["conflicts"]:
            print(
                row["symptom"],
                json.dumps(
                    row["conflicts"],
                    indent=2,
                ),
            )
