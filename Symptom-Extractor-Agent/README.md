# Doc AI — Phase 1 Python MVP

Local-only proof of concept for clinician-observed audio transcription and **manual speaker tagging**. This is **not a clinical decision support product**. It does not suggest diagnoses or make clinical decisions. Do not input identifiable patient data in this prototype.

## Setup (Windows PowerShell)

Use Python 3.11 or 3.12 (64-bit), a working microphone, and internet for the initial Whisper model download.

```powershell
cd doc_ai_mvp
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If your installation does not include `py -3.11`, use `python -m venv .venv` with Python 3.11/3.12. To view the local dashboard open http://localhost:8501. On Linux you may need PortAudio installed for `sounddevice`.

## How it works

1. Read the warning and obtain all participants' consent.
2. Check consent; click **Start consultation**.
3. Audio from **the host computer** microphone stays on until you click **Stop consultation** and is chunked in ~3-second windows; transcription may appear later on slower CPUs or during first model download.
4. Assign **Patient** or **Doctor** to each transcript segment. This is manual in Phase 1 because safe speaker identification is not built yet; each chunk may contain both speakers and should only be tagged if attribution is reliable.
5. Read illustrative keyword findings (only Patient-tagged segments contribute).
6. Stop capture, optionally download JSON, and clear the transcript. Audio is not saved.

**Important:** A chunk is a transcription window, not a timed recording. The microphone remains open for the entire consultation, unless manually stopped or a device/system error interrupts capture. Keep the application running and avoid Windows sleep. This uses the server computer's microphone, not the microphone of a remote browser.

## Limits

- Not true word-by-word streaming, but rolling chunk transcription; latency depends on hardware.
- No reliable speaker diarization, patient identity, clinical diagnoses, medical reasoning or EHR integration.
- Hindi/Hinglish matching is incomplete; only a handful of romanized terms are included.
- Basic rules may miss negations or match quoted/hypothetical symptoms. Every finding must be checked against the transcript.
- This is a single-local-user app. Don't deploy as a multiuser service or expose on the public internet.
- Transcript currently remains in Streamlit's session memory until cleared/session ends; downloadable JSON is user-controlled.
- Avoid real clinical use until privacy safeguards, security assessment, regulatory review, and clinically validated models are implemented.

## Next phases

Phase 2: streaming ASR + automatic speaker diarization with confidence/abstention and clinician corrections. Phase 3: clinically validated symptom extraction and evidence-linked, physician-reviewed differential diagnosis. Phase 4: patient consent/auditing/secure storage and EMR integration.

## Run extraction tests

```powershell
python -m unittest discover -s tests -v
```
