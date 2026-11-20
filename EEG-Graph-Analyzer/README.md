# EEG Clinical Analysis Agent — Local Windows Prototype

A local prototype for analyzing EEG waveform screenshots (PNG/JPG) and producing structured, evidence-backed explanations.

IMPORTANT: This is research/clinical decision-support software, NOT a diagnostic device. It must not be used as the sole basis for clinical decisions. A qualified EEG/neurology professional must review the original EEG and clinical context.

## Quick start
1. Double-click install.bat
2. Double-click start.bat
3. Open http://127.0.0.1:8000
4. Upload a clear EEG PNG/JPG
5. Click Analyze EEG

Python 3.10+ is recommended. No Node.js is required for this prototype.

## Optional LLM
Edit .env after installation:
OPENAI_API_KEY=your_key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini

If no key is configured, the app runs in clearly labeled demo fallback mode.

## RAG knowledge base
Add expert-reviewed .txt or .md EEG references to backend/knowledge_base/. The local TF-IDF retriever indexes them automatically on startup. Replace the demo reference with approved sources before serious evaluation.

## API
GET /api/health
POST /api/analyze (multipart field: file)
GET /docs
