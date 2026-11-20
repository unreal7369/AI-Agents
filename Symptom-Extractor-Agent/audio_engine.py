
"""
Doc AI - Continuous Live Medical Audio Engine

Features
--------
- Laptop microphone auto-selection
- Continuous microphone capture
- WAV consultation recording
- Deepgram Nova-3 Medical live WebSocket transcription
- English, Hindi, and Hinglish support
- Provisional transcription events while speaking
- Final transcript events compatible with app.py
- Existing solo / two-person speaker workflow
- Concurrent speaker identification
- Voice activity detection
- Safe worker shutdown and diagnostic warnings

Clinical prototype: obtain consent and review transcripts,
speaker assignments, and clinical findings before use.
"""

import json
import os
import queue
import threading
import time
import traceback
import urllib.parse
import wave

from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
import websocket

from diarization import OnlineSpeakers


MEDICAL_PROMPT = (
    "Doctor-patient medical consultation. "
    "Symptoms and medical terms: fever, headache, "
    "migraine, nausea, vomiting, diarrhea, "
    "constipation, stomach pain, abdominal pain, "
    "body pain, muscle pain, joint pain, "
    "shoulder pain, back pain, neck pain, "
    "chest pain, cough, sore throat, "
    "shortness of breath, breathlessness, "
    "dizziness, vertigo, weakness, fatigue, "
    "palpitations, chills, sweating, "
    "numbness, tingling, swelling, rash, "
    "itching, painful urination, "
    "frequent urination, loss of appetite, "
    "blood pressure, body temperature, "
    "degrees Celsius, degrees Fahrenheit."
)


# ============================================================
# MICROPHONE SELECTION
# ============================================================

def find_laptop_microphone():

    devices = sd.query_devices()

    preferred_names = (
        "microphone array",
        "internal microphone",
        "built-in microphone",
        "integrated microphone",
        "internal mic",
        "built in mic",
        "array microphone",
    )

    excluded_names = (
        "stereo mix",
        "what u hear",
        "loopback",
        "virtual",
        "cable output",
    )

    default_input = sd.default.device[0]

    override = os.getenv(
        "DOC_AI_MIC_DEVICE", ""
    ).strip()

    if override:

        try:
            index = int(override)
            device = sd.query_devices(index)

            if device["max_input_channels"] <= 0:
                raise RuntimeError(
                    "Selected device is not an input device."
                )

            return index, device["name"]

        except ValueError:

            for index, device in enumerate(devices):
                if (
                    device["max_input_channels"] > 0
                    and override.lower()
                    in device["name"].lower()
                ):
                    return index, device["name"]

            raise RuntimeError(
                f"Microphone matching '{override}' not found."
            )

    candidates = []

    for index, device in enumerate(devices):

        if device["max_input_channels"] <= 0:
            continue

        name = device["name"].lower()

        if any(
            item in name for item in excluded_names
        ):
            continue

        if any(
            item in name for item in preferred_names
        ):

            score = 10

            if index == default_input:
                score += 5

            candidates.append(
                (score, index, device["name"])
            )

    if candidates:

        candidates.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        _, index, name = candidates[0]
        return index, name

    if default_input is not None and default_input >= 0:

        device = sd.query_devices(default_input)

        if device["max_input_channels"] > 0:
            return int(default_input), device["name"]

    for index, device in enumerate(devices):

        if (
            device["max_input_channels"] > 0
            and not any(
                item in device["name"].lower()
                for item in excluded_names
            )
        ):
            return index, device["name"]

    raise RuntimeError(
        "No microphone detected. Check Windows "
        "Settings > System > Sound > Input."
    )


# ============================================================
# LIVE TRANSCRIBER
# ============================================================

class LiveTranscriber:

    def __init__(
        self,
        model_name="nova-3-medical",
        language="multi",
    ):

        # Audio: 16 kHz, mono, 100 ms frames.
        self.sample_rate = 16000
        self.frame_samples = 1600

        self.model_name = model_name
        self.language = language

        self.microphone_id = None
        self.microphone_name = None

        # Queues
        self.events = queue.Queue()
        self.audio_queue = queue.Queue(maxsize=500)
        self.recording_queue = queue.Queue(maxsize=1000)

        # Retained for compatibility with older app code.
        # Streaming transcription does not consume work_queue.
        self.work_queue = queue.Queue(maxsize=32)

        self.streaming_queue = queue.Queue(maxsize=500)
        self.speaker_queue = queue.Queue(maxsize=32)

        # Recording state
        self.recording_thread = None
        self.recording_path = None
        self._recording_drops = 0

        # Consultation state
        self.stop_event = threading.Event()
        self.speakers = OnlineSpeakers()

        self._mode = "two_person"
        self._test_role = "Patient"

        self._lock = threading.Lock()
        self._running = False
        self._threads = []
        self._finished = 0

        self._next_turn_id = 0
        self._turn_state = {}

        # Diagnostics
        self._overflow_count = 0
        self._dropped_audio = 0
        self._dropped_asr = 0
        self._dropped_speaker = 0
        self._dropped_streaming = 0

        # Voice activity detection
        self.min_rms = 0.001
        self.noise_multiplier = 1.5

        self.min_speech_seconds = 0.15
        self.end_silence_seconds = 0.35
        self.max_turn_seconds = 3.5

        self.pre_roll_seconds = 0.3
        self.start_speech_seconds = 0.1

        # Preserve older configuration attributes.
        self._model = None
        self._model_name_loaded = None

        self.no_speech_threshold = 0.5
        self.log_prob_threshold = -0.8
        self.compression_ratio_threshold = 2.4

        self.medical_prompt = MEDICAL_PROMPT
        self.beam_size = 1
        self.vad_filter = True

        self.device = os.getenv(
            "DOC_AI_DEVICE", "cpu"
        )

        self.compute_type = os.getenv(
            "DOC_AI_COMPUTE_TYPE",
            (
                "int8"
                if self.device == "cpu"
                else "float16"
            ),
        )

        self.cpu_threads = int(
            os.getenv("DOC_AI_CPU_THREADS", "4")
        )

    # ========================================================
    # BASIC STATE
    # ========================================================

    @property
    def active(self):

        with self._lock:
            return self._running

    def _emit(self, kind, value):
        self.events.put((kind, value))

    # ========================================================
    # CONSULTATION MODE
    # ========================================================

    def configure_mode(
        self,
        mode,
        role="Patient",
    ):

        if mode not in ("solo", "two_person"):
            raise ValueError(
                "Invalid consultation mode"
            )

        if role not in ("Doctor", "Patient"):
            raise ValueError(
                "Invalid consultation role"
            )

        with self._lock:

            if self._running:
                raise RuntimeError(
                    "Cannot change mode during consultation"
                )

            self._mode = mode
            self._test_role = role

    def set_test_role(self, role):

        if role not in ("Doctor", "Patient"):
            raise ValueError(
                "Role must be Doctor or Patient"
            )

        with self._lock:
            self._test_role = role

    def _role_snapshot(self):

        with self._lock:

            if self._mode == "solo":
                return self._test_role

        return None

    # ========================================================
    # START CONSULTATION
    # ========================================================

    def start(self):

        with self._lock:

            if self._running:
                return

            if any(
                thread.is_alive()
                for thread in self._threads
            ):
                raise RuntimeError(
                    "Previous consultation is still closing"
                )

            self._running = True
            self._finished = 0

            self._next_turn_id = 0
            self._turn_state.clear()

            self._overflow_count = 0
            self._dropped_audio = 0
            self._dropped_asr = 0
            self._dropped_speaker = 0
            self._dropped_streaming = 0

        try:

            self.stop_event.clear()

            for target_queue in (
                self.audio_queue,
                self.work_queue,
                self.speaker_queue,
                self.streaming_queue,
                self.recording_queue,
            ):

                while True:
                    try:
                        target_queue.get_nowait()
                    except queue.Empty:
                        break

            self.speakers.reset()

            self.recording_path = None
            self._recording_drops = 0

            self._threads = [

                threading.Thread(
                    target=self._run_guarded,
                    args=(self._asr_loop, "Deepgram"),
                    name="DocAI-Deepgram",
                    daemon=True,
                ),

                threading.Thread(
                    target=self._run_guarded,
                    args=(self._speaker_loop, "Speaker"),
                    name="DocAI-Speakers",
                    daemon=True,
                ),

                threading.Thread(
                    target=self._run_guarded,
                    args=(self._capture_loop, "Microphone"),
                    name="DocAI-Microphone",
                    daemon=True,
                ),
            ]

            self.recording_thread = threading.Thread(
                target=self._recording_loop,
                name="DocAI-Recording",
                daemon=True,
            )

            self.recording_thread.start()

            for thread in self._threads:
                thread.start()

            self._emit(
                "status",
                "Starting microphone and continuous "
                "Deepgram transcription...",
            )

        except Exception:

            self.stop_event.set()

            with self._lock:
                self._running = False

            raise

    # ========================================================
    # STOP CONSULTATION
    # ========================================================

    def stop(self):

        if self.active:

            self.stop_event.set()

            self._emit(
                "status",
                "Stopping microphone; finalizing "
                "pending transcription...",
            )

    # ========================================================
    # WORKER MANAGEMENT
    # ========================================================

    def _run_guarded(self, callback, name):

        try:
            callback()

        except Exception as exc:

            traceback.print_exc()

            self._emit(
                "error",
                f"{name}: {type(exc).__name__}: {exc}",
            )

            self.stop_event.set()

        finally:

            with self._lock:

                self._finished += 1
                final = self._finished == 3

                if final:
                    self._running = False
                    self._turn_state.clear()

            if final:

                if self._overflow_count:
                    self._emit(
                        "warning",
                        "Audio input warnings: "
                        f"{self._overflow_count}",
                    )

                if any((
                    self._dropped_audio,
                    self._dropped_asr,
                    self._dropped_speaker,
                    self._dropped_streaming,
                )):
                    self._emit(
                        "warning",
                        "Dropped: "
                        f"{self._dropped_audio} mic blocks, "
                        f"{self._dropped_asr} ASR turns, "
                        f"{self._dropped_speaker} speaker turns, "
                        f"{self._dropped_streaming} streaming frames",
                    )

                self._emit(
                    "partial_transcript",
                    {"text": ""},
                )

                self._emit(
                    "status",
                    "Consultation stopped. "
                    "Transcription complete.",
                )

    # ========================================================
    # WAV RECORDING
    # ========================================================

    def _recording_loop(self):

        try:

            folder = Path("consultation_recordings")
            folder.mkdir(
                parents=True,
                exist_ok=True,
            )

            filename = datetime.now().strftime(
                "consultation_%Y%m%d_%H%M%S.wav"
            )

            self.recording_path = str(
                folder / filename
            )

            with wave.open(
                self.recording_path, "wb"
            ) as wav:

                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(self.sample_rate)

                while True:

                    block = self.recording_queue.get()

                    if block is None:
                        break

                    pcm = (
                        np.clip(block, -1.0, 1.0)
                        * 32767.0
                    ).astype("<i2")

                    wav.writeframes(pcm.tobytes())

            self._emit(
                "status",
                f"Audio saved: {self.recording_path}",
            )

        except Exception as exc:

            traceback.print_exc()

            self._emit(
                "error",
                f"Audio recording failed: {exc}",
            )

    # ========================================================
    # MICROPHONE CALLBACK
    # ========================================================

    def _callback(
        self,
        indata,
        frames,
        time_info,
        status,
    ):

        if status:

            self._overflow_count += 1

            if self._overflow_count in (1, 10, 50):
                self._emit(
                    "warning",
                    f"Audio device: {status}",
                )

        if self.stop_event.is_set():
            return

        audio = indata[:, 0].copy()

        try:
            self.recording_queue.put_nowait(audio)

        except queue.Full:

            self._recording_drops += 1

            if self._recording_drops == 1:
                self._emit(
                    "warning",
                    "Recording buffer overflow: "
                    "audio may be missing.",
                )

        try:
            self.audio_queue.put_nowait(audio)

        except queue.Full:

            self._dropped_audio += 1

            if self._dropped_audio in (1, 10, 50):
                self._emit(
                    "warning",
                    "Microphone queue full. "
                    "Some audio was dropped.",
                )

    # ========================================================
    # AUDIO VALIDATION
    # ========================================================

    def _audio_rms(self, audio):

        audio = np.asarray(
            audio,
            dtype=np.float32,
        ).reshape(-1)

        if not audio.size:
            return 0.0

        if not np.all(np.isfinite(audio)):
            return 0.0

        return float(
            np.sqrt(
                np.mean(
                    np.square(
                        audio.astype(np.float64)
                    )
                )
            )
        )

    def _valid_audio(self, audio):

        audio = np.asarray(
            audio,
            dtype=np.float32,
        ).reshape(-1)

        if audio.size < int(
            self.sample_rate * 0.25
        ):
            return False

        if not np.all(np.isfinite(audio)):
            return False

        if self._audio_rms(audio) < self.min_rms:
            return False

        return True

    # ========================================================
    # REGISTER SPEECH TURN
    # ========================================================

    def _put_turn(
        self,
        audio,
        role_at_start=None,
    ):

        audio = np.asarray(
            audio,
            dtype=np.float32,
        ).reshape(-1)

        if not self._valid_audio(audio):
            return None

        audio = audio.copy()

        with self._lock:

            turn_id = self._next_turn_id
            self._next_turn_id += 1

            solo = self._mode == "solo"

            self._turn_state[turn_id] = {

                "published": False,

                "speaker": (
                    (
                        "Unverified",
                        "Solo demo: role selected manually",
                    )
                    if solo
                    else None
                ),

                "speaker_done": solo,

                "role_at_capture": (
                    role_at_start if solo else None
                ),

                "queued_at": time.monotonic(),
            }

        # Continuous streaming handles transcription.
        # Only speaker analysis uses completed audio turns.

        if solo:
            return turn_id

        try:

            self.speaker_queue.put_nowait(
                (turn_id, audio)
            )

        except queue.Full:

            self._dropped_speaker += 1

            with self._lock:

                record = self._turn_state.get(
                    turn_id
                )

                if record is not None:
                    record["speaker_done"] = True

            self._emit(
                "warning",
                "Speaker analysis backlog full.",
            )

        return turn_id

    # ========================================================
    # MICROPHONE CAPTURE AND VAD
    # ========================================================

    def _capture_loop(self):

        frame_seconds = (
            self.frame_samples / self.sample_rate
        )

        min_speech_frames = max(
            1,
            round(
                self.min_speech_seconds
                / frame_seconds
            ),
        )

        silence_frames_to_end = max(
            1,
            round(
                self.end_silence_seconds
                / frame_seconds
            ),
        )

        max_turn_frames = max(
            1,
            round(
                self.max_turn_seconds
                / frame_seconds
            ),
        )

        start_frames = max(
            1,
            round(
                self.start_speech_seconds
                / frame_seconds
            ),
        )

        pre_roll_frames = max(
            start_frames + 1,
            round(
                self.pre_roll_seconds
                / frame_seconds
            ),
        )

        noise_floor = 0.001

        pre_roll = deque(
            maxlen=pre_roll_frames
        )

        current = []

        speech_frames = 0
        quiet_frames = 0
        start_count = 0

        talking = False
        role_at_start = None

        pending = np.empty(
            0,
            dtype=np.float32,
        )

        last_level_status = time.monotonic()
        highest_rms = 0.0

        def stream_frame(frame):

            try:
                self.streaming_queue.put_nowait(
                    ("audio", frame.copy())
                )

            except queue.Full:

                self._dropped_streaming += 1

                if self._dropped_streaming in (1, 10, 50):
                    self._emit(
                        "warning",
                        "Deepgram streaming queue is full.",
                    )

        def send_end_event(turn_id):

            try:
                self.streaming_queue.put(
                    ("end", turn_id),
                    timeout=0.5,
                )

            except queue.Full:
                self._emit(
                    "warning",
                    "Could not queue Deepgram "
                    "turn boundary.",
                )

        def finish_turn():

            nonlocal current
            nonlocal speech_frames
            nonlocal quiet_frames
            nonlocal start_count
            nonlocal talking
            nonlocal role_at_start

            turn_id = None

            if (
                current
                and speech_frames >= min_speech_frames
            ):

                audio = np.concatenate(current)

                turn_id = self._put_turn(
                    audio,
                    role_at_start,
                )

            # Always close the stream segment,
            # even if the local VAD rejected it.
            send_end_event(turn_id)

            current = []
            speech_frames = 0
            quiet_frames = 0
            start_count = 0

            talking = False
            role_at_start = None

        def process_block(block):

            nonlocal pending
            nonlocal current
            nonlocal speech_frames
            nonlocal quiet_frames
            nonlocal start_count
            nonlocal talking
            nonlocal role_at_start
            nonlocal noise_floor
            nonlocal last_level_status
            nonlocal highest_rms

            block = np.asarray(
                block,
                dtype=np.float32,
            ).reshape(-1)

            if pending.size:

                block = np.concatenate(
                    (pending, block)
                )

            full = (
                len(block) // self.frame_samples
            ) * self.frame_samples

            pending = block[full:].copy()

            for offset in range(
                0,
                full,
                self.frame_samples,
            ):

                frame = block[
                    offset:
                    offset + self.frame_samples
                ].copy()

                if not np.all(np.isfinite(frame)):
                    continue

                rms = self._audio_rms(frame)

                highest_rms = max(
                    highest_rms, rms
                )

                now = time.monotonic()

                if now - last_level_status >= 5:

                    if (
                        highest_rms < 0.001
                        and talking
                    ):
                        self._emit(
                            "warning",
                            "Very low microphone input. "
                            "Check microphone volume.",
                        )

                    last_level_status = now
                    highest_rms = 0.0

                threshold = max(
                    self.min_rms,
                    noise_floor
                    * self.noise_multiplier,
                )

                speech = rms >= threshold

                if not talking:

                    if rms < threshold:

                        noise_floor = (
                            0.97 * noise_floor
                            + 0.03 * rms
                        )

                    pre_roll.append(frame)

                    if speech:
                        start_count += 1
                    else:
                        start_count = 0

                    if start_count >= start_frames:

                        talking = True

                        role_at_start = (
                            self._role_snapshot()
                        )

                        current = list(pre_roll)

                        for buffered_frame in current:
                            stream_frame(
                                buffered_frame
                            )

                        pre_roll.clear()

                        speech_frames = start_count
                        quiet_frames = 0

                else:

                    current.append(frame)

                    # Send speech immediately.
                    stream_frame(frame)

                    if speech:
                        speech_frames += 1
                        quiet_frames = 0
                    else:
                        quiet_frames += 1

                    if (
                        quiet_frames
                        >= silence_frames_to_end
                        or len(current)
                        >= max_turn_frames
                    ):
                        finish_turn()

        try:

            mic_id, mic_name = (
                find_laptop_microphone()
            )

            self.microphone_id = mic_id
            self.microphone_name = mic_name

            self._emit(
                "status",
                f"Selected microphone: {mic_name}",
            )

            with sd.InputStream(
                device=mic_id,
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self.frame_samples,
                latency="low",
                callback=self._callback,
            ):

                self._emit(
                    "status",
                    f"Microphone open: {mic_name}. "
                    "Continuous speech streaming enabled.",
                )

                while not self.stop_event.is_set():

                    try:
                        block = self.audio_queue.get(
                            timeout=0.1
                        )

                    except queue.Empty:
                        continue

                    process_block(block)

            # Drain microphone frames already captured.
            while True:

                try:
                    block = (
                        self.audio_queue.get_nowait()
                    )

                except queue.Empty:
                    break

                process_block(block)

            if talking:

                if pending.size:

                    current.append(
                        pending.copy()
                    )

                    stream_frame(
                        pending.copy()
                    )

                finish_turn()

        finally:

            self._send_sentinel(
                self.streaming_queue,
                "DocAI-Deepgram",
            )

            self._send_sentinel(
                self.speaker_queue,
                "DocAI-Speakers",
            )

            self._emit(
                "status",
                "Microphone closed. "
                "Finalizing pending turns...",
            )

            if self.recording_thread is not None:

                try:
                    self.recording_queue.put(
                        None,
                        timeout=5.0,
                    )
                except queue.Full:
                    self._emit(
                        "warning",
                        "Recording queue could not "
                        "receive its shutdown marker.",
                    )

                self.recording_thread.join(
                    timeout=10
                )

                if self.recording_thread.is_alive():
                    self._emit(
                        "warning",
                        "Audio recording has not "
                        "finished saving.",
                    )

    # ========================================================
    # WORKER SHUTDOWN
    # ========================================================

    def _send_sentinel(
        self,
        target_queue,
        worker_name,
    ):

        while True:

            worker = next(
                (
                    thread
                    for thread in self._threads
                    if thread.name == worker_name
                ),
                None,
            )

            if (
                worker is None
                or not worker.is_alive()
            ):
                return

            try:

                target_queue.put(
                    None,
                    timeout=0.5,
                )

                return

            except queue.Full:
                continue

    # ========================================================
    # DEEPGRAM CONTINUOUS STREAMING
    # ========================================================

    def _asr_loop(self):

        api_key = os.getenv(
            "DEEPGRAM_API_KEY", ""
        ).strip()

        if not api_key:
            raise RuntimeError(
                "DEEPGRAM_API_KEY is missing. "
                "Set it before starting Streamlit."
            )

        params = [
            ("model", self.model_name),
            ("language", self.language),
            ("encoding", "linear16"),
            ("sample_rate", str(self.sample_rate)),
            ("channels", "1"),
            ("smart_format", "true"),
            ("punctuate", "true"),
            ("interim_results", "true"),
            ("endpointing", "300"),
        ]

        url = (
            "wss://api.deepgram.com/v1/listen?"
            + urllib.parse.urlencode(params)
        )

        ws = None

        final_parts = []
        interim_text = ""
        uncertain_segments = 0

        last_sent_at = time.monotonic()
        last_preview = ""
        last_preview_at = 0.0

        turn_broken = False
        turn_has_audio = False

        def show_preview(text, force=False):

            nonlocal last_preview
            nonlocal last_preview_at

            text = str(text or "").strip()
            now = time.monotonic()

            if text == last_preview:
                return

            if (
                not force
                and now - last_preview_at < 0.15
            ):
                return

            self._emit(
                "partial_transcript",
                {"text": text},
            )

            last_preview = text
            last_preview_at = now

        def close_connection():

            nonlocal ws

            if ws is None:
                return

            old_ws = ws
            ws = None

            try:
                old_ws.send(
                    json.dumps({
                        "type": "CloseStream"
                    })
                )
            except Exception:
                pass

            try:
                old_ws.close()
            except Exception:
                pass

        def ensure_connection():

            nonlocal ws
            nonlocal last_sent_at

            if ws is None:

                ws = websocket.create_connection(
                    url,
                    header=[
                        f"Authorization: Token {api_key}"
                    ],
                    timeout=8,
                )

                ws.settimeout(0.01)

                last_sent_at = time.monotonic()

            return ws

        def receive_one(timeout=0.005):

            nonlocal interim_text
            nonlocal uncertain_segments

            if ws is None:
                return False

            ws.settimeout(timeout)

            try:
                message = ws.recv()

            except websocket.WebSocketTimeoutException:
                return False

            if not message:
                raise RuntimeError(
                    "Deepgram WebSocket disconnected."
                )

            if isinstance(message, bytes):
                message = message.decode(
                    "utf-8",
                    errors="replace",
                )

            result = json.loads(message)

            message_type = result.get("type")

            if message_type == "Error":

                raise RuntimeError(
                    result.get("description")
                    or result.get("message")
                    or "Deepgram streaming error"
                )

            if message_type != "Results":
                return False

            alternatives = (
                result.get("channel", {})
                .get("alternatives", [])
            )

            best = (
                alternatives[0]
                if alternatives
                else {}
            )

            segment_text = str(
                best.get("transcript") or ""
            ).strip()

            if result.get("is_final"):

                if segment_text:
                    final_parts.append(
                        segment_text
                    )

                interim_text = ""

                confidence = best.get(
                    "confidence"
                )

                if (
                    confidence is not None
                    and float(confidence) < 0.5
                ):
                    uncertain_segments = 1

            else:

                interim_text = segment_text

            preview = " ".join(
                final_parts
                + (
                    [interim_text]
                    if interim_text
                    else []
                )
            ).strip()

            show_preview(
                preview,
                force=bool(
                    result.get("is_final")
                ),
            )

            # Deepgram acknowledges Finalize
            # using from_finalize on a Results event.
            return bool(
                result.get("from_finalize")
            )

        def publish_turn(
            turn_id,
            started,
            text,
        ):

            finished = time.monotonic()

            asr_seconds = round(
                finished - started,
                2,
            )

            with self._lock:

                record = self._turn_state.get(
                    turn_id
                )

                if record is None:
                    return

                if not text:

                    record["published"] = True

                    if record.get("speaker_done"):
                        self._turn_state.pop(
                            turn_id,
                            None,
                        )

                    return

                known_speaker = record.get(
                    "speaker"
                )

                if known_speaker:
                    speaker, note = known_speaker

                else:
                    speaker = "Unverified"
                    note = "Speaker analysis pending"

                queued_at = record.get(
                    "queued_at",
                    started,
                )

                queue_seconds = round(
                    max(
                        0.0,
                        started - queued_at
                    ),
                    2,
                )

                total_seconds = round(
                    max(
                        0.0,
                        finished - queued_at
                    ),
                    2,
                )

                # Existing app.py-compatible event.
                self._emit(
                    "transcript",
                    {
                        "turn_id": turn_id,
                        "text": text,
                        "speaker": speaker,
                        "speaker_note": note,
                        "asr_seconds": asr_seconds,
                        "role_at_capture": record.get(
                            "role_at_capture"
                        ),
                        "queue_seconds": queue_seconds,
                        "total_seconds": total_seconds,
                    },
                )

                record["published"] = True

                if record.get("speaker_done"):
                    self._turn_state.pop(
                        turn_id,
                        None,
                    )

        def reset_turn():

            nonlocal interim_text
            nonlocal uncertain_segments
            nonlocal turn_broken
            nonlocal turn_has_audio

            final_parts.clear()
            interim_text = ""
            uncertain_segments = 0

            turn_broken = False
            turn_has_audio = False

            show_preview(
                "",
                force=True,
            )

        self._emit(
            "status",
            f"Deepgram {self.model_name} "
            f"({self.language}) live streaming ready.",
        )

        try:

            while True:

                try:

                    event = self.streaming_queue.get(
                        timeout=0.035
                    )

                except queue.Empty:

                    if ws is not None:

                        try:

                            receive_one(0.008)

                            if (
                                time.monotonic()
                                - last_sent_at >= 4.0
                            ):

                                ws.send(
                                    json.dumps({
                                        "type": "KeepAlive"
                                    })
                                )

                                last_sent_at = (
                                    time.monotonic()
                                )

                        except Exception as exc:

                            self._emit(
                                "warning",
                                "Deepgram streaming "
                                f"connection failed: {exc}",
                            )

                            close_connection()
                            turn_broken = True

                            show_preview(
                                "",
                                force=True,
                            )

                    continue

                if event is None:
                    break

                event_type, payload = event

                # ----------------------------------------
                # SEND AUDIO WHILE SPEAKING
                # ----------------------------------------

                if event_type == "audio":

                    if turn_broken:
                        continue

                    try:

                        connection = (
                            ensure_connection()
                        )

                        pcm = np.clip(
                            np.asarray(
                                payload,
                                dtype=np.float32,
                            ),
                            -1.0,
                            1.0,
                        )

                        pcm_bytes = (
                            (pcm * 32767.0)
                            .astype("<i2")
                            .tobytes()
                        )

                        connection.send(
                            pcm_bytes,
                            opcode=(
                                websocket.ABNF
                                .OPCODE_BINARY
                            ),
                        )

                        turn_has_audio = True
                        last_sent_at = (
                            time.monotonic()
                        )

                        # Receive interim results
                        # during the active conversation.
                        for _ in range(4):

                            if receive_one(0.003):
                                break

                    except Exception as exc:

                        self._emit(
                            "warning",
                            "Deepgram live audio "
                            f"failed: {exc}",
                        )

                        turn_broken = True

                        close_connection()

                        show_preview(
                            "",
                            force=True,
                        )

                # ----------------------------------------
                # FINALIZE COMPLETED SPEECH TURN
                # ----------------------------------------

                elif event_type == "end":

                    turn_id = payload
                    started = time.monotonic()

                    if (
                        ws is not None
                        and turn_has_audio
                        and not turn_broken
                    ):

                        try:

                            ws.send(
                                json.dumps({
                                    "type": "Finalize"
                                })
                            )

                            last_sent_at = (
                                time.monotonic()
                            )

                            acknowledged = False

                            deadline = (
                                time.monotonic()
                                + 2.0
                            )

                            while (
                                time.monotonic()
                                < deadline
                            ):

                                if receive_one(0.08):
                                    acknowledged = True
                                    break

                            if not acknowledged:

                                self._emit(
                                    "warning",
                                    "Deepgram turn "
                                    f"{turn_id}: "
                                    "finalization timed out.",
                                )

                                # Prevent late results from
                                # being assigned to a new turn.
                                close_connection()

                        except Exception as exc:

                            self._emit(
                                "warning",
                                "Deepgram finalization "
                                f"failed: {exc}",
                            )

                            close_connection()

                    text = " ".join(
                        final_parts
                    ).strip()

                    if turn_id is not None:

                        publish_turn(
                            turn_id,
                            started,
                            text,
                        )

                        if uncertain_segments:
                            self._emit(
                                "warning",
                                "Transcript turn "
                                f"{turn_id} may contain "
                                "uncertain words.",
                            )

                    reset_turn()

        finally:

            close_connection()

            show_preview(
                "",
                force=True,
            )

    # ========================================================
    # SPEAKER IDENTIFICATION
    # ========================================================

    def _speaker_loop(self):

        while True:

            task = self.speaker_queue.get()

            if task is None:
                return

            turn_id, audio = task

            try:

                speaker, note = (
                    self.speakers.identify(
                        audio
                    )
                )

            except Exception as exc:

                traceback.print_exc()

                speaker = "Unverified"

                note = (
                    "Voice analysis failed: "
                    f"{exc}"
                )

                self._emit(
                    "warning",
                    note,
                )

                # Do not stop microphone recording
                # because speaker analysis failed.

            with self._lock:

                record = self._turn_state.get(
                    turn_id
                )

                if record is None:
                    continue

                record["speaker"] = (
                    speaker,
                    note,
                )

                record["speaker_done"] = True

                already_published = (
                    record["published"]
                )

                if already_published:
                    self._turn_state.pop(
                        turn_id,
                        None,
                    )

            if already_published:

                self._emit(
                    "speaker_update",
                    {
                        "turn_id": turn_id,
                        "speaker": speaker,
                        "speaker_note": note,
                    },
                )
