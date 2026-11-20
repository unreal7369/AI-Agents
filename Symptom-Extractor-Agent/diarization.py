"""Experimental online two-speaker clustering for simulated conversations.

Not reliable identity recognition or full diarization. Expects single-speaker,
16-kHz float audio turns. Returns (speaker_label, explanation) to audio_engine.py.
"""

import threading
from collections import deque

import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav


class OnlineSpeakers:
    def __init__(
        self,
        similarity_threshold=0.73,
        margin_threshold=0.08,
        new_speaker_threshold=0.72,
        max_speakers=2,
        min_speech_seconds=1.0,
        update_threshold=0.82,
        candidate_similarity=0.80,
        candidate_required=2,
    ):
        """All scores are uncalibrated cosine similarities, not probabilities.

        A second voice requires repeated embeddings that resemble each other and
        are sufficiently distinct from the first voice. Thresholds require
        tuning with consented, hand-labeled test recordings.
        """
        self.similarity_threshold = float(similarity_threshold)
        self.margin_threshold = float(margin_threshold)
        self.new_speaker_threshold = float(new_speaker_threshold)
        self.min_speech_seconds = float(min_speech_seconds)
        self.update_threshold = float(update_threshold)
        self.candidate_similarity = float(candidate_similarity)
        self.candidate_required = max(2, int(candidate_required))
        self.max_speakers = max(1, int(max_speakers))
        self.sample_rate = 16000
        self._encoder = None
        self._profiles = []
        self._counts = []
        self._pending = deque(maxlen=5)
        self._lock = threading.RLock()

    @staticmethod
    def _normalize(x):
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if x.size == 0 or not np.all(np.isfinite(x)):
            return None
        norm = float(np.linalg.norm(x))
        return x / norm if norm > 1e-8 else None

    def _prepare_audio(self, audio):
        wav = np.asarray(audio, dtype=np.float32).reshape(-1)
        if wav.size < self.sample_rate or not np.all(np.isfinite(wav)):
            return None
        if float(np.sqrt(np.mean(wav.astype(np.float64) ** 2))) < 0.001:
            return None
        wav = preprocess_wav(wav, source_sr=self.sample_rate)
        if len(wav) < int(self.min_speech_seconds * self.sample_rate):
            return None
        return wav

    def _update_profile(self, index, emb, score):
        # Avoid profile drift from borderline, noisy or ambiguous segments.
        if score < self.update_threshold:
            return
        weight = min(self._counts[index], 20)
        merged = self._normalize((weight * self._profiles[index] + emb) / (weight + 1))
        if merged is not None:
            self._profiles[index] = merged
            self._counts[index] += 1

    def _check_second_speaker(self, emb, best_score):
        """Collect consistent candidates; never label a lone uncertain turn."""
        if self.max_speakers < 2 or len(self._profiles) >= self.max_speakers:
            return None
        if best_score > self.new_speaker_threshold:
            # Do not regard reasonably similar utterances as a new identity.
            return None
        self._pending.append(emb.copy())
        # Only compare candidates separated in time (separate identify calls).
        candidates = list(self._pending)
        for previous in candidates[:-1]:
            if float(np.dot(previous, emb)) < self.candidate_similarity:
                continue
            center = self._normalize(previous + emb)
            if center is None:
                continue
            separation = float(np.dot(center, self._profiles[0]))
            if separation > self.new_speaker_threshold:
                continue
            self._profiles.append(center)
            self._counts.append(2)
            self._pending.clear()
            return (
                'Speaker 2',
                'Second recurring voice candidate (provisional; confirm against audio)',
            )
        return None

    def identify(self, audio):
        """Return ('Speaker 1'|'Speaker 2'|'Unverified', explanation)."""
        with self._lock:
            wav = self._prepare_audio(audio)
            if wav is None:
                return 'Unverified', 'Insufficient clear speech for voice comparison'
            if self._encoder is None:
                self._encoder = VoiceEncoder(device='cpu')
            emb = self._normalize(self._encoder.embed_utterance(wav))
            if emb is None:
                return 'Unverified', 'Invalid voice embedding'

            if not self._profiles:
                self._profiles.append(emb.copy())
                self._counts.append(1)
                return 'Speaker 1', 'First voice profile (provisional)'

            scores = [float(np.dot(emb, p)) for p in self._profiles]
            ranking = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            best_index = ranking[0]
            best = scores[best_index]
            second = scores[ranking[1]] if len(ranking) > 1 else None

            if second is not None and (best - second) < self.margin_threshold:
                return (
                    'Unverified',
                    f'Ambiguous voice: S1={scores[0]:.2f}, S2={scores[1]:.2f}',
                )

            if best >= self.similarity_threshold:
                self._update_profile(best_index, emb, best)
                return (
                    f'Speaker {best_index + 1}',
                    'Experimental similarity ' +
                    ', '.join(f'S{i + 1}={s:.2f}' for i, s in enumerate(scores)),
                )

            discovered = self._check_second_speaker(emb, best)
            if discovered is not None:
                return discovered

            return (
                'Unverified',
                'No reliable speaker match (' +
                ', '.join(f'S{i + 1}={s:.2f}' for i, s in enumerate(scores)) +
                '); awaiting stronger evidence',
            )

    def reset(self):
        with self._lock:
            self._profiles.clear()
            self._counts.clear()
            self._pending.clear()

    @property
    def speaker_count(self):
        with self._lock:
            return len(self._profiles)
