
"""
Doc AI - Doctor/Patient Speaker Identification

Uses Resemblyzer voice embeddings to:
1. Enroll Doctor and Patient voices.
2. Compare live audio against enrolled voices.
3. Return Doctor, Patient, or Unverified.

Prototype for simulated conversations only.
Not suitable for clinical identification without validation.
"""

import threading

import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav


class SpeakerIdentifier:

    def __init__(
        self,
        similarity_threshold=0.72,
        margin_threshold=0.08,
        sample_rate=16000,
    ):
        self.similarity_threshold = similarity_threshold
        self.margin_threshold = margin_threshold
        self.sample_rate = sample_rate

        self.encoder = None
        self.profiles = {}
        self.lock = threading.RLock()

    # ---------------------------------------
    # ENROLLED VOICES
    # ---------------------------------------

    @property
    def enrolled(self):
        with self.lock:
            return list(self.profiles.keys())

    # ---------------------------------------
    # LOAD VOICE MODEL
    # ---------------------------------------

    def _load_encoder(self):
        if self.encoder is None:
            self.encoder = VoiceEncoder(
                device="cpu"
            )

        return self.encoder

    # ---------------------------------------
    # AUDIO PREPROCESSING
    # ---------------------------------------

    def _prepare_audio(self, audio):
        audio = np.asarray(
            audio,
            dtype=np.float32,
        ).flatten()

        if audio.size == 0:
            raise ValueError("Audio is empty")

        if not np.all(np.isfinite(audio)):
            raise ValueError(
                "Audio contains invalid values"
            )

        audio = preprocess_wav(
            audio,
            source_sr=self.sample_rate,
        )

        if len(audio) < self.sample_rate:
            raise ValueError(
                "Not enough clear speech detected. "
                "Speak for several seconds."
            )

        return audio

    # ---------------------------------------
    # ENROLL DOCTOR / PATIENT
    # ---------------------------------------

    def enroll(self, role, audio):
        if role not in ("Doctor", "Patient"):
            raise ValueError(
                "Role must be Doctor or Patient"
            )

        with self.lock:
            prepared_audio = self._prepare_audio(
                audio
            )

            encoder = self._load_encoder()

            embedding = encoder.embed_utterance(
                prepared_audio
            )

            embedding = np.asarray(
                embedding,
                dtype=np.float32,
            )

            if not np.all(np.isfinite(embedding)):
                raise ValueError(
                    "Invalid voice embedding"
                )

            # Normalize embedding.
            norm = np.linalg.norm(embedding)

            if norm < 1e-8:
                raise ValueError(
                    "Unable to extract voice profile"
                )

            embedding = embedding / norm

            self.profiles[role] = embedding

    # ---------------------------------------
    # SPEAKER CLASSIFICATION
    # ---------------------------------------

    def classify(self, audio):
        """
        Returns:
            (role, explanation)

        role:
            Doctor | Patient | Unverified
        """

        with self.lock:

            required = {"Doctor", "Patient"}

            if not required.issubset(
                self.profiles.keys()
            ):
                return (
                    "Unverified",
                    "Doctor and Patient must be enrolled",
                )

            try:
                prepared_audio = self._prepare_audio(
                    audio
                )

            except ValueError as exc:
                return (
                    "Unverified",
                    str(exc),
                )

            encoder = self._load_encoder()

            embedding = encoder.embed_utterance(
                prepared_audio
            )

            embedding = np.asarray(
                embedding,
                dtype=np.float32,
            )

            norm = np.linalg.norm(embedding)

            if norm < 1e-8 or not np.isfinite(norm):
                return (
                    "Unverified",
                    "Unable to extract speaker embedding",
                )

            embedding = embedding / norm

            # Cosine similarity scores.
            scores = {}

            for role in required:
                profile = self.profiles[role]

                scores[role] = float(
                    np.dot(embedding, profile)
                )

            ranked = sorted(
                scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )

            best_role, best_score = ranked[0]
            second_role, second_score = ranked[1]

            margin = best_score - second_score

            if best_score < self.similarity_threshold:
                return (
                    "Unverified",
                    "Speaker similarity below threshold",
                )

            if margin < self.margin_threshold:
                return (
                    "Unverified",
                    "Doctor and Patient voice scores too close",
                )

            return (
                best_role,
                (
                    f"Voice match: {best_role} | "
                    f"Similarity: {best_score:.2f} | "
                    f"Margin: {margin:.2f}"
                ),
            )
