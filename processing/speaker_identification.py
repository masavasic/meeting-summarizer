from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torchaudio

from speechbrain.inference.speaker import SpeakerRecognition
from speechbrain.utils.fetching import LocalStrategy


SPEECHBRAIN_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
TARGET_SAMPLE_RATE = 16000

SUPPORTED_PROFILE_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".m4a",
}


class SpeakerIdentifier:
    """
    Offline identifikacija poznatih govornika.

    Struktura foldera:
        speaker_profiles/
            ImeOsobe1/
                sample1.wav
                sample2.wav
            ImeOsobe2/
                sample1.wav

    Imena govornika se NE hardkoduju u kodu.
    Ime svakog podfoldera predstavlja ime/profil osobe.
    """

    def __init__(
        self,
        profiles_dir: str | Path,
        similarity_threshold: float = 0.55,
        min_segment_duration: float = 1.0,
        max_segments_per_speaker: int = 8,
        max_profile_files_per_speaker: int = 10,
    ) -> None:
        self.profiles_dir = Path(profiles_dir)
        self.similarity_threshold = float(similarity_threshold)
        self.min_segment_duration = float(min_segment_duration)
        self.max_segments_per_speaker = int(max_segments_per_speaker)
        self.max_profile_files_per_speaker = int(
            max_profile_files_per_speaker
        )

        self.device = (
            "cuda:0"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            "Učitavanje SpeechBrain ECAPA modela "
            "za speaker identification..."
        )
        print(f"Speaker identification koristi: {self.device}")

        self.model = SpeakerRecognition.from_hparams(
            source=SPEECHBRAIN_MODEL,
            savedir="models/spkrec-ecapa-voxceleb",
            run_opts={
                "device": self.device,
            },
            local_strategy=LocalStrategy.COPY,
        )

        print("SpeechBrain ECAPA model je učitan.")

        self.reference_embeddings: dict[str, torch.Tensor] = {}

    # =========================================================
    # AUDIO
    # =========================================================

    def _load_audio(
        self,
        audio_path: str | Path,
    ) -> tuple[torch.Tensor, int]:

        audio_path = Path(audio_path)

        if not audio_path.is_file():
            raise FileNotFoundError(
                f"Audio fajl nije pronađen: {audio_path}"
            )

        waveform, sample_rate = torchaudio.load(
            str(audio_path)
        )

        waveform = waveform.float()

        # Stereo/multichannel -> mono.
        if waveform.shape[0] > 1:
            waveform = waveform.mean(
                dim=0,
                keepdim=True,
            )

        return waveform, int(sample_rate)

    def _prepare_waveform(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
    ) -> torch.Tensor:

        waveform = waveform.float()

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(
                dim=0,
                keepdim=True,
            )

        if sample_rate != TARGET_SAMPLE_RATE:
            waveform = torchaudio.functional.resample(
                waveform,
                sample_rate,
                TARGET_SAMPLE_RATE,
            )

        return waveform.contiguous()

    # =========================================================
    # EMBEDDING
    # =========================================================

    @staticmethod
    def _normalize_embedding(
        embedding: torch.Tensor,
    ) -> torch.Tensor:

        embedding = (
            embedding
            .float()
            .reshape(-1)
        )

        norm = torch.linalg.vector_norm(
            embedding
        )

        if float(norm) > 0.0:
            embedding = embedding / norm

        return embedding.cpu()

    def _extract_embedding(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
    ) -> torch.Tensor:

        waveform = self._prepare_waveform(
            waveform,
            sample_rate,
        )

        if waveform.numel() == 0:
            raise ValueError(
                "Nije moguće napraviti embedding iz praznog audio segmenta."
            )

        with torch.no_grad():
            embedding = self.model.encode_batch(
                waveform
            )

        return self._normalize_embedding(
            embedding.squeeze()
        )

    @staticmethod
    def _mean_embedding(
        embeddings: Iterable[torch.Tensor],
    ) -> torch.Tensor:

        embeddings = list(embeddings)

        if not embeddings:
            raise ValueError(
                "Nema embedding-a za računanje proseka."
            )

        centroid = torch.stack(
            embeddings,
            dim=0,
        ).mean(dim=0)

        norm = torch.linalg.vector_norm(
            centroid
        )

        if float(norm) > 0.0:
            centroid = centroid / norm

        return centroid.cpu()

    @staticmethod
    def _cosine_similarity(
        first: torch.Tensor,
        second: torch.Tensor,
    ) -> float:

        first = first.reshape(-1)
        second = second.reshape(-1)

        denominator = (
            torch.linalg.vector_norm(first)
            * torch.linalg.vector_norm(second)
        )

        if float(denominator) == 0.0:
            return -1.0

        score = torch.dot(
            first,
            second,
        ) / denominator

        return float(score)

    # =========================================================
    # REFERENTNI PROFILI
    # =========================================================

    def _profile_audio_files(
        self,
        speaker_directory: Path,
    ) -> list[Path]:

        files = [
            path
            for path in speaker_directory.iterdir()
            if (
                path.is_file()
                and path.suffix.lower()
                in SUPPORTED_PROFILE_EXTENSIONS
            )
        ]

        return sorted(files)

    def load_profiles(self) -> dict[str, torch.Tensor]:
        """
        Automatski učitava sve podfoldere iz profiles_dir.

        Na primer:
            speaker_profiles/Masa/*.wav
            speaker_profiles/Ana/*.wav

        Nema liste imena u kodu.
        """

        if not self.profiles_dir.is_dir():
            raise FileNotFoundError(
                "Folder sa speaker profilima ne postoji: "
                f"{self.profiles_dir}"
            )

        speaker_directories = sorted(
            path
            for path in self.profiles_dir.iterdir()
            if path.is_dir()
        )

        if not speaker_directories:
            raise ValueError(
                "Folder sa speaker profilima nema nijedan podfolder."
            )

        profiles: dict[str, torch.Tensor] = {}

        for speaker_directory in speaker_directories:
            speaker_name = speaker_directory.name.strip()

            if not speaker_name:
                continue

            audio_files = self._profile_audio_files(
                speaker_directory
            )

            if not audio_files:
                print(
                    "Preskačem profil bez audio fajlova: "
                    f"{speaker_directory}"
                )
                continue

            audio_files = audio_files[
                :self.max_profile_files_per_speaker
            ]

            embeddings: list[torch.Tensor] = []

            print(
                f"Profil '{speaker_name}': "
                f"{len(audio_files)} referentnih snimaka"
            )

            for audio_file in audio_files:
                try:
                    waveform, sample_rate = (
                        self._load_audio(
                            audio_file
                        )
                    )

                    duration = (
                        waveform.shape[1]
                        / sample_rate
                    )

                    if duration < self.min_segment_duration:
                        print(
                            "  Preskačem prekratak referentni snimak: "
                            f"{audio_file.name} ({duration:.2f}s)"
                        )
                        continue

                    embedding = self._extract_embedding(
                        waveform,
                        sample_rate,
                    )

                    embeddings.append(
                        embedding
                    )

                except Exception as error:
                    print(
                        "  Ne mogu da obradim "
                        f"{audio_file.name}: {error}"
                    )

            if not embeddings:
                print(
                    f"Profil '{speaker_name}' nema "
                    "nijedan upotrebljiv snimak."
                )
                continue

            profiles[speaker_name] = (
                self._mean_embedding(
                    embeddings
                )
            )

        if not profiles:
            raise ValueError(
                "Nije pronađen nijedan upotrebljiv speaker profil."
            )

        self.reference_embeddings = profiles

        print(
            "Učitani poznati govornici: "
            f"{list(profiles.keys())}"
        )

        return profiles

    # =========================================================
    # EMBEDDING GOVORNIKA IZ SASTANKA
    # =========================================================

    def _build_meeting_speaker_embeddings(
        self,
        audio_path: str | Path,
        diarization_segments: list[dict],
    ) -> dict[str, torch.Tensor]:

        waveform, sample_rate = (
            self._load_audio(
                audio_path
            )
        )

        grouped_segments: dict[
            str,
            list[dict],
        ] = defaultdict(list)

        for segment in diarization_segments:
            speaker = str(
                segment["speaker"]
            )

            start = float(
                segment["start"]
            )

            end = float(
                segment["end"]
            )

            duration = end - start

            if duration < self.min_segment_duration:
                continue

            grouped_segments[
                speaker
            ].append(
                {
                    "start": start,
                    "end": end,
                    "duration": duration,
                }
            )

        speaker_embeddings: dict[
            str,
            torch.Tensor,
        ] = {}

        for speaker, segments in grouped_segments.items():
            # Biramo najduže segmente jer su obično pouzdaniji
            # za speaker embedding od veoma kratkih replika.
            selected_segments = sorted(
                segments,
                key=lambda item: item["duration"],
                reverse=True,
            )[:self.max_segments_per_speaker]

            embeddings: list[
                torch.Tensor
            ] = []

            for segment in selected_segments:
                start_sample = max(
                    0,
                    int(
                        round(
                            segment["start"]
                            * sample_rate
                        )
                    ),
                )

                end_sample = min(
                    waveform.shape[1],
                    int(
                        round(
                            segment["end"]
                            * sample_rate
                        )
                    ),
                )

                if end_sample <= start_sample:
                    continue

                segment_waveform = waveform[
                    :,
                    start_sample:end_sample,
                ]

                duration = (
                    segment_waveform.shape[1]
                    / sample_rate
                )

                if duration < self.min_segment_duration:
                    continue

                embedding = self._extract_embedding(
                    segment_waveform,
                    sample_rate,
                )

                embeddings.append(
                    embedding
                )

            if embeddings:
                speaker_embeddings[
                    speaker
                ] = self._mean_embedding(
                    embeddings
                )

        return speaker_embeddings

    # =========================================================
    # IDENTIFIKACIJA
    # =========================================================

    def identify(
        self,
        audio_path: str | Path,
        diarization_segments: list[dict],
    ) -> dict[str, str]:

        if not self.reference_embeddings:
            self.load_profiles()

        meeting_embeddings = (
            self._build_meeting_speaker_embeddings(
                audio_path,
                diarization_segments,
            )
        )

        mapping: dict[str, str] = {}

        diarized_speakers = sorted(
            {
                str(segment["speaker"])
                for segment in diarization_segments
            }
        )

        print()
        print("Speaker identification rezultati:")

        for speaker in diarized_speakers:
            embedding = (
                meeting_embeddings.get(
                    speaker
                )
            )

            if embedding is None:
                unknown_name = (
                    f"UNKNOWN_{speaker}"
                )

                mapping[speaker] = (
                    unknown_name
                )

                print(
                    f"  {speaker} -> {unknown_name} "
                    "(nema dovoljno govora za embedding)"
                )
                continue

            scores = {
                known_name: (
                    self._cosine_similarity(
                        embedding,
                        known_embedding,
                    )
                )
                for (
                    known_name,
                    known_embedding,
                ) in self.reference_embeddings.items()
            }

            best_name, best_score = max(
                scores.items(),
                key=lambda item: item[1],
            )

            score_text = ", ".join(
                f"{name}={score:.3f}"
                for name, score
                in sorted(
                    scores.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            )

            if (
                best_score
                >= self.similarity_threshold
            ):
                mapping[speaker] = (
                    best_name
                )

                print(
                    f"  {speaker} -> {best_name} "
                    f"(score={best_score:.3f})"
                )
            else:
                unknown_name = (
                    f"UNKNOWN_{speaker}"
                )

                mapping[speaker] = (
                    unknown_name
                )

                print(
                    f"  {speaker} -> {unknown_name} "
                    f"(najbolji score={best_score:.3f})"
                )

            print(
                f"    svi score-ovi: {score_text}"
            )

        return mapping


def identify_speakers(
    audio_path: str | Path,
    diarization_segments: list[dict],
    profiles_dir: str | Path,
    similarity_threshold: float = 0.55,
    min_segment_duration: float = 1.0,
    max_segments_per_speaker: int = 8,
) -> dict[str, str]:

    identifier = SpeakerIdentifier(
        profiles_dir=profiles_dir,
        similarity_threshold=similarity_threshold,
        min_segment_duration=min_segment_duration,
        max_segments_per_speaker=max_segments_per_speaker,
    )

    return identifier.identify(
        audio_path=audio_path,
        diarization_segments=diarization_segments,
    )


def apply_speaker_names(
    diarization_segments: list[dict],
    speaker_mapping: dict[str, str],
) -> list[dict]:
    """
    Vraća NOVU listu segmenata.
    Originalni diarization rezultat ne menja inplace.
    """

    identified_segments: list[dict] = []

    for segment in diarization_segments:
        updated = dict(segment)

        original_speaker = str(
            segment["speaker"]
        )

        updated["speaker"] = (
            speaker_mapping.get(
                original_speaker,
                f"UNKNOWN_{original_speaker}",
            )
        )

        identified_segments.append(
            updated
        )

    return identified_segments
