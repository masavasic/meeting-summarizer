from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def identify_speakers_subprocess(
    audio_path: str,
    diarization_segments: list[dict],
    profiles_dir: str,
    similarity_threshold: float = 0.55,
) -> dict[str, str]:
    """
    Pokreće SpeechBrain speaker identification u posebnom Python procesu.

    Glavni pipeline nikada ne importuje SpeechBrain, pa se izbegava
    konflikt sa transformers/Gemma lazy import mehanizmom.
    """

    project_root = (
        Path(__file__).resolve().parent.parent
    )

    with tempfile.TemporaryDirectory(
        prefix="speaker_identification_"
    ) as temp_dir:
        temp_dir = Path(temp_dir)

        segments_path = (
            temp_dir / "diarization_segments.json"
        )
        mapping_path = (
            temp_dir / "speaker_mapping.json"
        )

        with segments_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                diarization_segments,
                file,
                ensure_ascii=False,
                indent=2,
            )

        command = [
            sys.executable,
            "-u",
            "-m",
            "processing.speaker_identification_worker",
            "--audio",
            str(Path(audio_path).resolve()),
            "--segments",
            str(segments_path),
            "--profiles",
            str(
                (project_root / profiles_dir).resolve()
                if not Path(profiles_dir).is_absolute()
                else Path(profiles_dir)
            ),
            "--output",
            str(mapping_path),
            "--threshold",
            str(similarity_threshold),
        ]

        print(
            "Pokretanje speaker identification "
            "u posebnom procesu..."
        )

        process = subprocess.run(
            command,
            cwd=str(project_root),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            errors="replace",
        )

        if process.stdout:
            print(
                process.stdout,
                end=""
                if process.stdout.endswith("\n")
                else "\n",
            )

        if process.returncode != 0:
            raise RuntimeError(
                "Speaker identification subprocess "
                f"je završen greškom "
                f"(exit code {process.returncode})."
            )

        if not mapping_path.is_file():
            raise RuntimeError(
                "Speaker identification nije "
                "generisao mapping fajl."
            )

        with mapping_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            mapping = json.load(file)

    return {
        str(speaker): str(name)
        for speaker, name in mapping.items()
    }


def apply_speaker_names(
    diarization_segments: list[dict],
    speaker_mapping: dict[str, str],
) -> list[dict]:

    identified_segments = []

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
