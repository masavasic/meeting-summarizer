import json
import os
import subprocess
import sys


def diarize_audio(audio_path: str) -> list[dict]:
    if not os.path.exists(audio_path):
        raise FileNotFoundError(
            f"Audio fajl nije pronađen: {audio_path}"
        )

    output_path = "outputs/diarization_segments.json"

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True,
    )

    print(
        "Pokretanje speaker diarization "
        "u posebnom procesu..."
    )

    result = subprocess.run(
        [
            sys.executable,
            __file__,
            "--worker",
            audio_path,
            output_path,
        ],
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Speaker diarization proces "
            "je završen greškom."
        )

    if not os.path.exists(output_path):
        raise FileNotFoundError(
            "Diarization rezultat nije pronađen."
        )

    with open(
        output_path,
        "r",
        encoding="utf-8",
    ) as file:
        diarization_segments = json.load(file)

    return diarization_segments


def _run_diarization_worker(
    audio_path: str,
    output_path: str,
) -> None:

    # PyTorch 2.6+ compatibility
    os.environ[
        "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"
    ] = "1"

    import torch
    import huggingface_hub

    # pyannote 3.x koristi use_auth_token,
    # a novi huggingface_hub koristi token.
    original_hf_hub_download = (
        huggingface_hub.hf_hub_download
    )

    def compatible_hf_hub_download(
        *args,
        **kwargs,
    ):
        if "use_auth_token" in kwargs:
            kwargs["token"] = kwargs.pop(
                "use_auth_token"
            )

        return original_hf_hub_download(
            *args,
            **kwargs,
        )

    huggingface_hub.hf_hub_download = (
        compatible_hf_hub_download
    )

    # VAŽNO:
    # pyannote se učitava samo u worker procesu.
    from pyannote.audio import Pipeline

    model_name = (
        "pyannote/speaker-diarization-3.1"
    )

    token = os.getenv("HF_TOKEN")

    if not token:
        raise ValueError(
            "HF_TOKEN nije postavljen."
        )

    print(
        "Učitavanje speaker diarization modela..."
    )

    pipeline = Pipeline.from_pretrained(
        model_name,
        use_auth_token=token,
    )

    if torch.cuda.is_available():
        pipeline.to(
            torch.device("cuda")
        )

        print(
            "Speaker diarization koristi GPU."
        )
    else:
        print(
            "Speaker diarization koristi CPU."
        )

    print(
        "Pokretanje speaker diarization..."
    )

    diarization = pipeline(audio_path)

    results = []

    for turn, _, speaker in diarization.itertracks(
        yield_label=True
    ):
        segment = {
            "start": float(turn.start),
            "end": float(turn.end),
            "speaker": speaker,
        }

        results.append(segment)

        print(
            f"[{turn.start:.2f}s - "
            f"{turn.end:.2f}s] "
            f"{speaker}"
        )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=2,
        )


def assign_speakers_to_transcript(
    whisper_segments: list[dict],
    diarization_segments: list[dict],
) -> list[dict]:

    merged_segments = []

    for whisper_segment in whisper_segments:
        whisper_start = whisper_segment[
            "start"
        ]
        whisper_end = whisper_segment[
            "end"
        ]

        best_speaker = "UNKNOWN"
        best_overlap = 0.0

        for diarization_segment in (
            diarization_segments
        ):
            diar_start = diarization_segment[
                "start"
            ]
            diar_end = diarization_segment[
                "end"
            ]

            overlap_start = max(
                whisper_start,
                diar_start,
            )

            overlap_end = min(
                whisper_end,
                diar_end,
            )

            overlap = max(
                0.0,
                overlap_end - overlap_start,
            )

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = (
                    diarization_segment[
                        "speaker"
                    ]
                )

        merged_segments.append(
            {
                "start": whisper_start,
                "end": whisper_end,
                "speaker": best_speaker,
                "text": whisper_segment["text"],
            }
        )

    return merged_segments


def save_speaker_transcript(
    merged_segments: list[dict],
    output_path: str,
) -> None:

    output_directory = os.path.dirname(
        output_path
    )

    if output_directory:
        os.makedirs(
            output_directory,
            exist_ok=True,
        )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        for segment in merged_segments:
            line = (
                f"[{segment['start']:.2f}s - "
                f"{segment['end']:.2f}s] "
                f"{segment['speaker']}: "
                f"{segment['text']}\n"
            )

            file.write(line)

    print(
        "Transkript sa govornicima "
        f"je sačuvan: {output_path}"
    )


if __name__ == "__main__":

    if (
        len(sys.argv) >= 4
        and sys.argv[1] == "--worker"
    ):
        _run_diarization_worker(
            audio_path=sys.argv[2],
            output_path=sys.argv[3],
        )