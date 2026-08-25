from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from processing.speaker_identification import identify_speakers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--segments", required=True)
    parser.add_argument("--profiles", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=0.55)

    args = parser.parse_args()

    segments_path = Path(args.segments)
    output_path = Path(args.output)

    with segments_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        diarization_segments = json.load(file)

    mapping = identify_speakers(
        audio_path=args.audio,
        diarization_segments=diarization_segments,
        profiles_dir=args.profiles,
        similarity_threshold=args.threshold,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            mapping,
            file,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
