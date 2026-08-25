import soundfile as sf
import numpy as np
from scipy.signal import resample_poly

from processing.realtime_diarization import RealtimeDiarizer


AUDIO_PATH = "audio/live_recording.wav"

TARGET_SR = 16000
CHUNK_SECONDS = 0.5
DELTA_VALUES = [
    0.90,
    0.92,
    0.94,
    0.96,
    0.98,
    1.00,
    1.02,
    1.04,
]


def load_audio():
    audio, sr = sf.read(
        AUDIO_PATH,
        dtype="float32",
        always_2d=False,
    )

    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    if sr != TARGET_SR:
        audio = resample_poly(
            audio,
            TARGET_SR,
            sr,
        ).astype(np.float32)

    return audio


def test_delta(audio, delta):
    diarizer = RealtimeDiarizer(
        sample_rate=TARGET_SR,
        duration=5.0,
        step=0.5,
        delta_new=delta,
    )

    chunk_size = int(
        CHUNK_SECONDS * TARGET_SR
    )

    speakers = set()
    durations = {}

    for start in range(
        0,
        len(audio),
        chunk_size,
    ):
        chunk = audio[
            start:start + chunk_size
        ]

        results = diarizer.process_audio(
            chunk
        )

        for item in results:
            speaker = item["speaker"]

            speakers.add(speaker)

            duration = (
                item["end"]
                - item["start"]
            )

            durations[speaker] = (
                durations.get(
                    speaker,
                    0.0,
                )
                + duration
            )

    return speakers, durations


def main():
    audio = load_audio()

    print()
    print("=" * 70)
    print("DIART DELTA_NEW SWEEP")
    print("=" * 70)

    for delta in DELTA_VALUES:

        print()
        print("#" * 70)
        print(
            f"TEST delta_new = {delta}"
        )
        print("#" * 70)

        speakers, durations = (
            test_delta(
                audio,
                delta,
            )
        )

        print(
            f"Broj govornika: "
            f"{len(speakers)}"
        )

        print(
            f"Govornici: "
            f"{sorted(speakers)}"
        )

        print(
            "Trajanje po govorniku:"
        )

        for speaker in sorted(
            durations
        ):
            print(
                f"  {speaker}: "
                f"{durations[speaker]:.2f}s"
            )


if __name__ == "__main__":
    main()