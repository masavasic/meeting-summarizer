import sys
import soundfile as sf
import numpy as np
from scipy.signal import resample_poly
from processing.realtime_diarization import RealtimeDiarizer


AUDIO_PATH = "audio/Kineska veštačka inteligencija plaši SAD I RTS jutarnji program.wav"

TARGET_SR = 16000
CHUNK_SECONDS = 0.5


def main():
    print("=" * 70)
    print("DIART FILE TEST")
    print("=" * 70)

    audio, sr = sf.read(
        AUDIO_PATH,
        dtype="float32",
        always_2d=False,
    )

    print(f"Audio: {AUDIO_PATH}")
    print(f"Original sample rate: {sr}")
    print(f"Shape: {audio.shape}")

    # Stereo -> mono
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    # Stereo -> mono
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    # Resample na 16 kHz ako je potrebno
    if sr != TARGET_SR:
        print(f"Resampling: {sr} Hz -> {TARGET_SR} Hz")

        audio = resample_poly(
            audio,
            TARGET_SR,
            sr,
        ).astype(np.float32)

        sr = TARGET_SR

    diarizer = RealtimeDiarizer(
        sample_rate=TARGET_SR,
        duration=5.0,
        step=0.5,
        delta_new=0.94,
    )
    chunk_size = int(
        CHUNK_SECONDS * TARGET_SR
    )

    all_results = []

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

        if not results:
            continue

        print()
        print(
            f"--- do {(start + len(chunk)) / TARGET_SR:.1f}s ---"
        )

        for item in results:
            print(
                f"{item['start']:.2f}s - "
                f"{item['end']:.2f}s | "
                f"{item['speaker']}"
            )

        all_results.extend(results)

    speakers = sorted(
        set(
            item["speaker"]
            for item in all_results
        )
    )

    print()
    print("=" * 70)
    print("REZULTAT")
    print("=" * 70)

    print(
        f"Detektovani govornici: "
        f"{len(speakers)}"
    )

    print(
        f"Govornici: {speakers}"
    )
    print()
    print("=" * 70)
    print("SVE DIARIZATION SEKCIJE")
    print("=" * 70)

    for item in all_results:
        print(
            f"{item['start']:.2f}s - "
            f"{item['end']:.2f}s | "
            f"{item['speaker']}"
        )


if __name__ == "__main__":
    main()