import time
from pathlib import Path

import numpy as np
from scipy.io.wavfile import read, write
from scipy.signal import resample_poly

from processing.winrt_capture_process import WinRTCaptureProcess


NAME = "Masa"
OUTPUT = Path(f"speaker_profiles/{NAME}/03.wav")
TEMP_OUTPUT = Path(f"speaker_profiles/{NAME}/temp_raw.wav")

TARGET_SAMPLE_RATE = 16000


def convert_to_16k_mono(
    input_path: Path,
    output_path: Path,
):
    sample_rate, audio = read(str(input_path))

    # int16 -> float32
    if np.issubdtype(audio.dtype, np.integer):
        audio = audio.astype(np.float32) / 32768.0
    else:
        audio = audio.astype(np.float32)

    # stereo -> mono
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    # 44100 -> 16000
    if sample_rate != TARGET_SAMPLE_RATE:
        audio = resample_poly(
            audio,
            TARGET_SAMPLE_RATE,
            sample_rate,
        )

    audio = np.clip(audio, -1.0, 1.0)

    audio_int16 = (
        audio * 32767.0
    ).astype(np.int16)

    write(
        str(output_path),
        TARGET_SAMPLE_RATE,
        audio_int16,
    )


def main():
    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    capture = WinRTCaptureProcess(
        raw_output_path=str(TEMP_OUTPUT)
    )

    print("Pokrećem snimanje...")
    capture.start()

    print("Pričaj 10 sekundi...")
    time.sleep(10)

    print("Zaustavljam snimanje...")
    capture.request_stop()

    while True:
        ended, _ = capture.read_audio(
            timeout=0.2
        )

        if ended:
            break

    capture.finish()

    print("Konvertujem u 16 kHz mono WAV...")

    convert_to_16k_mono(
        TEMP_OUTPUT,
        OUTPUT,
    )

    if TEMP_OUTPUT.exists():
        TEMP_OUTPUT.unlink()

    print()
    print("Snimanje završeno.")
    print(f"Sačuvano: {OUTPUT.resolve()}")
    print("Format: 16000 Hz, mono")


if __name__ == "__main__":
    main()