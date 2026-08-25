import time
from pathlib import Path

from processing.winrt_capture_process import WinRTCaptureProcess


def main():
    output = Path("audio/test_capture_raw.wav")

    capture = WinRTCaptureProcess(
        raw_output_path=str(output)
    )

    print("Pokrećem samo mikrofon...")

    info = capture.start()

    print("Mikrofon spreman:")
    print(info)

    print()
    print("Govori normalno 20 sekundi...")
    print("NEMA Diart-a.")
    print("NEMA Whisper-a.")

    time.sleep(20)

    print("Zaustavljam...")

    capture.request_stop()

    # Moramo isprazniti audio queue dok capture završava.
    while True:
        ended, _ = capture.read_audio(
            timeout=0.2
        )

        if ended:
            break

    result = capture.finish()

    print()
    print("Rezultat:")
    print(result)
    print()
    print(
        "Snimak:",
        output.resolve(),
    )


if __name__ == "__main__":
    main()