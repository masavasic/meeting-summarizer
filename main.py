import os
import sys
import time

from processing.vad import detect_speech
from processing.transcription import transcribe_audio
from processing.transcript_cleaner import clean_transcript
from processing.summarization_gemma import summarize_transcript_file


AUDIO_PATH = "audio/etf-predstavljanje.wav"

TRANSCRIPT_PATH = "outputs/transcript.txt"
CLEAN_TRANSCRIPT_PATH = "outputs/transcript_clean.txt"
SUMMARY_PATH = "outputs/summary_gemma.txt"
SUMMARY_JSON_PATH = "outputs/summary_gemma.json"
SUMMARY_DEBUG_PATH = "outputs/summary_gemma_debug.json"


def create_output_directories() -> None:
    paths = [
        TRANSCRIPT_PATH,
        CLEAN_TRANSCRIPT_PATH,
        SUMMARY_PATH,
        SUMMARY_JSON_PATH,
        SUMMARY_DEBUG_PATH,
    ]

    for path in paths:
        directory = os.path.dirname(path)

        if directory:
            os.makedirs(
                directory,
                exist_ok=True,
            )


def run_pipeline(audio_path: str) -> None:
    if not os.path.exists(audio_path):
        raise FileNotFoundError(
            f"Audio fajl nije pronađen: {audio_path}"
        )

    create_output_directories()

    start_time = time.time()

    print("=" * 70)
    print("POKRETANJE OFFLINE PIPELINE-A")
    print("=" * 70)
    print(f"Audio fajl: {audio_path}")
    print()

    print("[1/4] Detekcija govora...")
    detect_speech(audio_path)

    print()
    print("[2/4] Transkripcija...")
    transcribe_audio(
        audio_path=audio_path,
        output_path=TRANSCRIPT_PATH,
    )

    print()
    print("[3/4] Čišćenje transkripta...")
    clean_transcript(
        input_path=TRANSCRIPT_PATH,
        output_path=CLEAN_TRANSCRIPT_PATH,
    )

    print()
    print("[4/4] Sumarizacija...")
    summarize_transcript_file(
        input_path=CLEAN_TRANSCRIPT_PATH,
        output_path=SUMMARY_PATH,
        json_output_path=SUMMARY_JSON_PATH,
        debug_path=SUMMARY_DEBUG_PATH,
    )

    elapsed_time = time.time() - start_time

    print()
    print("=" * 70)
    print("PIPELINE JE USPEŠNO ZAVRŠEN")
    print("=" * 70)
    print(f"Transkript: {TRANSCRIPT_PATH}")
    print(f"Očišćen transkript: {CLEAN_TRANSCRIPT_PATH}")
    print(f"Sažetak: {SUMMARY_PATH}")
    print(f"JSON sažetak: {SUMMARY_JSON_PATH}")
    print(f"Debug podaci: {SUMMARY_DEBUG_PATH}")
    print(
        f"Ukupno vreme obrade: "
        f"{elapsed_time:.2f} sekundi"
    )


def main() -> None:
    audio_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else AUDIO_PATH
    )

    try:
        run_pipeline(audio_path)

    except KeyboardInterrupt:
        print("\nObrada je prekinuta.")
        sys.exit(130)

    except Exception as error:
        print()
        print("=" * 70)
        print("GREŠKA TOKOM OBRADE")
        print("=" * 70)
        print(f"{type(error).__name__}: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()