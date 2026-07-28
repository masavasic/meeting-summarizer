import os
import sys
import time
import json

from processing.vad import detect_speech
from processing.transcription import transcribe_audio
from processing.transcript_cleaner import clean_transcript
from processing.summarization_gemma import summarize_transcript_file
from processing.pdf_report import generate_pdf_report


AUDIO_PATH = "audio/etf-predstavljanje.wav"

TRANSCRIPT_PATH = "outputs/transcript.txt"
LIVE_TRANSCRIPT_PATH = "outputs/transcript_live.txt"
CLEAN_TRANSCRIPT_PATH = "outputs/transcript_clean.txt"
SUMMARY_PATH = "outputs/summary_gemma.txt"
SUMMARY_JSON_PATH = "outputs/summary_gemma.json"
SUMMARY_DEBUG_PATH = "outputs/summary_gemma_debug.json"
PDF_REPORT_PATH = "outputs/meeting_report.pdf"


def create_output_directories() -> None:
    paths = [
        TRANSCRIPT_PATH,
        CLEAN_TRANSCRIPT_PATH,
        SUMMARY_PATH,
        SUMMARY_JSON_PATH,
        SUMMARY_DEBUG_PATH,
        PDF_REPORT_PATH,
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

    print("[1/5] Detekcija govora...")
    detect_speech(audio_path)

    print()
    print("[2/5] Transkripcija...")
    transcribe_audio(
        audio_path=audio_path,
        output_path=TRANSCRIPT_PATH,
    )

    print()
    print("[3/5] Čišćenje transkripta...")
    clean_transcript(
        input_path=TRANSCRIPT_PATH,
        output_path=CLEAN_TRANSCRIPT_PATH,
    )

    print()
    print("[4/5] Sumarizacija...")
    summarize_transcript_file(
        input_path=CLEAN_TRANSCRIPT_PATH,
        output_path=SUMMARY_PATH,
        json_output_path=SUMMARY_JSON_PATH,
        debug_path=SUMMARY_DEBUG_PATH,
    )

    print()
    print("[5/5] Generisanje PDF sažetka...")
    generate_pdf_report(
        summary_json_path=SUMMARY_JSON_PATH,
        output_pdf_path=PDF_REPORT_PATH,
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
    print(f"PDF sažetak: {PDF_REPORT_PATH}")
    print(
        f"Ukupno vreme obrade: "
        f"{elapsed_time:.2f} sekundi"
    )
def create_short_live_summary(
    transcript_path: str,
) -> None:
    with open(
        transcript_path,
        "r",
        encoding="utf-8",
    ) as file:
        transcript = file.read().strip()

    if not transcript:
        raise ValueError(
            "Live transkript je prazan."
        )

    lines = [
        line.strip()
        for line in transcript.splitlines()
        if line.strip()
    ]

    first_line = (
        lines[0]
        if lines
        else transcript
    )

    title_words = first_line.split()[:8]
    title = " ".join(title_words).strip(" .,!?:;")

    if not title:
        title = "Kratak razgovor"

    important_information = [
        line.rstrip(" .") + "."
        for line in lines
    ]

    summary_data = {
        "title": title,
        "content_type": "razgovor",
        "short_overview": transcript,
        "main_topics": [
            title.rstrip(".") + "."
        ],
        "important_information": important_information,
        "numbers_and_dates": [],
        "conclusions": [],
        "action_items": [],
        "unclear_parts": [],
    }

    with open(
        SUMMARY_JSON_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary_data,
            file,
            ensure_ascii=False,
            indent=2,
        )

    formatted_lines = [
        "NASLOV",
        summary_data["title"],
        "",
        "VRSTA SADRŽAJA",
        summary_data["content_type"],
        "",
        "KRATAK PREGLED",
        summary_data["short_overview"],
        "",
        "GLAVNE TEME",
    ]

    formatted_lines.extend(
        f"- {item}"
        for item in summary_data["main_topics"]
    )

    formatted_lines.extend(
        [
            "",
            "VAŽNE INFORMACIJE",
        ]
    )

    formatted_lines.extend(
        f"- {item}"
        for item in summary_data[
            "important_information"
        ]
    )

    formatted_lines.extend(
        [
            "",
            "BROJEVI I DATUMI",
            "- Nema izdvojenih stavki.",
            "",
            "ZAKLJUČCI",
            "- Nema izdvojenih stavki.",
            "",
            "AKCIONE STAVKE",
            "- Nema izdvojenih stavki.",
            "",
            "NEJASNI DELOVI TRANSKRIPTA",
            "- Nema izdvojenih stavki.",
        ]
    )

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            "\n".join(formatted_lines)
        )

    with open(
        SUMMARY_DEBUG_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "mode": "short_live_transcript",
                "gemma_used": False,
                "reason": (
                    "Transkript je prekratak za "
                    "pouzdanu LLM sumarizaciju."
                ),
                "source_transcript": transcript,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
def run_transcript_pipeline(
    transcript_path: str,
) -> None:
    if not os.path.exists(transcript_path):
        raise FileNotFoundError(
            f"Transkript nije pronađen: {transcript_path}"
        )

    create_output_directories()

    start_time = time.time()

    print("=" * 70)
    print("POKRETANJE REAL-TIME PIPELINE-A")
    print("=" * 70)
    print(f"Ulazni transkript: {transcript_path}")
    print()

    print("[1/3] Čišćenje transkripta...")
    clean_transcript(
        input_path=transcript_path,
        output_path=CLEAN_TRANSCRIPT_PATH,
    )

    print()
    print("[2/3] Sumarizacija...")

    with open(
            CLEAN_TRANSCRIPT_PATH,
            "r",
            encoding="utf-8",
    ) as file:
        clean_live_transcript = file.read().strip()

    word_count = len(
        clean_live_transcript.split()
    )

    line_count = len(
        [
            line
            for line in clean_live_transcript.splitlines()
            if line.strip()
        ]
    )

    if word_count < 80 or line_count < 4:
        print(
            "Live transkript je kratak. "
            "Kreira se bezbedan sažetak bez Gemme."
        )

        create_short_live_summary(
            CLEAN_TRANSCRIPT_PATH
        )

    else:
        summarize_transcript_file(
            input_path=CLEAN_TRANSCRIPT_PATH,
            output_path=SUMMARY_PATH,
            json_output_path=SUMMARY_JSON_PATH,
            debug_path=SUMMARY_DEBUG_PATH,
        )

    print()
    print("[3/3] Generisanje PDF sažetka...")
    generate_pdf_report(
        summary_json_path=SUMMARY_JSON_PATH,
        output_pdf_path=PDF_REPORT_PATH,
    )

    elapsed_time = time.time() - start_time

    print()
    print("=" * 70)
    print("REAL-TIME PIPELINE JE USPEŠNO ZAVRŠEN")
    print("=" * 70)
    print(f"Ulazni transkript: {transcript_path}")
    print(f"Očišćen transkript: {CLEAN_TRANSCRIPT_PATH}")
    print(f"Sažetak: {SUMMARY_PATH}")
    print(f"JSON sažetak: {SUMMARY_JSON_PATH}")
    print(f"Debug podaci: {SUMMARY_DEBUG_PATH}")
    print(f"PDF sažetak: {PDF_REPORT_PATH}")
    print(
        f"Ukupno vreme obrade: "
        f"{elapsed_time:.2f} sekundi"
    )
def main() -> None:
    try:
        if (
            len(sys.argv) >= 3
            and sys.argv[1] == "--transcript"
        ):
            transcript_path = sys.argv[2]

            run_transcript_pipeline(
                transcript_path
            )

        else:
            audio_path = (
                sys.argv[1]
                if len(sys.argv) > 1
                else AUDIO_PATH
            )

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