from faster_whisper import WhisperModel
import os
import time


def transcribe_audio(
    audio_path: str,
    output_path: str,
) -> list[dict]:
    print("Pokretanje transkripcije...")
    start_time = time.time()

    if not os.path.exists(audio_path):
        raise FileNotFoundError(
            f"Audio fajl nije pronađen: {audio_path}"
        )

    model = WhisperModel(
        "large-v3",
        device="cuda",
        compute_type="float16",
    )

    segments, info = model.transcribe(
        audio_path,
        language="sr",
        beam_size=5,
        temperature=0.0,
        vad_filter=True,
        condition_on_previous_text=True,
        initial_prompt=(
            "Elektrotehnički fakultet, ETF, Univerzitet u Beogradu, "
            "elektrotehnika, računarstvo, softversko inženjerstvo, "
            "prijemni ispit, matematika, fizika, studijski program."
        ),
    )

    print(f"Detektovan jezik: {info.language}")
    print(
        f"Verovatnoća jezika: "
        f"{info.language_probability:.2f}"
    )

    output_directory = os.path.dirname(output_path)

    if output_directory:
        os.makedirs(
            output_directory,
            exist_ok=True,
        )

    transcript_segments = []

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        for segment in segments:
            text = segment.text.strip()

            if not text:
                continue

            transcript_segment = {
                "start": float(segment.start),
                "end": float(segment.end),
                "text": text,
            }

            transcript_segments.append(
                transcript_segment
            )

            line = (
                f"[{segment.start:.2f}s - "
                f"{segment.end:.2f}s] "
                f"{text}\n"
            )

            file.write(line)
            print(line, end="")

    elapsed_time = time.time() - start_time

    print("\nTranskripcija je završena.")
    print(
        f"Vreme obrade: "
        f"{elapsed_time:.2f} sekundi."
    )

    return transcript_segments