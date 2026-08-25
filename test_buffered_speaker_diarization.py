import numpy as np
import torchaudio

from processing.buffered_speaker_diarizer import (
    BufferedSpeakerDiarizer,
)

def main():
    AUDIO_PATH = (
        "audio/live_recording.wav"
    )

    # =========================================================
    # AUDIO
    # =========================================================

    waveform, sample_rate = (
        torchaudio.load(
            AUDIO_PATH
        )
    )

    if waveform.shape[0] > 1:
        waveform = waveform.mean(
            dim=0,
            keepdim=True,
        )

    if sample_rate != 16000:
        waveform = (
            torchaudio.functional.resample(
                waveform,
                sample_rate,
                16000,
            )
        )

        sample_rate = 16000

    audio = (
        waveform
        .squeeze()
        .numpy()
        .astype(
            np.float32
        )
    )

    duration = (
            len(audio)
            / sample_rate
    )

    print(
        f"Audio trajanje: "
        f"{duration:.2f}s"
    )

    # =========================================================
    # BUFFERED DIARIZER
    # =========================================================

    diarizer = (
        BufferedSpeakerDiarizer(
            buffer_seconds=30.0,
            step_seconds=20.0,
        )
    )

    buffer_seconds = 30.0
    step_seconds = 20.0

    buffer_samples = int(
        buffer_seconds
        * sample_rate
    )

    step_samples = int(
        step_seconds
        * sample_rate
    )

    all_results = []

    # =========================================================
    # BUFFER PO BUFFER
    # =========================================================

    for start_sample in range(
            0,
            len(audio),
            step_samples,
    ):

        end_sample = min(
            start_sample
            + buffer_samples,
            len(audio),
        )

        buffer_audio = audio[
                       start_sample:
                       end_sample
                       ]

        buffer_duration = (
                len(buffer_audio)
                / sample_rate
        )

        # Zadnji buffer može biti kraći.
        #
        # Ali ako ima manje od 5 sekundi,
        # nema dovoljno smisla da ga
        # posebno šaljemo pyannote-u.
        if buffer_duration < 5.0:
            continue

        buffer_start_seconds = (
                start_sample
                / sample_rate
        )

        results = (
            diarizer.process_buffer(
                buffer_audio,
                buffer_start=(
                    buffer_start_seconds
                ),
                sample_rate=(
                    sample_rate
                ),
            )
        )

        all_results.extend(
            results
        )

    # =========================================================
    # FINALNI DEBUG
    # =========================================================

    print()
    print("=" * 70)
    print("BUFFERED TEST ZAVRŠEN")
    print("=" * 70)

    print()

    print(
        "Ukupan broj globalnih govornika:",
        len(
            diarizer.global_speakers
        ),
    )

    print(
        "Globalni govornici:",
        list(
            diarizer
            .global_speakers
            .keys()
        ),
    )

    print()

    print(
        "Svi rezultati "
        "(uključuju overlap između buffer-a):"
    )

    print()

    for item in all_results:
        print(
            f"["
            f"{item['start']:.2f}s - "
            f"{item['end']:.2f}s"
            f"] "
            f"{item['speaker']} "
            f"(lokalno "
            f"{item['local_speaker']})"
        )
if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()