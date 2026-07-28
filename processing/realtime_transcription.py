import numpy as np
from faster_whisper import WhisperModel


class RealTimeTranscriber:
    def __init__(self) -> None:
        print("Učitavanje Whisper modela za real-time transkripciju...")

        self.model = WhisperModel(
            "large-v3",
            device="cuda",
            compute_type="float16",
        )

        print("Whisper model za real-time je učitan.")

    def transcribe_segment(
        self,
        audio: np.ndarray,
    ) -> str:
        if audio.size == 0:
            return ""

        audio = audio.astype(np.float32).reshape(-1)

        segments, _ = self.model.transcribe(
            audio,
            language="sr",
            beam_size=5,
            temperature=0.0,
            vad_filter=False,
            condition_on_previous_text=False,
            initial_prompt=(
                "Razgovor na srpskom jeziku, ekavicom. "
                "Govornik govori prirodno i razgovetno. "
                "Tačno prepiši izgovorene reči. "
                "Ne prevodi, ne dopunjavaj i ne menjaj smisao."
            ),
        )

        texts: list[str] = []

        for segment in segments:
            text = segment.text.strip()

            if text:
                texts.append(text)

        return " ".join(texts).strip()