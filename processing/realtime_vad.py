import queue
import threading
from pathlib import Path
from typing import Callable

import numpy as np
import sounddevice as sd
import torch
from scipy.io.wavfile import write
from scipy.signal import resample_poly
from silero_vad import VADIterator, load_silero_vad

from processing.realtime_transcription import RealTimeTranscriber


class RealTimeVAD:
    def __init__(
        self,
        device: int = 1,
        input_sample_rate: int = 44100,
        channels: int = 2,
        vad_sample_rate: int = 16000,
        on_transcript: Callable[[str], None] | None = None,
    ) -> None:
        self.device = device
        self.input_sample_rate = input_sample_rate
        self.channels = channels
        self.vad_sample_rate = vad_sample_rate
        self.on_transcript = on_transcript

        self.is_running = False
        self.stream: sd.InputStream | None = None

        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self.worker_thread: threading.Thread | None = None

        self.model = load_silero_vad()
        self.transcriber = RealTimeTranscriber()

        self.current_speech_chunks: list[np.ndarray] = []
        self.full_audio_chunks: list[np.ndarray] = []
        self.transcript_parts: list[str] = []

        self.is_speaking = False

        self.vad_iterator = VADIterator(
            self.model,
            sampling_rate=self.vad_sample_rate,
            threshold=0.5,
            min_silence_duration_ms=700,
            speech_pad_ms=200,
        )

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info,
        status,
    ) -> None:
        if status:
            print(f"Upozorenje mikrofona: {status}")

        if self.is_running:
            audio_copy = indata.copy()

            # Čuva ceo razgovor za završni WAV.
            self.full_audio_chunks.append(audio_copy)

            # Šalje blok real-time VAD obradi.
            self.audio_queue.put(audio_copy)

    def _prepare_audio(
        self,
        audio: np.ndarray,
    ) -> np.ndarray:
        # Stereo u mono.
        if audio.ndim == 2 and audio.shape[1] > 1:
            audio = np.mean(audio, axis=1)
        else:
            audio = audio.reshape(-1)

        # Ulazni sample rate u 16 kHz.
        if self.input_sample_rate != self.vad_sample_rate:
            audio = resample_poly(
                audio,
                up=self.vad_sample_rate,
                down=self.input_sample_rate,
            )

        return audio.astype(np.float32)

    def _transcribe_current_speech(self) -> None:
        if not self.current_speech_chunks:
            return

        speech_audio = np.concatenate(
            self.current_speech_chunks
        )
        self.current_speech_chunks = []

        duration = (
            len(speech_audio)
            / self.vad_sample_rate
        )

        if duration < 0.4:
            print(
                "Govorni segment je prekratak "
                "i biće preskočen."
            )
            return

        print("Transkripcija govornog segmenta...")

        try:
            text = self.transcriber.transcribe_segment(
                speech_audio
            )

            if not text:
                print("Whisper nije prepoznao tekst.")
                return

            self.transcript_parts.append(text)
            print(f"Prepoznato: {text}")

            if self.on_transcript is not None:
                self.on_transcript(text)

        except Exception as error:
            print(
                "Greška tokom real-time transkripcije: "
                f"{error}"
            )

    def _process_audio(self) -> None:
        pending_audio = np.array(
            [],
            dtype=np.float32,
        )
        vad_window_size = 512

        while self.is_running or not self.audio_queue.empty():
            try:
                chunk = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            prepared_audio = self._prepare_audio(chunk)

            pending_audio = np.concatenate(
                [pending_audio, prepared_audio]
            )

            while len(pending_audio) >= vad_window_size:
                vad_chunk = pending_audio[:vad_window_size]
                pending_audio = pending_audio[vad_window_size:]

                tensor = torch.from_numpy(vad_chunk)

                speech_event = self.vad_iterator(
                    tensor,
                    return_seconds=True,
                )

                if speech_event and "start" in speech_event:
                    self.is_speaking = True
                    self.current_speech_chunks = []

                    print(
                        "Govor je počeo: "
                        f"{speech_event['start']:.2f}s"
                    )

                if self.is_speaking:
                    self.current_speech_chunks.append(
                        vad_chunk.copy()
                    )

                if speech_event and "end" in speech_event:
                    print(
                        "Govor je završen: "
                        f"{speech_event['end']:.2f}s"
                    )

                    self.is_speaking = False
                    self._transcribe_current_speech()

    def start(self) -> None:
        if self.is_running:
            raise RuntimeError(
                "Real-time VAD je već pokrenut."
            )

        self.vad_iterator.reset_states()

        self.current_speech_chunks = []
        self.full_audio_chunks = []
        self.transcript_parts = []
        self.is_speaking = False

        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

        self.is_running = True

        self.worker_thread = threading.Thread(
            target=self._process_audio,
            daemon=True,
        )
        self.worker_thread.start()

        self.stream = sd.InputStream(
            device=self.device,
            samplerate=self.input_sample_rate,
            channels=self.channels,
            dtype="float32",
            callback=self._audio_callback,
        )
        self.stream.start()

        print("Real-time VAD je pokrenut.")

    def stop(
        self,
        audio_output_path: str = "audio/live_recording.wav",
        transcript_output_path: str = (
            "outputs/transcript_live.txt"
        ),
    ) -> tuple[str, str]:
        if not self.is_running:
            raise RuntimeError(
                "Real-time snimanje nije pokrenuto."
            )

        self.is_running = False

        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        if self.worker_thread is not None:
            self.worker_thread.join(timeout=10)
            self.worker_thread = None

        # Obradi i poslednju rečenicu čak i ako nije bilo
        # dovoljno tišine pre klika na Zaustavi.
        if self.is_speaking and self.current_speech_chunks:
            print(
                "Obrada poslednjeg govornog segmenta..."
            )
            self.is_speaking = False
            self._transcribe_current_speech()

        audio_path = self.save_recording(
            audio_output_path
        )
        transcript_path = self.save_transcript(
            transcript_output_path
        )

        self.vad_iterator.reset_states()

        print("Real-time VAD je zaustavljen.")

        return audio_path, transcript_path

    def save_recording(
        self,
        output_path: str,
    ) -> str:
        if not self.full_audio_chunks:
            raise RuntimeError(
                "Nije snimljen nijedan audio-podatak."
            )

        audio = np.concatenate(
            self.full_audio_chunks,
            axis=0,
        )

        audio = self._prepare_audio(audio)
        audio = np.clip(audio, -1.0, 1.0)

        audio_int16 = (
            audio * 32767
        ).astype(np.int16)

        output_file = Path(output_path)
        output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        write(
            str(output_file),
            self.vad_sample_rate,
            audio_int16,
        )

        print(f"Snimak je sačuvan: {output_file}")

        return str(output_file.resolve())

    def save_transcript(
        self,
        output_path: str,
    ) -> str:
        output_file = Path(output_path)
        output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        transcript = "\n".join(
            self.transcript_parts
        ).strip()

        output_file.write_text(
            transcript,
            encoding="utf-8",
        )

        print(
            f"Live transkript je sačuvan: {output_file}"
        )

        return str(output_file.resolve())

    def get_transcript(self) -> str:
        return "\n".join(
            self.transcript_parts
        ).strip()