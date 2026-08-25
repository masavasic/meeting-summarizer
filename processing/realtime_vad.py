import queue
import time
from pathlib import Path
from typing import Callable
import threading

import numpy as np
import torch

from scipy.io.wavfile import read, write
from scipy.signal import resample_poly
from silero_vad import VADIterator, load_silero_vad

from processing.realtime_transcription import RealTimeTranscriber
from processing.realtime_diarization import DiartSubprocessDiarizer
from processing.winrt_capture_process import WinRTCaptureProcess


class RealTimeVAD:
    def __init__(
        self,
        device: int = 1,
        input_sample_rate: int = 44100,
        channels: int = 2,
        vad_sample_rate: int = 16000,
        on_transcript: Callable[[str], None] | None = None,
    ) -> None:
        # Ostavljeno zbog kompatibilnosti sa postojećim gui.py.
        self.device = device
        self.input_sample_rate = input_sample_rate
        self.channels = channels
        self.vad_sample_rate = vad_sample_rate
        self.on_transcript = on_transcript

        self.is_running = False
        self.worker_thread: threading.Thread | None = None

        # =====================================================
        # SILERO VAD
        # =====================================================

        self.model = load_silero_vad()

        self.vad_iterator = VADIterator(
            self.model,
            sampling_rate=self.vad_sample_rate,
            threshold=0.5,
            min_silence_duration_ms=700,
            speech_pad_ms=200,
        )

        self.is_speaking = False

        # =====================================================
        # WHISPER
        # =====================================================

        print(
            "Učitavanje Whisper modela za real-time transkripciju..."
        )

        self.transcriber = RealTimeTranscriber()

        print(
            "Whisper model za real-time je učitan."
        )

        # =====================================================
        # DIART U .venv-diart SUBPROCESS-U
        # =====================================================

        print(
            "Pokretanje DIART realtime diarization worker-a..."
        )

        self.speaker_diarizer = DiartSubprocessDiarizer()

        print(
            "DIART realtime diarization worker je spreman."
        )

        # =====================================================
        # WINRT CAPTURE JE POSEBAN PROCES
        # =====================================================

        project_dir = Path(__file__).resolve().parent.parent

        self.raw_capture_path = (
            project_dir
            / "audio"
            / ".live_capture_raw.wav"
        )

        self.capture = WinRTCaptureProcess(
            raw_output_path=str(self.raw_capture_path)
        )

        # =====================================================
        # PROCESSING AUDIO
        # =====================================================

        # 16 kHz mono audio koji su stvarno dobili Diart/Whisper.
        self.processed_audio = np.zeros(
            0,
            dtype=np.float32,
        )

        self.current_speaker_group: dict | None = None
        self.detected_speakers: set[str] = set()

        self.merge_gap_seconds = 0.65
        self.min_transcription_seconds = 1.0
        self.max_transcription_seconds = 8.0

        self.transcript_parts: list[str] = []
        self.transcript_entries: list[dict] = []

        self.recording_start_time: float | None = None
        self.processing_audio_frames = 0
        self.capture_info: dict = {}

    # =========================================================
    # AUDIO PREPARATION
    # =========================================================

    def _prepare_audio(
        self,
        audio: np.ndarray,
    ) -> np.ndarray:

        if (
            audio.ndim == 2
            and audio.shape[1] > 1
        ):
            audio = np.mean(
                audio.astype(np.float32),
                axis=1,
            )
        else:
            audio = audio.astype(
                np.float32
            ).reshape(-1)

        if self.input_sample_rate != self.vad_sample_rate:
            audio = resample_poly(
                audio,
                up=self.vad_sample_rate,
                down=self.input_sample_rate,
            )

        return audio.astype(np.float32)

    # =========================================================
    # SILERO VAD - SAMO SPEECH DIJAGNOSTIKA
    # =========================================================

    def _process_vad_audio(
        self,
        audio: np.ndarray,
        pending_vad_audio: np.ndarray,
    ) -> np.ndarray:

        vad_window_size = 512

        pending_vad_audio = np.concatenate(
            [pending_vad_audio, audio]
        )

        while len(pending_vad_audio) >= vad_window_size:
            vad_chunk = pending_vad_audio[:vad_window_size]
            pending_vad_audio = pending_vad_audio[
                vad_window_size:
            ]

            tensor = torch.from_numpy(vad_chunk)

            speech_event = self.vad_iterator(
                tensor,
                return_seconds=True,
            )

            if speech_event and "start" in speech_event:
                self.is_speaking = True
                print(
                    "Govor je počeo: "
                    f"{speech_event['start']:.2f}s"
                )

            if speech_event and "end" in speech_event:
                self.is_speaking = False
                print(
                    "Govor je završen: "
                    f"{speech_event['end']:.2f}s"
                )

        return pending_vad_audio

    # =========================================================
    # SPEAKER LABEL
    # =========================================================

    def _normalize_speaker_label(
        self,
        label: str,
    ) -> str:

        text = str(label)
        lowered = text.lower()

        if lowered.startswith("speaker"):
            suffix = lowered[len("speaker"):]

            if suffix.isdigit():
                return (
                    f"SPEAKER_{int(suffix):02d}"
                )

        return text

    # =========================================================
    # DIART -> WHISPER GRUPE
    # =========================================================

    def _consume_diart_segments(
        self,
        segments: list[dict],
    ) -> None:

        if not segments:
            return

        ordered = sorted(
            segments,
            key=lambda item: (
                item["start"],
                item["end"],
            ),
        )

        for item in ordered:
            start = float(item["start"])
            end = float(item["end"])

            if end <= start:
                continue

            if end - start < 0.08:
                continue

            speaker = (
                self._normalize_speaker_label(
                    item["speaker"]
                )
            )

            self.detected_speakers.add(
                speaker
            )

            segment = {
                "start": start,
                "end": end,
                "speaker": speaker,
            }

            if self.current_speaker_group is None:
                self.current_speaker_group = segment
                continue

            current = self.current_speaker_group
            gap = start - current["end"]

            if (
                speaker == current["speaker"]
                and gap <= self.merge_gap_seconds
            ):
                current["end"] = max(
                    current["end"],
                    end,
                )

                if (
                    current["end"]
                    - current["start"]
                    >= self.max_transcription_seconds
                ):
                    self._flush_current_speaker_group()

                continue

            self._flush_current_speaker_group()
            self.current_speaker_group = segment

    def _flush_current_speaker_group(
        self,
        force: bool = False,
    ) -> None:

        group = self.current_speaker_group

        if group is None:
            return

        self.current_speaker_group = None

        duration = (
            group["end"] - group["start"]
        )

        if (
            duration
            < self.min_transcription_seconds
        ):
            print(
                "Preskačem prekratku speaker grupu: "
                f"{group['speaker']} "
                f"({duration:.2f}s)"
            )
            return

        self._transcribe_speaker_group(
            group
        )

    # =========================================================
    # WHISPER
    # =========================================================

    def _transcribe_speaker_group(
        self,
        group: dict,
    ) -> None:

        start_sample = int(
            round(
                group["start"]
                * self.vad_sample_rate
            )
        )

        end_sample = int(
            round(
                group["end"]
                * self.vad_sample_rate
            )
        )

        start_sample = max(
            0,
            start_sample,
        )

        end_sample = min(
            len(self.processed_audio),
            end_sample,
        )

        if end_sample <= start_sample:
            return

        group_audio = self.processed_audio[
            start_sample:end_sample
        ]

        duration = (
            len(group_audio)
            / self.vad_sample_rate
        )

        if (
            duration
            < self.min_transcription_seconds
        ):
            return

        speaker = group["speaker"]

        print(
            f"Transkripcija: {speaker} "
            f"[{group['start']:.2f}s - "
            f"{group['end']:.2f}s] "
            f"({duration:.1f}s)"
        )

        text = (
            self.transcriber
            .transcribe_segment(
                group_audio
            )
        )

        if not text:
            print(
                f"{speaker}: Whisper nije "
                "prepoznao tekst."
            )
            return

        transcript_line = (
            f"{speaker}: {text}"
        )

        self.transcript_parts.append(
            transcript_line
        )

        self.transcript_entries.append(
            {
                "start": group["start"],
                "end": group["end"],
                "speaker": speaker,
                "text": text,
            }
        )

        print(
            f"Prepoznato: {transcript_line}"
        )

        if self.on_transcript is not None:
            self.on_transcript(
                transcript_line
            )

    # =========================================================
    # AUDIO PROCESSING WORKER
    # =========================================================

    def _process_audio(self) -> None:
        pending_input_chunks: list[
            np.ndarray
        ] = []

        pending_input_frames = 0

        # Resampling radimo na ~1s kontinualnog inputa.
        input_block_frames = (
            self.input_sample_rate
        )

        pending_vad_audio = np.zeros(
            0,
            dtype=np.float32,
        )

        capture_ended = False

        while not capture_ended:
            ended, chunk = (
                self.capture.read_audio(
                    timeout=0.1
                )
            )

            if ended:
                capture_ended = True
                break

            if chunk is None:
                continue

            self.processing_audio_frames += (
                chunk.shape[0]
            )

            pending_input_chunks.append(
                chunk
            )

            pending_input_frames += (
                chunk.shape[0]
            )

            if (
                pending_input_frames
                < input_block_frames
            ):
                continue

            continuous_input = np.concatenate(
                pending_input_chunks,
                axis=0,
            )

            pending_input_chunks = []
            pending_input_frames = 0

            prepared_audio = (
                self._prepare_audio(
                    continuous_input
                )
            )

            self.processed_audio = np.concatenate(
                [
                    self.processed_audio,
                    prepared_audio,
                ]
            )

            pending_vad_audio = (
                self._process_vad_audio(
                    prepared_audio,
                    pending_vad_audio,
                )
            )

            try:
                segments = (
                    self.speaker_diarizer
                    .process_audio(
                        prepared_audio
                    )
                )

                self._consume_diart_segments(
                    segments
                )

            except Exception as error:
                print(
                    "Greška tokom DIART "
                    f"diarizacije: {error}"
                )

        # Poslednji <1 s blok.
        if pending_input_chunks:
            continuous_input = np.concatenate(
                pending_input_chunks,
                axis=0,
            )

            prepared_audio = (
                self._prepare_audio(
                    continuous_input
                )
            )

            self.processed_audio = np.concatenate(
                [
                    self.processed_audio,
                    prepared_audio,
                ]
            )

            pending_vad_audio = (
                self._process_vad_audio(
                    prepared_audio,
                    pending_vad_audio,
                )
            )

            try:
                segments = (
                    self.speaker_diarizer
                    .process_audio(
                        prepared_audio
                    )
                )

                self._consume_diart_segments(
                    segments
                )

            except Exception as error:
                print(
                    "Greška tokom poslednje "
                    f"DIART obrade: {error}"
                )

        self._flush_current_speaker_group(
            force=True
        )

    # =========================================================
    # START
    # =========================================================

    def start(self) -> None:
        if self.is_running:
            raise RuntimeError(
                "Real-time VAD je već pokrenut."
            )

        self.vad_iterator.reset_states()

        self.processed_audio = np.zeros(
            0,
            dtype=np.float32,
        )

        self.current_speaker_group = None
        self.detected_speakers = set()

        self.transcript_parts = []
        self.transcript_entries = []

        self.processing_audio_frames = 0
        self.is_speaking = False

        print(
            "Pokrećem izolovani WinRT "
            "audio capture proces..."
        )

        ready = self.capture.start()

        self.input_sample_rate = int(
            ready["sample_rate"]
        )

        self.channels = int(
            ready["channels"]
        )

        print(
            "WinRT capture proces je spreman."
        )

        print(
            "WinRT audio format: "
            f"{ready.get('subtype')}, "
            f"{self.input_sample_rate} Hz, "
            f"{self.channels} kanala, "
            f"{ready.get('bits_per_sample')} bit"
        )

        print(
            "WinRT reader status: "
            f"{ready.get('reader_status')}"
        )

        # Wall-clock merimo tek kada je mikrofon stvarno spreman.
        self.recording_start_time = (
            time.monotonic()
        )

        self.is_running = True

        self.worker_thread = threading.Thread(
            target=self._process_audio,
            daemon=True,
        )

        self.worker_thread.start()

        print(
            "Real-time VAD je pokrenut."
        )

        print(
            "DIART live diarization: "
            "5s kontekst / 0.5s korak."
        )

    # =========================================================
    # STOP
    # =========================================================

    def stop(
        self,
        audio_output_path: str = (
            "audio/live_recording.wav"
        ),
        transcript_output_path: str = (
            "outputs/transcript_live.txt"
        ),
    ) -> tuple[str, str]:

        if not self.is_running:
            raise RuntimeError(
                "Real-time snimanje nije pokrenuto."
            )

        wall_duration = None

        if (
            self.recording_start_time
            is not None
        ):
            wall_duration = (
                time.monotonic()
                - self.recording_start_time
            )

            print(
                "Realno vreme snimanja "
                "(od kada je mikrofon spreman): "
                f"{wall_duration:.1f}s"
            )

        print(
            "Zaustavljam izolovani "
            "WinRT capture proces..."
        )

        # 1) Kažemo capture-u da stane.
        self.capture.request_stop()

        # 2) Processing worker isprazni SVE što je capture
        #    već poslao i dobije sentinel.
        if self.worker_thread is not None:
            print(
                "Čekam završetak "
                "DIART/Whisper obrade..."
            )

            self.worker_thread.join(
                timeout=60.0
            )

            if self.worker_thread.is_alive():
                raise RuntimeError(
                    "Processing worker se nije "
                    "završio na vreme."
                )

            self.worker_thread = None

        # 3) Tek sada joinujemo capture proces i uzimamo statistiku.
        self.capture_info = (
            self.capture.finish()
        )

        self.is_running = False

        try:
            self.speaker_diarizer.close()
        except Exception as error:
            print(
                "Greška pri gašenju "
                f"DIART worker-a: {error}"
            )

        capture_duration = float(
            self.capture_info.get(
                "duration",
                0.0,
            )
        )

        processing_duration = (
            self.processing_audio_frames
            / self.input_sample_rate
            if self.input_sample_rate > 0
            else 0.0
        )

        print(
            "Ukupno audio podataka u "
            "capture procesu: "
            f"{capture_duration:.1f}s"
        )

        print(
            "Audio prosleđen live obradi: "
            f"{processing_duration:.1f}s"
        )

        print(
            "Broj WinRT audio frame-ova: "
            f"{self.capture_info.get('winrt_frames', 0)}"
        )

        print(
            "Discontinuity frame-ova: "
            f"{self.capture_info.get('discontinuities', 0)}"
        )

        print(
            "Izgubljeno u lokalnom capture queue-u: "
            f"{self.capture_info.get('local_queue_drops', 0)}"
        )

        print(
            "Izgubljeno ka live processing queue-u: "
            f"{self.capture_info.get('processing_queue_drops', 0)}"
        )

        print(
            "Broj DIART govornika: "
            f"{len(self.detected_speakers)}"
        )

        print(
            "DIART govornici: "
            f"{sorted(self.detected_speakers)}"
        )

        # Finalni WAV pravimo IZ FAJLA koji je capture proces
        # samostalno čuvao, a ne iz processing queue-a.
        audio_path = self._make_final_recording(
            audio_output_path
        )

        transcript_path = (
            self.save_transcript(
                transcript_output_path
            )
        )

        self.vad_iterator.reset_states()
        self.recording_start_time = None

        print(
            "Real-time VAD je zaustavljen."
        )

        return (
            audio_path,
            transcript_path,
        )

    # =========================================================
    # FINALNI WAV IZ NEZAVISNOG CAPTURE FAJLA
    # =========================================================

    def _make_final_recording(
        self,
        output_path: str,
    ) -> str:

        raw_path_text = self.capture_info.get(
            "raw_path",
            str(self.raw_capture_path),
        )

        raw_path = Path(
            raw_path_text
        )

        if not raw_path.is_file():
            raise RuntimeError(
                "Capture WAV nije pronađen: "
                f"{raw_path}"
            )

        sr, audio = read(
            str(raw_path)
        )

        if audio.size == 0:
            raise RuntimeError(
                "Capture WAV je prazan."
            )

        if np.issubdtype(
            audio.dtype,
            np.integer,
        ):
            max_value = float(
                np.iinfo(audio.dtype).max
            )

            audio = (
                audio.astype(np.float32)
                / max_value
            )
        else:
            audio = audio.astype(
                np.float32
            )

        if (
            audio.ndim == 2
            and audio.shape[1] > 1
        ):
            audio = np.mean(
                audio,
                axis=1,
            )
        else:
            audio = audio.reshape(-1)

        if sr != self.vad_sample_rate:
            audio = resample_poly(
                audio,
                up=self.vad_sample_rate,
                down=sr,
            ).astype(np.float32)

        audio = np.clip(
            audio,
            -1.0,
            1.0,
        )

        audio_int16 = (
            audio * 32767.0
        ).astype(np.int16)

        output_file = Path(
            output_path
        )

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        write(
            str(output_file),
            self.vad_sample_rate,
            audio_int16,
        )

        print(
            "Snimak je sačuvan iz "
            "nezavisnog capture procesa: "
            f"{output_file}"
        )

        # Temp raw fajl više nije potreban.
        try:
            raw_path.unlink()
        except Exception:
            pass

        return str(
            output_file.resolve()
        )

    # =========================================================
    # SAVE TRANSCRIPT
    # =========================================================

    def save_transcript(
        self,
        output_path: str,
    ) -> str:

        output_file = Path(
            output_path
        )

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
            "Live transkript je sačuvan: "
            f"{output_file}"
        )

        return str(
            output_file.resolve()
        )

    def get_transcript(
        self,
    ) -> str:

        return "\n".join(
            self.transcript_parts
        ).strip()
