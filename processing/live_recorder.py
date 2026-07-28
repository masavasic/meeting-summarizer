import threading
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from scipy.signal import resample_poly


class LiveRecorder:
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        device: int | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.device = device

        self.is_recording = False
        self.audio_chunks: list[np.ndarray] = []
        self.stream: sd.InputStream | None = None

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time,
        status,
    ) -> None:
        if status:
            print(f"Upozorenje mikrofona: {status}")

        if self.is_recording:
            self.audio_chunks.append(indata.copy())

    def start_recording(self) -> None:
        if self.is_recording:
            raise RuntimeError("Snimanje je već pokrenuto.")

        self.audio_chunks = []
        self.is_recording = True

        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            callback=self._audio_callback,
            device=self.device,
        )

        self.stream.start()
        print("Snimanje je pokrenuto.")

    def stop_recording(
        self,
        output_path: str = "audio/live_recording.wav",
    ) -> str:
        if not self.is_recording:
            raise RuntimeError("Snimanje nije pokrenuto.")

        self.is_recording = False

        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        if not self.audio_chunks:
            raise RuntimeError("Nije snimljen nijedan audio-podatak.")

        audio_data = np.concatenate(
            self.audio_chunks,
            axis=0,
        )
        # Ako je snimak stereo, pretvori ga u mono
        if audio_data.ndim == 2 and audio_data.shape[1] > 1:
            audio_data = np.mean(audio_data, axis=1)

        # Pretvori sample rate sa 44100 Hz na 16000 Hz
        if self.sample_rate != 16000:
            audio_data = resample_poly(
                audio_data,
                up=16000,
                down=self.sample_rate,
            )

        output_sample_rate = 16000

        # Pretvaranje float32 vrednosti u WAV int16 format
        audio_data = np.clip(audio_data, -1.0, 1.0)
        audio_int16 = (audio_data * 32767).astype(np.int16)

        output_file = Path(output_path)
        output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        write(
            str(output_file),
            output_sample_rate,
            audio_int16,
        )

        print(f"Snimak je sačuvan: {output_file}")

        return str(output_file.resolve())