import asyncio
import multiprocessing as mp
import queue
import threading
import time
import wave
from pathlib import Path

import numpy as np

from winrt.windows.media import AudioBufferAccessMode

from winrt.windows.media.capture import (

    MediaCapture,
    MediaCaptureInitializationSettings,
    MediaStreamType,
    StreamingCaptureMode,
)
from winrt.windows.media.capture.frames import (
    MediaFrameReaderAcquisitionMode,
)


def _extract_pcm(audio_frame) -> np.ndarray:
    audio_buffer = audio_frame.lock_buffer(AudioBufferAccessMode.READ)
    reference = audio_buffer.create_reference()

    try:
        view = memoryview(reference)
        valid_bytes = int(audio_buffer.length)
        sample_count = valid_bytes // 4

        return np.frombuffer(
            view,
            dtype=np.float32,
            count=sample_count,
        ).copy()

    finally:
        reference.close()
        audio_buffer.close()


def _capture_process_main(
    audio_queue,
    status_queue,
    stop_event,
    raw_output_path: str,
) -> None:
    """
    Poseban proces koji radi SAMO WinRT capture.

    FrameArrived callback radi minimum:
      - preuzme PCM
      - stavi ga u lokalni queue

    Poseban writer/forwarder thread:
      - kontinuirano upisuje WAV
      - šalje blokove glavnom procesu za VAD/Diart/Whisper

    Zato Whisper/Diart iz glavnog procesa ne mogu da blokiraju
    samo snimanje audio fajla.
    """

    local_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=2048)

    counters = {
        "winrt_frames": 0,
        "callback_frames": 0,
        "discontinuities": 0,
        "local_queue_drops": 0,
        "processing_queue_drops": 0,
    }

    state = {
        "sample_rate": 0,
        "channels": 0,
        "writer_done": False,
    }

    writer_stop = threading.Event()

    raw_path = Path(raw_output_path)
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    def writer_forwarder() -> None:
        wav_file = None

        try:
            # Čekamo da WinRT utvrdi pravi format.
            while (
                state["sample_rate"] <= 0
                or state["channels"] <= 0
            ):
                if writer_stop.is_set():
                    return
                time.sleep(0.01)

            wav_file = wave.open(str(raw_path), "wb")
            wav_file.setnchannels(int(state["channels"]))
            wav_file.setsampwidth(2)
            wav_file.setframerate(int(state["sample_rate"]))

            pending = []
            pending_frames = 0
            # Ka glavnom procesu šaljemo ~250 ms blokove.
            target_frames = max(
                1,
                int(state["sample_rate"] * 0.25),
            )

            while (
                not writer_stop.is_set()
                or not local_queue.empty()
            ):
                try:
                    pcm = local_queue.get(timeout=0.05)
                except queue.Empty:
                    continue

                if pcm.size == 0:
                    continue

                # 1) Snimak se čuva NEZAVISNO od obrade.
                pcm16 = (
                    np.clip(pcm, -1.0, 1.0) * 32767.0
                ).astype(np.int16)

                wav_file.writeframesraw(
                    pcm16.reshape(-1).tobytes()
                )

                # 2) Isti audio grupišemo za glavni processing proces.
                pending.append(pcm)
                pending_frames += pcm.shape[0]

                if pending_frames >= target_frames:
                    block = np.concatenate(pending, axis=0)
                    pending = []
                    pending_frames = 0

                    try:
                        audio_queue.put(
                            ("audio", block),
                            timeout=0.5,
                        )
                    except queue.Full:
                        # Ovo NE utiče na WAV; samo live obrada može
                        # izgubiti blok ako je glavni proces potpuno zaglavljen.
                        counters["processing_queue_drops"] += 1

            if pending:
                block = np.concatenate(pending, axis=0)
                try:
                    audio_queue.put(
                        ("audio", block),
                        timeout=0.5,
                    )
                except queue.Full:
                    counters["processing_queue_drops"] += 1

        except Exception as error:
            status_queue.put(
                {
                    "type": "writer_error",
                    "error": repr(error),
                }
            )

        finally:
            if wav_file is not None:
                try:
                    wav_file.close()
                except Exception:
                    pass

            state["writer_done"] = True

            # Sentinel za processing worker.
            try:
                audio_queue.put(("end", None), timeout=2.0)
            except Exception:
                pass

    async def capture_async() -> None:
        reader = None
        media_capture = None
        frame_token = None
        writer_thread = None

        def on_frame_arrived(sender, args) -> None:
            frame_ref = sender.try_acquire_latest_frame()

            if frame_ref is None:
                return

            audio_frame = None

            try:
                audio_media_frame = frame_ref.audio_media_frame
                if audio_media_frame is None:
                    return

                audio_frame = audio_media_frame.get_audio_frame()
                if audio_frame is None:
                    return

                counters["winrt_frames"] += 1

                if audio_frame.is_discontinuous:
                    counters["discontinuities"] += 1

                pcm = _extract_pcm(audio_frame)

                if pcm.size == 0:
                    return

                channels = int(state["channels"])

                if channels > 1:
                    usable = len(pcm) - (len(pcm) % channels)
                    pcm = pcm[:usable].reshape(-1, channels)
                else:
                    pcm = pcm.reshape(-1, 1)

                counters["callback_frames"] += pcm.shape[0]

                # Callback ne radi resampling, Whisper, Diart niti disk I/O.
                try:
                    local_queue.put_nowait(pcm)
                except queue.Full:
                    counters["local_queue_drops"] += 1

            except Exception as error:
                status_queue.put(
                    {
                        "type": "callback_error",
                        "error": repr(error),
                    }
                )

            finally:
                if audio_frame is not None:
                    try:
                        audio_frame.close()
                    except Exception:
                        pass

                try:
                    frame_ref.close()
                except Exception:
                    pass

        try:
            media_capture = MediaCapture()

            settings = MediaCaptureInitializationSettings()
            settings.streaming_capture_mode = StreamingCaptureMode.AUDIO


            await media_capture.initialize_with_settings_async(
                settings
            )

            audio_source = None

            for _, source in media_capture.frame_sources.items():
                if (
                    source.info.media_stream_type
                    == MediaStreamType.AUDIO
                ):
                    audio_source = source
                    break

            if audio_source is None:
                raise RuntimeError(
                    "WinRT nije pronašao audio input source."
                )

            current_format = audio_source.current_format
            properties = current_format.audio_encoding_properties

            if properties is None:
                raise RuntimeError(
                    "WinRT audio source nema audio encoding properties."
                )

            state["sample_rate"] = int(properties.sample_rate)
            state["channels"] = int(properties.channel_count)

            bits_per_sample = int(properties.bits_per_sample)

            writer_thread = threading.Thread(
                target=writer_forwarder,
                daemon=False,
            )
            writer_thread.start()

            reader = await media_capture.create_frame_reader_async(
                audio_source
            )

            reader.acquisition_mode = (
                MediaFrameReaderAcquisitionMode.BUFFERED
            )

            frame_token = reader.add_frame_arrived(
                on_frame_arrived
            )

            reader_status = await reader.start_async()

            status_queue.put(
                {
                    "type": "ready",
                    "sample_rate": state["sample_rate"],
                    "channels": state["channels"],
                    "bits_per_sample": bits_per_sample,
                    "subtype": str(current_format.subtype),
                    "reader_status": int(reader_status),
                    "raw_path": str(raw_path.resolve()),
                }
            )

            while not stop_event.is_set():
                await asyncio.sleep(0.02)

        except Exception as error:
            status_queue.put(
                {
                    "type": "error",
                    "error": repr(error),
                }
            )

        finally:
            if reader is not None:
                try:
                    await reader.stop_async()
                except Exception:
                    pass

                if frame_token is not None:
                    try:
                        reader.remove_frame_arrived(frame_token)
                    except Exception:
                        pass

            # Tek kada WinRT više ne može da doda frame,
            # puštamo writer da isprazni lokalni queue.
            writer_stop.set()

            if writer_thread is not None:
                writer_thread.join(timeout=30.0)

            duration = 0.0
            if state["sample_rate"] > 0:
                duration = (
                    counters["callback_frames"]
                    / state["sample_rate"]
                )

            status_queue.put(
                {
                    "type": "stopped",
                    "sample_rate": state["sample_rate"],
                    "channels": state["channels"],
                    "duration": duration,
                    "winrt_frames": counters["winrt_frames"],
                    "callback_frames": counters["callback_frames"],
                    "discontinuities": counters["discontinuities"],
                    "local_queue_drops": counters["local_queue_drops"],
                    "processing_queue_drops": counters[
                        "processing_queue_drops"
                    ],
                    "raw_path": str(raw_path.resolve()),
                }
            )

    try:
        asyncio.run(capture_async())
    except Exception as error:
        status_queue.put(
            {
                "type": "error",
                "error": repr(error),
            }
        )


class WinRTCaptureProcess:
    def __init__(
        self,
        raw_output_path: str,
    ) -> None:
        self.raw_output_path = raw_output_path

        self._ctx = mp.get_context("spawn")

        # Dosta prostora za live processing blokove.
        self._audio_queue = self._ctx.Queue(maxsize=256)
        self._status_queue = self._ctx.Queue()
        self._stop_event = self._ctx.Event()

        self._process = None
        self.ready_info: dict | None = None
        self.stopped_info: dict | None = None

    def start(self, timeout: float = 20.0) -> dict:
        if self._process is not None:
            raise RuntimeError(
                "WinRT capture process je već pokrenut."
            )

        self._stop_event.clear()

        self._process = self._ctx.Process(
            target=_capture_process_main,
            args=(
                self._audio_queue,
                self._status_queue,
                self._stop_event,
                self.raw_output_path,
            ),
            daemon=False,
        )

        self._process.start()

        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            try:
                message = self._status_queue.get(timeout=0.25)
            except queue.Empty:
                if (
                    self._process is not None
                    and not self._process.is_alive()
                ):
                    raise RuntimeError(
                        "WinRT capture process se ugasio "
                        "pre inicijalizacije."
                    )
                continue

            msg_type = message.get("type")

            if msg_type == "ready":
                self.ready_info = message
                return message

            if msg_type == "error":
                raise RuntimeError(
                    "WinRT capture nije mogao da se pokrene: "
                    f"{message.get('error')}"
                )

            print(f"[CAPTURE] {message}")

        self.request_stop()

        raise RuntimeError(
            "WinRT capture process se nije inicijalizovao "
            "u očekivanom vremenu."
        )

    def read_audio(
        self,
        timeout: float = 0.1,
    ) -> tuple[bool, np.ndarray | None]:
        """
        Returns:
          (False, audio)  -> audio blok
          (True, None)    -> capture je završio i queue je ispražnjen
          (False, None)   -> timeout, trenutno nema bloka
        """
        try:
            kind, payload = self._audio_queue.get(
                timeout=timeout
            )
        except queue.Empty:
            return False, None

        if kind == "end":
            return True, None

        if kind == "audio":
            return False, payload

        return False, None

    def request_stop(self) -> None:
        self._stop_event.set()

    def finish(self, timeout: float = 30.0) -> dict:
        if self._process is None:
            return self.stopped_info or {}

        deadline = time.monotonic() + timeout
        stopped = None

        while time.monotonic() < deadline:
            try:
                message = self._status_queue.get(timeout=0.25)
            except queue.Empty:
                if not self._process.is_alive():
                    break
                continue

            msg_type = message.get("type")

            if msg_type == "stopped":
                stopped = message
                break

            if msg_type in {
                "error",
                "writer_error",
                "callback_error",
            }:
                print(f"[CAPTURE] {message}")

        self._process.join(timeout=max(0.0, deadline - time.monotonic()))

        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5.0)
            raise RuntimeError(
                "WinRT capture process nije mogao uredno da se zaustavi."
            )

        self.stopped_info = stopped or {}
        self._process = None

        return self.stopped_info