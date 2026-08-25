import base64
import contextlib
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np


# =========================================================
# DIART ENGINE
#
# Ova klasa se koristi SAMO unutar .venv-diart subprocess-a.
# Importi za diart/pyannote/torch su namerno unutar __init__ da
# glavni GUI environment ne mora da ima Diart instaliran.
# =========================================================

class RealtimeDiarizer:
    def __init__(
        self,
        sample_rate: int = 16000,
        duration: float = 5.0,
        step: float = 0.5,
        delta_new: float = 0.94,
    ) -> None:
        import torch
        from diart import SpeakerDiarization, SpeakerDiarizationConfig
        from diart.models import SegmentationModel, EmbeddingModel
        from pyannote.core import SlidingWindow, SlidingWindowFeature

        self._SlidingWindow = SlidingWindow
        self._SlidingWindowFeature = SlidingWindowFeature

        self.sample_rate = sample_rate
        self.duration = duration
        self.step = step
        self.delta_new = delta_new

        self.duration_samples = int(duration * sample_rate)
        self.step_samples = int(step * sample_rate)

        self.audio = np.zeros(0, dtype=np.float32)
        self.next_end_sample = self.duration_samples

        token = os.getenv("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN nije postavljen.")

        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print("Učitavanje DIART sistema...", flush=True)
        print(f"DIART koristi: {device}", flush=True)

        segmentation = SegmentationModel.from_pretrained(
            "pyannote/segmentation-3.0",
            use_hf_token=token,
        )

        # ECAPA se u našim testovima pokazao stabilnije od starog
        # pyannote/embedding checkpoint-a.
        embedding = EmbeddingModel.from_pretrained(
            "speechbrain/spkrec-ecapa-voxceleb"
        )

        config = SpeakerDiarizationConfig(
            segmentation=segmentation,
            embedding=embedding,
            duration=duration,
            step=step,
            latency="min",
            tau_active=0.507,
            rho_update=0.006,
            delta_new=delta_new,
            max_speakers=10,
            device=device,
            sample_rate=sample_rate,
        )

        self.pipeline = SpeakerDiarization(config)

        print(
            "DIART spreman: "
            f"{duration:.1f}s buffer / "
            f"{step:.1f}s step / "
            f"latency={config.latency:.2f}s",
            flush=True,
        )
        print(f"delta_new={delta_new}", flush=True)

    def process_audio(self, audio: np.ndarray) -> list[dict]:
        if audio is None:
            return []

        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return []

        self.audio = np.concatenate((self.audio, audio))
        results: list[dict] = []

        while len(self.audio) >= self.next_end_sample:
            start_sample = self.next_end_sample - self.duration_samples
            end_sample = self.next_end_sample

            window_audio = self.audio[start_sample:end_sample]
            start_time = start_sample / self.sample_rate

            sliding_window = self._SlidingWindow(
                start=start_time,
                duration=1.0 / self.sample_rate,
                step=1.0 / self.sample_rate,
            )

            waveform = self._SlidingWindowFeature(
                window_audio.reshape(-1, 1),
                sliding_window,
            )

            output = self.pipeline([waveform])

            if output:
                annotation, _ = output[-1]

                for segment, _, speaker in annotation.itertracks(
                    yield_label=True
                ):
                    results.append(
                        {
                            "start": float(segment.start),
                            "end": float(segment.end),
                            "speaker": str(speaker),
                        }
                    )

            self.next_end_sample += self.step_samples

        return results


# =========================================================
# SUBPROCESS CLIENT
#
# Ova klasa se importuje iz glavnog .venv-a. Ona NE importuje Diart,
# već podiže processing/realtime_diarization.py kroz .venv-diart.
# =========================================================

class DiartSubprocessDiarizer:
    def __init__(self) -> None:
        project_dir = Path(__file__).resolve().parent.parent
        python_exe = (
            project_dir
            / ".venv-diart"
            / "Scripts"
            / "python.exe"
        )

        if not python_exe.is_file():
            raise RuntimeError(
                "Nije pronađen .venv-diart Python: "
                f"{python_exe}"
            )

        if not os.getenv("HF_TOKEN"):
            raise RuntimeError(
                "HF_TOKEN nije postavljen u procesu iz kog je pokrenut GUI."
            )

        script_path = Path(__file__).resolve()
        environment = os.environ.copy()
        environment["PYTHONIOENCODING"] = "utf-8"
        environment["PYTHONUTF8"] = "1"

        self._lock = threading.Lock()
        self._process = subprocess.Popen(
            [
                str(python_exe),
                "-u",
                str(script_path),
                "--diart-worker",
            ],
            cwd=str(project_dir),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=environment,
        )

        self._stderr_thread = threading.Thread(
            target=self._forward_stderr,
            daemon=True,
        )
        self._stderr_thread.start()

        response = self._read_response()
        if response.get("type") != "ready":
            self.close(force=True)
            raise RuntimeError(
                "DIART worker se nije pravilno pokrenuo: "
                + response.get("error", str(response))
            )

    def _forward_stderr(self) -> None:
        process = self._process
        if process.stderr is None:
            return

        for line in process.stderr:
            line = line.rstrip()
            if line:
                print(f"[DIART] {line}")

    def _read_response(self) -> dict:
        process = self._process
        if process.stdout is None:
            raise RuntimeError("DIART worker nema stdout pipe.")

        line = process.stdout.readline()
        if not line:
            code = process.poll()
            raise RuntimeError(
                "DIART worker je neočekivano prekinut"
                + (f" (exit code {code})." if code is not None else ".")
            )

        try:
            return json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                "DIART worker je vratio neispravan odgovor: "
                f"{line.strip()}"
            ) from error

    def _request(self, message: dict) -> dict:
        with self._lock:
            process = self._process

            if process.poll() is not None:
                raise RuntimeError(
                    f"DIART worker više ne radi (exit code {process.returncode})."
                )

            if process.stdin is None:
                raise RuntimeError("DIART worker nema stdin pipe.")

            process.stdin.write(
                json.dumps(message, ensure_ascii=False) + "\n"
            )
            process.stdin.flush()

            response = self._read_response()

            if response.get("type") == "error":
                raise RuntimeError(
                    "DIART worker greška: "
                    + response.get("error", "nepoznata greška")
                )

            return response

    def process_audio(self, audio: np.ndarray) -> list[dict]:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return []

        payload = base64.b64encode(
            np.ascontiguousarray(audio).tobytes()
        ).decode("ascii")

        response = self._request(
            {
                "command": "process",
                "audio": payload,
            }
        )

        if response.get("type") != "result":
            raise RuntimeError(
                "DIART worker je vratio neočekivan tip odgovora: "
                f"{response}"
            )

        return response.get("segments", [])

    def close(self, force: bool = False) -> None:
        process = getattr(self, "_process", None)
        if process is None:
            return

        if process.poll() is None and not force:
            try:
                self._request({"command": "stop"})
            except Exception:
                pass

        if process.poll() is None:
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=1.0)

        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass

        self._process = None


# =========================================================
# WORKER
# =========================================================

def _send_worker_message(message: dict) -> None:
    sys.stdout.write(json.dumps(message, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _run_worker() -> None:
    try:
        # Sve normalne print poruke iz Diart-a/modela šaljemo na stderr,
        # jer stdout koristimo isključivo za JSON protokol.
        with contextlib.redirect_stdout(sys.stderr):
            diarizer = RealtimeDiarizer(
                sample_rate=16000,
                duration=5.0,
                step=0.5,
                delta_new=0.94,
            )

        _send_worker_message({"type": "ready"})

    except Exception as error:
        _send_worker_message(
            {
                "type": "error",
                "error": f"{type(error).__name__}: {error}",
            }
        )
        return

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
            command = request.get("command")

            if command == "stop":
                _send_worker_message({"type": "stopped"})
                return

            if command != "process":
                raise RuntimeError(
                    f"Nepoznata DIART worker komanda: {command}"
                )

            encoded_audio = request.get("audio")
            if not encoded_audio:
                _send_worker_message(
                    {"type": "result", "segments": []}
                )
                continue

            audio = np.frombuffer(
                base64.b64decode(encoded_audio),
                dtype=np.float32,
            ).copy()

            with contextlib.redirect_stdout(sys.stderr):
                segments = diarizer.process_audio(audio)

            _send_worker_message(
                {
                    "type": "result",
                    "segments": segments,
                }
            )

        except Exception as error:
            _send_worker_message(
                {
                    "type": "error",
                    "error": f"{type(error).__name__}: {error}",
                }
            )


if __name__ == "__main__" and "--diart-worker" in sys.argv:
    _run_worker()
