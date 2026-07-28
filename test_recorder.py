import time

from processing.live_recorder import LiveRecorder


recorder = LiveRecorder(
    device=1,
    sample_rate=44100,
    channels=2,
)

input("Pritisni Enter da započneš snimanje...")

recorder.start_recording()

print("Pričaj 10 sekundi...")
time.sleep(10)

audio_path = recorder.stop_recording()

print(f"Gotovo. Snimak je sačuvan u: {audio_path}")