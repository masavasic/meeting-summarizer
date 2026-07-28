import time

from processing.realtime_vad import RealTimeVAD


vad = RealTimeVAD(
    device=1,
    input_sample_rate=44100,
    channels=2,
)

input("Pritisni Enter da pokreneš real-time VAD...")

vad.start()

print()
print("Pričaj, pa napravi pauzu od najmanje jedne sekunde.")
print("Test traje 20 sekundi.")
print()

try:
    time.sleep(20)
finally:
    vad.stop()