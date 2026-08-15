import torch #Koristi se za rad sa tensorima i neuronskim mrezama.
import soundfile as sf
from silero_vad import load_silero_vad, get_speech_timestamps


def detect_speech(audio_path: str): #funkcija za detektovanje govora
    # Ucitava Silero VAD model koji detektuje delove audio snimka gde postoji govor
    model = load_silero_vad()

    # Cita audio fajl i vraca audio signal i sample rate
    audio, sample_rate = sf.read(audio_path)

    # Ako je audio stereo, pretvara ga u mono format
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # Pretvara audio iz numpy formata u torch tensor,
    # jer Silero model ocekuje podatke u torch formatu
    audio_tensor = torch.from_numpy(audio).float()

    # Pokrece VAD model i vraca vremenske intervale u kojima postoji govor
    speech_timestamps = get_speech_timestamps(
        audio_tensor,
        model,
        sampling_rate=sample_rate,
        return_seconds=True
    )

    # Ispisuje pronadjene govorne segmente
    print("Detektovani delovi gde postoji govor:")

    for item in speech_timestamps:
        print(f"Pocetak: {item['start']:.2f}s, Kraj: {item['end']:.2f}s")

    # Vraca listu intervala govora, da bismo ih kasnije mogli koristiti u drugim modulima
    return speech_timestamps