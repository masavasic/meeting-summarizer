from faster_whisper import WhisperModel
import time

#WhisperModel je AI model koji pretvara govor iz audio fajla u tekst.
def transcribe_audio(audio_path: str, output_path: str) -> None:
    print("Pokretanje transkripcije...")
    start_time = time.time()
    model = WhisperModel("large-v3", device="cuda", compute_type="float16") #ucitavanje Whisper modela

    segments, info = model.transcribe( #pretvaranje audia u tekst, segments- delovi transkripta podeljeni po vremenu
        audio_path,
        language="sr"
    )

    print(f"Detektovan jezik: {info.language}")
    print(f"Verovatnoca jezika: {info.language_probability:.2f}")

    with open(output_path, "w", encoding="utf-8") as file: #otvaranje fajla za upis teksta
        for segment in segments:
            line = f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.text}\n"
            file.write(line)
            #print(line)

    end_time = time.time()
    print(f"Transkripcija je zavrsena za {end_time - start_time:.2f} sekundi.")