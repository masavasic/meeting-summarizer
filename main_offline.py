from processing.transcription import transcribe_audio
from processing.transcript_cleaner import clean_transcript


AUDIO_PATH = "audio/etf-predstavljanje.wav"
OUTPUT_PATH = "outputs/transcript.txt"
CLEAN_TRANSCRIPT_PATH = "outputs/transcript_clean.txt"


def main():
    #print("\n1. Pokretanje transkripcije...")
    #transcribe_audio(AUDIO_PATH, OUTPUT_PATH)

    print("\n2. Čišćenje transkripta...")
    clean_transcript(
        OUTPUT_PATH,
        CLEAN_TRANSCRIPT_PATH
    )


if __name__ == "__main__":
    main()