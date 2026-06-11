from processing.transcription import transcribe_audio
from processing.vad import detect_speech
from processing.summarization import summarize_text
from processing.correction import correct_transcript

AUDIO_PATH = "audio/etf-predstavljanje.wav"
OUTPUT_PATH = "outputs/transcript.txt"
SUMMARY_PATH = "outputs/summary.txt"
CORRECTED_TRANSCRIPT_PATH = "outputs/transcript_corrected.txt"


def main():
    #print("1. Pokretanje detekcije govora...")
    #detect_speech(AUDIO_PATH)

    print("\n2. Pokretanje transkripcije...")
    transcribe_audio(AUDIO_PATH, OUTPUT_PATH)

    print("\n3. Pokretanje korekcije transkripta...")
    correct_transcript(OUTPUT_PATH, CORRECTED_TRANSCRIPT_PATH)

    #print("\n4. Pokretanje sumiranja...")
    #summarize_text(CORRECTED_TRANSCRIPT_PATH, SUMMARY_PATH)


if __name__ == "__main__":
    main()