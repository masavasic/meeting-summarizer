from processing.summarization import summarize_transcript_file


TRANSCRIPT_PATH = "outputs/transcript_clean.txt"
SUMMARY_PATH = "outputs/meeting_summary.txt"


def main():
    summarize_transcript_file(
        transcript_path=TRANSCRIPT_PATH,
        summary_path=SUMMARY_PATH
    )


if __name__ == "__main__":
    main()