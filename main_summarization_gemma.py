from processing.summarization_gemma import (
    summarize_transcript_file,
)


if __name__ == "__main__":
    summarize_transcript_file(
        input_path="outputs/transcript_clean.txt",
        output_path="outputs/summary_gemma.txt",
        json_output_path="outputs/summary_gemma.json",
        debug_path="outputs/summary_gemma_debug.json",
    )