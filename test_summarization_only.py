from processing.summarization_gemma import summarize_transcript_file


summarize_transcript_file(
    input_path="outputs/transcript_clean.txt",
    output_path="outputs/summary_test.txt",
    json_output_path="outputs/summary_test.json",
    debug_path="outputs/summary_test_debug.json",
)
