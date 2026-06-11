from transformers import pipeline


def split_text(text: str, max_chars: int = 1800):
    chunks = []

    for i in range(0, len(text), max_chars):
        chunk = text[i:i + max_chars]
        chunks.append(chunk)

    return chunks


def summarize_text(transcript_path: str, summary_path: str):

    print("Pokretanje sumiranja...")

    with open(transcript_path, "r", encoding="utf-8") as file:
        transcript = file.read()

    print(f"Duzina transkripta: {len(transcript)} karaktera")

    summarizer = pipeline("text2text-generation", model="csebuetnlp/mT5_multilingual_XLSum")

    chunks = split_text(transcript, max_chars=1800)

    partial_summaries = []

    for index, chunk in enumerate(chunks, start=1):
        print(f"Sumiranje dela {index}/{len(chunks)}...")

        summary = summarizer(
            chunk,
            max_length=120,
            min_length=30,
            do_sample=False,
            truncation=True
        )

        partial_summaries.append(summary[0]["summary_text"])

    combined_summary = "\n".join(partial_summaries)

    with open(summary_path, "w", encoding="utf-8") as file:
        file.write(combined_summary)

    print("\nRezime:")
    print(combined_summary)

    print("\nSumiranje zavrseno.")