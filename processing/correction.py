CORRECTIONS = {
    "softvorskom inženjerstvo": "Softversko inženjerstvo",
    "elektrotehničkom fakultetu": "Elektrotehničkom fakultetu",
}


def correct_transcript(input_path: str, output_path: str) -> None:
    with open(input_path, "r", encoding="utf-8") as file:
        text = file.read()

    for wrong, correct in CORRECTIONS.items():
        text = text.replace(wrong, correct)

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(text)

    print("Korekcija transkripta je zavrsena.")