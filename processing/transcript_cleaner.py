import re #regex
from difflib import SequenceMatcher #Python modul za poredjenje tekstova


def remove_timestamp(line: str) -> str:
    """
    Uklanja vremensku oznaku oblika:
    [0.00s - 7.42s]
    """

    return re.sub(
        r"^\[\d+(?:\.\d+)?s\s*-\s*\d+(?:\.\d+)?s\]\s*",
        "",
        line,
    ).strip()

def normalize_text(text: str) -> str:
    """
    Radi samo bezbedne i opste normalizacije.

    Ne menja strucne termine, brojeve, URL-ove
    niti pokusava da pretpostavi sta je govornik rekao.
    """

    text = text.replace("\u00a0", " ") #\u00a0 je non-breaking space, zamenjujemo ga sa normalnim space

    # Visestruki razmaci
    text = re.sub(r"[ \t]+"," ", text,).strip()

    # Uklanja razmak pre interpunkcije
    text = re.sub(r"\s+([,!?;:])",r"\1",text,)

    # Uklanja razmak pre tacke, ali cuva tri tacke
    text = re.sub(r"\s+\.(?!\.)", ".", text,)

    # Dodaje razmak posle zareza, upitnika, uzvicnika,
    # tacke-zareza i dvotacke, ali samo ako nedostaje
    text = re.sub(r"([,!?;:])(?=[^\s])", r"\1 ", text,)

    # Dodaje razmak posle tacke samo kada:
    # prethodni znak nije cifra
    # sledeci znak nije cifra, tacka ili deo URL-a
    text = re.sub(r"(?<!\d)\.(?=[A-ZČĆŽŠĐ])", ". ", text,)

    # Sredjuje tri ili vise tacaka.
    text = re.sub(r"(?:\.\s*){3,}", "...", text,)

    return text.strip()

def are_similar(first: str, second: str, threshold: float = 0.94,) -> bool: #threshold-prag slicnosti
    """
    Proverava da li su dva susedna segmenta
    gotovo ista.
    """

    if not first or not second:
        return False

    normalized_first = re.sub(
        r"\W+",
        " ",
        first.casefold(), #isto radi kao lower
    ).strip()

    normalized_second = re.sub(
        r"\W+",
        " ",
        second.casefold(),
    ).strip()

    similarity = SequenceMatcher( #ratio vraca vrednost izmedju 0-1
        None,
        normalized_first,
        normalized_second,
    ).ratio()

    return similarity >= threshold

def extract_speaker(text: str) -> str | None:
    match = re.match(
        r"^([^:]+):\s*",
        text,
    )

    if match:
        return match.group(1).strip()

    return None


def remove_speaker_prefix(text: str) -> str:
    return re.sub(
        r"^[^:]+:\s*",
        "",
        text,
        count=1,
    ).strip()
def should_merge(
    previous: str,
    current: str,
) -> bool:
    if not previous or not current:
        return False

    previous_speaker = extract_speaker(previous)
    current_speaker = extract_speaker(current)

    if (
        previous_speaker is None
        or current_speaker is None
    ):
        return False

    return previous_speaker == current_speaker

def clean_transcript(input_path: str,output_path: str,) -> None:
    segments: list[str] = []

    with open(
        input_path,
        "r",
        encoding="utf-8",
    ) as input_file:
        for raw_line in input_file:
            text = remove_timestamp(raw_line)
            text = normalize_text(text)

            if not text:
                continue

            #uklanjanje duplikata
            if (segments and are_similar(text,segments[-1],)):
                continue

            segments.append(text)

    paragraphs: list[str] = []
    current_paragraph = ""

    for segment in segments:
        if not current_paragraph:
            current_paragraph = segment
            continue

        if should_merge(
                current_paragraph,
                segment,
        ):
            segment_without_speaker = (
                remove_speaker_prefix(segment)
            )

            current_paragraph = (
                f"{current_paragraph} "
                f"{segment_without_speaker}"
            )
        else:
            paragraphs.append(current_paragraph.strip())
            current_paragraph = segment

    if current_paragraph:
        paragraphs.append(
            current_paragraph.strip()
        )

    full_text = "\n".join(paragraphs)

    full_text = re.sub(
        r"[ \t]+",
        " ",
        full_text,
    )

    #uklanjamo visestruke horizontalne razmake
    full_text = re.sub(
        r"\n{3,}",
        "\n\n",
        full_text,
    ).strip()

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as output_file:
        output_file.write(full_text)

    print(
        f"Očišćen transkript je sačuvan u: {output_path}"
    )
    print(
        f"Broj ulaznih segmenata: {len(segments)}"
    )
    print(
        f"Broj pasusa: {len(paragraphs)}"
    )
    print(
        f"Broj reči: {len(full_text.split())}"
    )