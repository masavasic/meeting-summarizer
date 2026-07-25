import re
from difflib import SequenceMatcher


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
    Radi samo bezbedne i opšte normalizacije.

    Ne menja stručne termine, brojeve, URL-ove
    niti pokušava da pretpostavi šta je govornik rekao.
    """

    text = text.replace("\u00a0", " ")

    # Višestruki razmaci.
    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    ).strip()

    # Uklanja razmak pre interpunkcije.
    text = re.sub(
        r"\s+([,!?;:])",
        r"\1",
        text,
    )

    # Uklanja razmak pre tačke, ali čuva tri tačke.
    text = re.sub(
        r"\s+\.(?!\.)",
        ".",
        text,
    )

    # Dodaje razmak posle zareza, upitnika, uzvičnika,
    # tačke-zareza i dvotačke, ali samo ako nedostaje.
    text = re.sub(
        r"([,!?;:])(?=[^\s])",
        r"\1 ",
        text,
    )

    # Dodaje razmak posle tačke samo kada:
    # - prethodni znak nije cifra
    # - sledeći znak nije cifra, tačka ili deo URL-a
    text = re.sub(
        r"(?<!\d)\.(?=[A-ZČĆŽŠĐ])",
        ". ",
        text,
    )

    # Sređuje tri ili više tačaka.
    text = re.sub(
        r"(?:\.\s*){3,}",
        "...",
        text,
    )

    return text.strip()

def are_similar(
    first: str,
    second: str,
    threshold: float = 0.94,
) -> bool:
    """
    Proverava da li su dva susedna segmenta
    gotovo ista.
    """

    if not first or not second:
        return False

    normalized_first = re.sub(
        r"\W+",
        " ",
        first.casefold(),
    ).strip()

    normalized_second = re.sub(
        r"\W+",
        " ",
        second.casefold(),
    ).strip()

    similarity = SequenceMatcher(
        None,
        normalized_first,
        normalized_second,
    ).ratio()

    return similarity >= threshold


def should_merge(
    previous: str,
    current: str,
) -> bool:
    """
    Odlučuje da li dva segmenta pripadaju
    istoj rečenici.

    Ako prethodni segment nema završnu interpunkciju,
    vrlo verovatno se rečenica nastavlja.
    """

    if not previous or not current:
        return False

    return previous[-1] not in ".!?"


def clean_transcript(
    input_path: str,
    output_path: str,
) -> None:
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

            if (
                segments
                and are_similar(
                    text,
                    segments[-1],
                )
            ):
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
            current_paragraph = (
                f"{current_paragraph} {segment}"
            )
        else:
            paragraphs.append(
                current_paragraph.strip()
            )
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