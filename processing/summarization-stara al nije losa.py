import json
import os
import re
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


class MeetingSummarizer:
    def __init__(self) -> None:
        # Ostavljamo Google Gemma model koji već koristiš.
        self.model_name = "google/gemma-3-12b-it"

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA nije dostupna. Proveri NVIDIA drajvere "
                "i CUDA verziju PyTorch-a."
            )

        print(
            f"Učitavanje modela za sumiranje: "
            f"{self.model_name}"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=True,
        )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = (
                self.tokenizer.eos_token
            )

        # Neke grafičke kartice ne podržavaju bfloat16.
        # U tom slučaju automatski koristimo float16.
        if torch.cuda.is_bf16_supported():
            compute_dtype = torch.bfloat16
        else:
            compute_dtype = torch.float16

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            dtype=compute_dtype,
        )

        self.model.eval()

        print("Model je uspešno učitan.")

    @staticmethod
    def _transliterate_to_latin(text: str) -> str:
        """
        Pretvara srpsku ćirilicu u latinicu kako model
        ne bi mešao dva pisma u istom rezimeu.
        """

        mapping = {
            "А": "A", "а": "a",
            "Б": "B", "б": "b",
            "В": "V", "в": "v",
            "Г": "G", "г": "g",
            "Д": "D", "д": "d",
            "Ђ": "Đ", "ђ": "đ",
            "Е": "E", "е": "e",
            "Ж": "Ž", "ж": "ž",
            "З": "Z", "з": "z",
            "И": "I", "и": "i",
            "Ј": "J", "ј": "j",
            "К": "K", "к": "k",
            "Л": "L", "л": "l",
            "Љ": "Lj", "љ": "lj",
            "М": "M", "м": "m",
            "Н": "N", "н": "n",
            "Њ": "Nj", "њ": "nj",
            "О": "O", "о": "o",
            "П": "P", "п": "p",
            "Р": "R", "р": "r",
            "С": "S", "с": "s",
            "Т": "T", "т": "t",
            "Ћ": "Ć", "ћ": "ć",
            "У": "U", "у": "u",
            "Ф": "F", "ф": "f",
            "Х": "H", "х": "h",
            "Ц": "C", "ц": "c",
            "Ч": "Č", "ч": "č",
            "Џ": "Dž", "џ": "dž",
            "Ш": "Š", "ш": "š",
        }

        return "".join(
            mapping.get(character, character)
            for character in text
        )

    def _clean_final_output(
            self,
            text: str,
            transcript_text: str,
    ) -> str:
        """
        Uklanja sve što je model dodao nakon završetka rezimea,
        pretvara ćirilicu u latinicu i uklanja stavke sa
        brojevima koji ne postoje u transkriptu.
        """

        text = text.replace("\r\n", "\n").strip()

        stop_markers = [
            "[KRAJ_REZIMEA]",
            "```json",
            "```python",
            "Has the model followed",
            "Have you identified",
            "Here's a breakdown",
            "</code>",
            "Thank you for your feedback",
        ]

        for marker in stop_markers:
            marker_position = text.find(marker)

            if marker_position != -1:
                text = text[:marker_position]

        text = text.replace("```", "")
        text = self._transliterate_to_latin(text)
        text = self._normalize_text(text)

        source_numbers = self._numbers(transcript_text)
        cleaned_lines: list[str] = []

        for line in text.splitlines():
            stripped_line = line.strip()

            # Brojeve proveravamo prvenstveno u stavkama liste.
            if stripped_line.startswith("-"):
                line_numbers = self._numbers(stripped_line)

                if (
                        line_numbers
                        and not line_numbers.issubset(source_numbers)
                ):
                    print(
                        "Uklonjena je stavka sa brojem koji ne postoji "
                        f"u transkriptu: {stripped_line}"
                    )
                    continue

            cleaned_lines.append(line.rstrip())

        text = "\n".join(cleaned_lines)

        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text,
        )

        return text.strip()
    def _generate(
        self,
        prompt: str,
        max_new_tokens: int = 900,
    ) -> str:
        """
        Šalje prompt modelu i vraća samo tekst
        koji je model generisao.
        """

        messages = [
            {
                "role": "system",
                "content": (
                    "Ti si stručni urednik transkripata "
                    "na srpskom jeziku. "
                    "Pišeš prirodno, jasno i gramatički "
                    "ispravno, ekavicom i latinicom. "
                    "Koristiš isključivo informacije iz "
                    "dostavljenog teksta. "
                    "Ne izmišljaš činjenice, brojeve, imena, "
                    "odluke, zadatke ili zaključke. "
                    "Ne menjaš značenje, negacije, brojeve, "
                    "valute ili jedinice. "
                    "Izbegavaš prazne promotivne rečenice, "
                    "neodređene tvrdnje i nepotrebna "
                    "ponavljanja. "
                    "Kada se zahteva JSON, vraćaš samo validan "
                    "JSON objekat bez Markdown oznaka i "
                    "dodatnog objašnjenja."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        formatted_prompt = (
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=24000,
        )

        input_device = next(
            self.model.parameters()
        ).device

        inputs = {
            key: value.to(input_device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.12,
                no_repeat_ngram_size = 5,
                pad_token_id=(
                    self.tokenizer.pad_token_id
                ),
                eos_token_id=(
                    self.tokenizer.eos_token_id
                ),
                use_cache=True,
            )

        input_length = inputs[
            "input_ids"
        ].shape[-1]

        generated_tokens = output[0][
            input_length:
        ]

        generated_text = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return generated_text.strip()

    @staticmethod
    def _normalize_text(text: str) -> str:
        """
        Sređuje razmake i neobične znakove
        koje model ponekad vrati.
        """

        replacements = {
            "，": ",",
            "。": ".",
            "；": ";",
            "：": ":",
            "–": "-",
            "—": "-",
        }

        for wrong, correct in replacements.items():
            text = text.replace(
                wrong,
                correct,
            )

        text = re.sub(
            r"[ \t]+",
            " ",
            text,
        )

        text = re.sub(
            r" *\n *",
            "\n",
            text,
        )

        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text,
        )

        return text.strip()

    @staticmethod
    def _extract_json_object(
        text: str,
    ) -> dict[str, Any]:
        """
        Izdvaja JSON čak i kada model slučajno
        doda ```json oznake.
        """

        text = text.strip()

        fenced = re.search(
            r"```(?:json)?\s*(\{.*\})\s*```",
            text,
            flags=(
                re.DOTALL
                | re.IGNORECASE
            ),
        )

        if fenced:
            text = fenced.group(1)

        start = text.find("{")
        end = text.rfind("}")

        if (
            start == -1
            or end == -1
            or end <= start
        ):
            raise ValueError(
                "Model nije vratio JSON objekat."
            )

        candidate = text[
            start:end + 1
        ]

        try:
            parsed = json.loads(candidate)

        except json.JSONDecodeError as error:
            raise ValueError(
                "Neispravan JSON iz modela: "
                f"{error}"
            ) from error

        if not isinstance(parsed, dict):
            raise ValueError(
                "Model nije vratio JSON objekat."
            )

        return parsed

    def _generate_json(
        self,
        prompt: str,
        max_new_tokens: int,
        attempts: int = 2,
    ) -> dict[str, Any]:
        """
        Ako model prvi put vrati neispravan JSON,
        pokušava ponovo.
        """

        last_error: Exception | None = None

        for attempt in range(
            1,
            attempts + 1,
        ):
            raw_result = self._generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
            )

            try:
                return self._extract_json_object(
                    raw_result
                )

            except ValueError as error:
                last_error = error

                print(
                    "Model je vratio neispravan "
                    "JSON. "
                    f"Pokušaj {attempt}/{attempts}."
                )

        raise ValueError(
            "Model nije vratio validan JSON: "
            f"{last_error}"
        )

    def split_text(
        self,
        text: str,
        max_tokens: int = 1500,
        overlap_tokens: int = 180,
    ) -> list[str]:
        """
        Deli transkript po rečenicama i tokenima.

        Svaki sledeći deo sadrži mali završetak
        prethodnog dela. To sprečava gubitak
        konteksta na granici između dva dela.
        """

        sentences = re.split(
            r"(?<=[.!?])\s+|\n+",
            text.strip(),
        )

        sentences = [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]

        chunks: list[str] = []

        current_sentences: list[str] = []
        current_token_count = 0

        for sentence in sentences:
            sentence_tokens = (
                self.tokenizer.encode(
                    sentence,
                    add_special_tokens=False,
                )
            )

            sentence_token_count = len(
                sentence_tokens
            )

            # Ako je jedna veoma dugačka rečenica
            # veća od celog chunk-a, delimo i nju.
            if sentence_token_count > max_tokens:
                if current_sentences:
                    chunks.append(
                        " ".join(
                            current_sentences
                        )
                    )

                    current_sentences = []
                    current_token_count = 0

                step = (
                    max_tokens
                    - overlap_tokens
                )

                for start in range(
                    0,
                    sentence_token_count,
                    step,
                ):
                    piece_ids = sentence_tokens[
                        start:
                        start + max_tokens
                    ]

                    if not piece_ids:
                        continue

                    piece_text = (
                        self.tokenizer.decode(
                            piece_ids,
                            skip_special_tokens=True,
                        )
                    ).strip()

                    if piece_text:
                        chunks.append(piece_text)

                    if (
                        start + max_tokens
                        >= sentence_token_count
                    ):
                        break

                continue

            if (
                current_sentences
                and (
                    current_token_count
                    + sentence_token_count
                    > max_tokens
                )
            ):
                chunks.append(
                    " ".join(
                        current_sentences
                    )
                )

                overlap_sentences: list[str] = []
                overlap_count = 0

                for previous_sentence in reversed(
                    current_sentences
                ):
                    previous_count = len(
                        self.tokenizer.encode(
                            previous_sentence,
                            add_special_tokens=False,
                        )
                    )

                    if (
                        overlap_sentences
                        and (
                            overlap_count
                            + previous_count
                            > overlap_tokens
                        )
                    ):
                        break

                    overlap_sentences.insert(
                        0,
                        previous_sentence,
                    )

                    overlap_count += (
                        previous_count
                    )

                current_sentences = (
                    overlap_sentences
                )

                current_token_count = (
                    overlap_count
                )

            current_sentences.append(
                sentence
            )

            current_token_count += (
                sentence_token_count
            )

        if current_sentences:
            chunks.append(
                " ".join(
                    current_sentences
                )
            )

        return [
            chunk
            for chunk in chunks
            if chunk.strip()
        ]

    def summarize_chunk(
            self,
            chunk: str,
            chunk_number: int,
            total_chunks: int,
    ) -> str:
        """
        Pravi povezan tekstualni mini-rezime jednog dela.
        Ne koristi JSON jer ga Gemma često vrati neispravno.
        """

        prompt = f"""
    Ovo je deo {chunk_number} od ukupno {total_chunks} delova
    istog transkripta.

    Napravi precizan i povezan mini-rezime ovog dela na srpskom jeziku,
    ekavicom i latinicom.

    Pravila:

    - Napiši između 6 i 10 smislenih rečenica.
    - Obradi ceo dostavljeni deo, a ne samo njegov početak.
    - Sačuvaj konkretne nazive, institucije, smerove, procedure,
      uslove, mogućnosti, ograničenja i poređenja.
    - Sačuvaj sve važne brojeve, datume, procente, iznose i trajanja
      tačno kako su navedeni.
    - Ne izmišljaj podatke niti zaključke.
    - Ne dodaj informacije koje ne postoje u tekstu.
    - Ne koristi neodređene rečenice kao:
      "pričalo se o različitim temama" ili
      "navedene su brojne mogućnosti".
    - Ne ponavljaj istu informaciju.
    - Ne piši naslov, uvod, JSON, Markdown niti dodatno objašnjenje.
    - Vrati samo povezan tekst mini-rezimea.

    DEO TRANSKRIPTA:

    {chunk}
    """

        result = self._generate(
            prompt=prompt,
            max_new_tokens=700,
        )

        if not result.strip():
            raise ValueError(
                f"Model nije napravio rezime dela {chunk_number}."
            )

        return self._normalize_text(result)

    @staticmethod
    def _clean_list(
        values: Any,
    ) -> list[str]:
        """
        Čisti liste koje vraća model i
        uklanja potpuno iste stavke.
        """

        if not isinstance(values, list):
            return []

        result: list[str] = []
        seen: set[str] = set()

        for value in values:
            clean = re.sub(
                r"\s+",
                " ",
                str(value),
            ).strip(
                " -\n\t"
            )

            key = clean.casefold()

            if (
                clean
                and key not in seen
            ):
                result.append(clean)
                seen.add(key)

        return result

    @staticmethod
    def _clean_tasks(
        values: Any,
    ) -> list[dict[str, str]]:
        """
        Proverava da li zadaci imaju
        očekivanu strukturu.
        """

        if not isinstance(values, list):
            return []

        result: list[
            dict[str, str]
        ] = []

        seen: set[
            tuple[str, str, str]
        ] = set()

        for value in values:
            if not isinstance(value, dict):
                continue

            task = re.sub(
                r"\s+",
                " ",
                str(
                    value.get(
                        "task",
                        "",
                    )
                ),
            ).strip()

            owner = re.sub(
                r"\s+",
                " ",
                str(
                    value.get(
                        "owner",
                        "",
                    )
                ),
            ).strip()

            deadline = re.sub(
                r"\s+",
                " ",
                str(
                    value.get(
                        "deadline",
                        "",
                    )
                ),
            ).strip()

            if not task:
                continue

            key = (
                task.casefold(),
                owner.casefold(),
                deadline.casefold(),
            )

            if key not in seen:
                result.append(
                    {
                        "task": task,
                        "owner": owner,
                        "deadline": deadline,
                    }
                )

                seen.add(key)

        return result

    @staticmethod
    def _numbers(
        text: str,
    ) -> set[str]:
        """
        Izdvaja brojeve iz teksta.
        Zarez i tačka se tretiraju jednako
        radi lakšeg poređenja.
        """

        return {
            number.replace(",", ".")
            for number in re.findall(
                r"\d+(?:[.,]\d+)*",
                text,
            )
        }

    def _ground_numbered_statements(
        self,
        statements: list[str],
        transcript_text: str,
    ) -> list[str]:
        """
        Zadržava numeričku tvrdnju samo ako
        se svi brojevi iz nje nalaze u
        originalnom transkriptu.
        """

        source_numbers = self._numbers(
            transcript_text
        )

        grounded_statements: list[str] = []

        for statement in statements:
            statement_numbers = self._numbers(
                statement
            )

            if not statement_numbers:
                continue

            if statement_numbers.issubset(
                source_numbers
            ):
                grounded_statements.append(
                    statement
                )

        return grounded_statements

    def collect_chunk_summaries(
            self,
            transcript_text: str,
    ) -> list[dict[str, Any]]:
        """
        Deli transkript i pravi tekstualni mini-rezime svakog dela.
        """

        chunks = self.split_text(transcript_text)

        if not chunks:
            raise ValueError(
                "Transkript nema sadržaj za obradu."
            )

        print(
            f"Transkript je podeljen na {len(chunks)} delova."
        )

        results: list[dict[str, Any]] = []

        for index, chunk in enumerate(chunks, start=1):
            print(
                f"Sumiranje dela {index}/{len(chunks)}..."
            )

            try:
                chunk_summary = self.summarize_chunk(
                    chunk=chunk,
                    chunk_number=index,
                    total_chunks=len(chunks),
                )

            except Exception as error:
                print(
                    f"Prvi pokušaj za deo {index} nije uspeo: {error}"
                )
                print(
                    f"Ponovni pokušaj za deo {index}..."
                )

                try:
                    chunk_summary = self.summarize_chunk(
                        chunk=chunk,
                        chunk_number=index,
                        total_chunks=len(chunks),
                    )

                except Exception as second_error:
                    raise RuntimeError(
                        f"Deo {index} nije moguće sažeti ni posle "
                        f"ponovnog pokušaja: {second_error}"
                    ) from second_error

            results.append(
                {
                    "chunk_number": index,
                    "source_text": chunk,
                    "summary": chunk_summary,
                }
            )

        return results
    @staticmethod
    def _fix_section_headings(text: str) -> str:
        """
        Ispravlja česte varijacije i greške u nazivima sekcija
        koje model može da napravi.
        """

        heading_patterns = {
            r"(?im)^\s*naslov\s*:?\s*$":
                "NASLOV",

            r"(?im)^\s*vrsta\s+sadr[ažz]aja\s*:?\s*$":
                "VRSTA SADRŽAJA",

            r"(?im)^\s*kratak\s+(pregled|predlog)\s*:?\s*$":
                "KRATAK PREGLED",

            r"(?im)^\s*glavne\s+teme\s*:?\s*$":
                "GLAVNE TEME",

            r"(?im)^\s*važne\s+in(?:f)?ormacije\s*:?\s*$":
                "VAŽNE INFORMACIJE",

            r"(?im)^\s*važni\s+br(?:ojevi|vovi)\s+i\s+"
            r"detalji\s*:?\s*$":
                "VAŽNI BROJEVI I DETALJI",

            r"(?im)^\s*odluke\s*:?\s*$":
                "ODLUKE",

            r"(?im)^\s*zadaci\s+i\s+odgovornosti\s*:?\s*$":
                "ZADACI I ODGOVORNOSTI",

            r"(?im)^\s*otvorena\s+pitanja\s*:?\s*$":
                "OTVORENA PITANJA",

            r"(?im)^\s*zaklju[cčć]ak\s*:?\s*$":
                "ZAKLJUČAK",
        }

        for pattern, replacement in heading_patterns.items():
            text = re.sub(
                pattern,
                replacement,
                text,
            )

        return text.strip()
    def create_final_summary(
            self,
            transcript_text: str,
            chunk_results: list[dict[str, Any]],
    ) -> str:
        """
        Spaja mini-rezimee u konačan tekstualni rezime.
        """

        ordered_summaries: list[str] = []

        for item in chunk_results:
            part_number = item["chunk_number"]
            summary_text = str(
                item["summary"]
            ).strip()

            ordered_summaries.append(
                f"DEO {part_number}:\n{summary_text}"
            )

        all_summaries = "\n\n".join(
            ordered_summaries
        )

        prompt = f"""
    Napravi konačan rezime celog transkripta na osnovu
    mini-rezimea koji su navedeni ispod.

    Vrati samo gotov rezime u tačno ovom formatu:

    NASLOV
    Precizan naslov

    VRSTA SADRŽAJA
    predavanje, prezentacija, poslovni sastanak,
    intervju, diskusija ili drugo

    KRATAK PREGLED
    Povezan pregled od 6 do 9 rečenica.

    GLAVNE TEME
    - konkretna tema

    VAŽNE INFORMACIJE
    - konkretna i potpuna informacija

    VAŽNI BROJEVI I DETALJI
    - potpuna tvrdnja sa važnim brojem ili preciznim podatkom

    ODLUKE
    - odluka ili Nisu pomenute.

    ZADACI I ODGOVORNOSTI
    - zadatak ili Nisu pomenuti.

    OTVORENA PITANJA
    - pitanje ili Nisu pomenuta.

    ZAKLJUČAK
    Objektivan zaključak od 2 do 3 rečenice.

    [KRAJ_REZIMEA]

    STROGA PRAVILA:

    1. Koristi isključivo informacije iz mini-rezimea.
    2. Obuhvati početak, sredinu i kraj sadržaja.
    3. Ne izmišljaj podatke, nazive, brojeve ili zaključke.
    4. Sve brojeve prepiši potpuno isto kako su dati.
    5. Ne popravljaj i ne procenjuj brojeve na osnovu pretpostavke.
    6. Piši isključivo srpskom latinicom i ekavicom.
    7. Ne mešaj latinicu i ćirilicu.
    8. Koristi pravilna slova č, ć, š, ž i đ.
    9. Ne piši JSON, Markdown, kod, objašnjenje ili analizu odgovora.
    10. Ne procenjuj da li je odgovor dobar.
    11. Ne dodaj sadržaj o drugom fakultetu, ustanovi ili temi.
    12. Poslednja linija mora biti tačno:
    [KRAJ_REZIMEA]
    13. Posle oznake [KRAJ_REZIMEA] ne piši ništa.
    14. Navedi 6 do 10 glavnih tema.
    15. Navedi 10 do 16 važnih informacija.
    16. Ne ponavljaj istu informaciju u više formulacija.
    17. Ako su u pitanju predavanje ili prezentacija, odluke,
    zadaci i otvorena pitanja najčešće nisu pomenuti.

    MINI-REZIMEI REDOM:

    {all_summaries}
    """

        raw_result = self._generate(
            prompt=prompt,
            max_new_tokens=1500,
        )
        with open(
            "outputs/raw_final_model_output.txt",
            "w",
            encoding="utf-8",
        ) as file:
            file.write(raw_result)

        final_result = self._clean_final_output(
            text=raw_result,
            transcript_text=transcript_text,
        )
        final_result = self._fix_section_headings(
            final_result
        )

        required_sections = [
            "NASLOV",
            "VRSTA SADRŽAJA",
            "KRATAK PREGLED",
            "GLAVNE TEME",
            "VAŽNE INFORMACIJE",
            "VAŽNI BROJEVI I DETALJI",
            "ODLUKE",
            "ZADACI I ODGOVORNOSTI",
            "OTVORENA PITANJA",
            "ZAKLJUČAK",
        ]

        missing_sections = [
            section
            for section in required_sections
            if section not in final_result.upper()
        ]

        if missing_sections:
            print(
                "Upozorenje: model nije pravilno napisao "
                "sledeće sekcije: "
                + ", ".join(missing_sections)
            )

            with open(
                    "outputs/incomplete_final_summary.txt",
                    "w",
                    encoding="utf-8",
            ) as file:
                file.write(final_result)

        forbidden_fragments = [
            "Has the model",
            "Here's a breakdown",
            "Thank you for your feedback",
            "Ekonomski fakultet u Sarajevu",
            "```json",
        ]

        for fragment in forbidden_fragments:
            if fragment.casefold() in final_result.casefold():
                raise ValueError(
                    "Model je dodao nedozvoljen sadržaj: "
                    f"{fragment}"
                )

        return final_result
    @staticmethod
    def _format_bullets(
        items: list[str],
        empty_text: str,
    ) -> str:
        """
        Pretvara listu u tekst sa crticama.
        """

        if not items:
            return f"- {empty_text}"

        return "\n".join(
            f"- {item}"
            for item in items
        )

    def build_final_text(
        self,
        result: dict[str, Any],
    ) -> str:
        """
        Formatira završni JSON u čitljiv
        tekstualni rezime.
        """

        title = self._normalize_text(
            str(
                result.get(
                    "title",
                    "Sažetak sadržaja",
                )
            )
        )

        content_type = (
            self._normalize_text(
                str(
                    result.get(
                        "content_type",
                        "drugo",
                    )
                )
            )
        )

        overview = self._normalize_text(
            str(
                result.get(
                    "overview",
                    "",
                )
            )
        )

        conclusion = self._normalize_text(
            str(
                result.get(
                    "conclusion",
                    "",
                )
            )
        )

        topics = self._clean_list(
            result.get(
                "main_topics",
                [],
            )
        )[:10]

        important_information = (
            self._clean_list(
                result.get(
                    "important_information",
                    [],
                )
            )[:18]
        )

        numbers_and_details = (
            self._clean_list(
                result.get(
                    "numbers_and_details",
                    [],
                )
            )[:12]
        )

        decisions = self._clean_list(
            result.get(
                "decisions",
                [],
            )
        )

        tasks = self._clean_tasks(
            result.get(
                "tasks",
                [],
            )
        )

        open_questions = (
            self._clean_list(
                result.get(
                    "open_questions",
                    [],
                )
            )
        )

        normalized_type = (
            content_type.casefold()
        )

        # Predavanje i prezentacija uglavnom
        # nemaju poslovne zadatke i odluke.
        if normalized_type in {
            "predavanje",
            "prezentacija",
        }:
            decisions = []
            tasks = []
            open_questions = []

        task_lines: list[str] = []

        for task in tasks:
            line = task["task"]

            if task["owner"]:
                line += (
                    " | Odgovorna osoba: "
                    f"{task['owner']}"
                )

            if task["deadline"]:
                line += (
                    " | Rok: "
                    f"{task['deadline']}"
                )

            task_lines.append(line)

        sections = [
            "NASLOV",
            title,
            "",
            "VRSTA SADRŽAJA",
            content_type,
            "",
            "KRATAK PREGLED",
            (
                overview
                or (
                    "Pregled nije mogao "
                    "pouzdano da se formira."
                )
            ),
            "",
            "GLAVNE TEME",
            self._format_bullets(
                topics,
                "Nisu pouzdano izdvojene.",
            ),
            "",
            "VAŽNE INFORMACIJE",
            self._format_bullets(
                important_information,
                "Nisu pouzdano izdvojene.",
            ),
            "",
            "VAŽNI BROJEVI I DETALJI",
            self._format_bullets(
                numbers_and_details,
                "Nisu pomenuti.",
            ),
            "",
            "ODLUKE",
            self._format_bullets(
                decisions,
                "Nisu pomenute.",
            ),
            "",
            "ZADACI I ODGOVORNOSTI",
            self._format_bullets(
                task_lines,
                "Nisu pomenuti.",
            ),
            "",
            "OTVORENA PITANJA",
            self._format_bullets(
                open_questions,
                "Nisu pomenuta.",
            ),
            "",
            "ZAKLJUČAK",
            (
                conclusion
                or (
                    "Zaključak nije mogao "
                    "pouzdano da se formira."
                )
            ),
        ]

        return "\n".join(
            sections
        ).strip()

    def summarize_meeting(
            self,
            transcript_text: str,
    ) -> str:
        """
        Glavna metoda za sumarizaciju transkripta.
        """

        os.makedirs(
            "outputs",
            exist_ok=True,
        )

        chunk_results = (
            self.collect_chunk_summaries(
                transcript_text
            )
        )

        with open(
                "outputs/chunk_summaries.json",
                "w",
                encoding="utf-8",
        ) as file:
            json.dump(
                chunk_results,
                file,
                ensure_ascii=False,
                indent=2,
            )

        print(
            "Kreiranje konačnog rezimea..."
        )

        final_summary = (
            self.create_final_summary(
                transcript_text=transcript_text,
                chunk_results=chunk_results,
            )
        )

        with open(
                "outputs/final_summary_debug.txt",
                "w",
                encoding="utf-8",
        ) as file:
            file.write(
                final_summary
            )

        return final_summary


def summarize_transcript_file(
    transcript_path: str,
    summary_path: str,
) -> str:
    """
    Učitava transkript iz fajla,
    pokreće sumarizaciju i čuva rezultat.
    """

    if not os.path.exists(
        transcript_path
    ):
        raise FileNotFoundError(
            "Transkript nije pronađen: "
            f"{transcript_path}"
        )

    with open(
        transcript_path,
        "r",
        encoding="utf-8",
    ) as file:
        transcript_text = (
            file.read().strip()
        )

    if not transcript_text:
        raise ValueError(
            "Transkript je prazan."
        )

    summarizer = MeetingSummarizer()

    summary = (
        summarizer.summarize_meeting(
            transcript_text
        )
    )

    output_directory = os.path.dirname(
        summary_path
    )

    if output_directory:
        os.makedirs(
            output_directory,
            exist_ok=True,
        )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(summary)

    print(
        "Rezime je sačuvan u: "
        f"{summary_path}"
    )

    return summary
