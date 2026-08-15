import json
import os
import re
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Gemma3ForConditionalGeneration,
)


@dataclass
class SummaryResult: #struktura konacnog sazetka
    title: str
    content_type: str
    short_overview: str
    main_topics: list[str]
    important_information: list[str]
    numbers_and_dates: list[str]
    conclusions: list[str]
    action_items: list[str]
    unclear_parts: list[str]


class GemmaTranscriptSummarizer: #glavna klasa, nas ceo gemma sumarizer
    CHUNK_SECTIONS = [
        "GLAVNE_TEME",
        "VAZNE_INFORMACIJE",
        "BROJEVI_I_DATUMI",
        "ZAKLJUCCI",
        "AKCIONE_STAVKE",
        "NEJASNI_DELOVI",
    ]

    SECTION_PATTERN = re.compile(
        r"^(?:[-*•]\s*)?\[([A-Z_]+)\]\s*:?\s*$"
    )

    CYRILLIC_TO_LATIN = str.maketrans(
        {
            "А": "A", "Б": "B", "В": "V", "Г": "G", "Д": "D",
            "Ђ": "Đ", "Е": "E", "Ж": "Ž", "З": "Z", "И": "I",
            "Ј": "J", "К": "K", "Л": "L", "Љ": "Lj", "М": "M",
            "Н": "N", "Њ": "Nj", "О": "O", "П": "P", "Р": "R",
            "С": "S", "Т": "T", "Ћ": "Ć", "У": "U", "Ф": "F",
            "Х": "H", "Ц": "C", "Ч": "Č", "Џ": "Dž", "Ш": "Š",
            "а": "a", "б": "b", "в": "v", "г": "g", "д": "d",
            "ђ": "đ", "е": "e", "ж": "ž", "з": "z", "и": "i",
            "ј": "j", "к": "k", "л": "l", "љ": "lj", "м": "m",
            "н": "n", "њ": "nj", "о": "o", "п": "p", "р": "r",
            "с": "s", "т": "t", "ћ": "ć", "у": "u", "ф": "f",
            "х": "h", "ц": "c", "ч": "č", "џ": "dž", "ш": "š",
        }
    )

    SAFE_TEXT_REPLACEMENTS = {
        "bolenskom deklaracijom": "Bolonjskom deklaracijom",
        "bolenska deklaracija": "Bolonjska deklaracija",
        "diagnostičkih": "dijagnostičkih",
        "diagnostički": "dijagnostički",
        "prosena ocena": "prosečna ocena",
        "računostka tehnika": "računarska tehnika",
        "računostku tehniku": "računarsku tehniku",
        "elektrotehničarstvo": "elektrotehnika",
        "maskebal": "maskenbal",
        "kafinac": "kafić",
        "juni": "jun",
        "studenta diplomira": "studenata diplomira",
        "480-500 studenta": "480–500 studenata",
        "480 do 500 studenta": "480 do 500 studenata",
        "1200€": "1.200 evra",
        "24/7": "24 sata dnevno",
        "robotička inženjerija": "robotika",
        "biomedicinska inženjerija": "biomedicina",
    }

    def __init__(
        self,
        model_name: str = "google/gemma-3-4b-it",
    ) -> None:
        self.model_name = model_name

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA nije dostupna. Gemma 3 4B je u ovoj verziji "
                "podesena za rad na NVIDIA GPU-u."
            )

        print("Sumarizacija ce koristiti GPU.")
        print(f"Ucitavanje modela: {self.model_name}")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        self.processor = AutoProcessor.from_pretrained(self.model_name)

        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        self.model.eval() #model se koristi za inference, ne za treniranje.

        print("Gemma 3 model je uspesno ucitan.")

    @classmethod
    def to_latin(cls, text: str) -> str:
        return text.translate(cls.CYRILLIC_TO_LATIN)

    @classmethod
    def normalize_text(cls, text: str) -> str:
        text = cls.to_latin(text)
        text = text.replace("\u00a0", " ")
        text = text.replace("\u200b", "")
        text = text.replace("**", "")
        text = re.sub(r"(?m)^\s*[*#]+\s*", "", text)

        for wrong, correct in cls.SAFE_TEXT_REPLACEMENTS.items():
            text = re.sub(
                re.escape(wrong),
                correct,
                text,
                flags=re.IGNORECASE, #ignorsi velika i mala slova
            )

        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r" *\n *", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def normalize_item_for_comparison(text: str) -> str:
        text = text.casefold()
        text = re.sub(r"[^a-z0-9čćžšđ]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def count_tokens(self, text: str) -> int:
        return len(
            self.processor.tokenizer.encode( #pretvara tekst u listu token ID-eva.
                text,
                add_special_tokens=False,
            )
        )

    @staticmethod
    def split_into_sentences(text: str) -> list[str]: #podeli veliki transkript na recenice
        protected_text = text

        abbreviations = {
            "dr.": "dr<DOT>",
            "mr.": "mr<DOT>",
            "prof.": "prof<DOT>",
            "npr.": "npr<DOT>",
            "tj.": "tj<DOT>",
            "itd.": "itd<DOT>",
            "br.": "br<DOT>",
        }

        for abbreviation, replacement in abbreviations.items():
            protected_text = re.sub(
                re.escape(abbreviation),
                replacement,
                protected_text,
                flags=re.IGNORECASE,
            )

        protected_text = re.sub(
            r"(\d+)\.\s+"
            r"(?=(?:godine|godinu|razreda|mesta|mjesta|"
            r"su|je|bio|bila|bili|god)\b)",
            r"\1<ORDINAL> ",
            protected_text,
            flags=re.IGNORECASE,
        )

        #podela recenica
        raw_sentences = re.split(
            r"(?<=[!?])\s+"
            r"|(?<=[A-Za-zČĆŽŠĐčćžšđ])\.\s+(?=[A-ZČĆŽŠĐ])"
            r"|\n+",
            protected_text,
        )

        #ciscenje recenica
        sentences: list[str] = []
        for sentence in raw_sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence = sentence.replace("<DOT>", ".")
            sentence = sentence.replace("<ORDINAL>", ".")
            sentences.append(sentence)

        return sentences

    def split_into_chunks( #veliki transkript podeli na delove od najvise priblizno 1800 tokena
        self,
        transcript: str,
        max_tokens: int = 1800,
    ) -> list[str]:
        sentences = self.split_into_sentences(transcript)

        chunks: list[str] = []
        current_sentences: list[str] = []
        current_token_count = 0

        for sentence in sentences:
            sentence_token_count = self.count_tokens(sentence)

            if sentence_token_count > max_tokens:
                if current_sentences:
                    chunks.append(" ".join(current_sentences))
                    current_sentences = []
                    current_token_count = 0

                chunks.extend(
                    self.split_long_text(
                        sentence,
                        max_tokens=max_tokens,
                    )
                )
                continue

            if (
                current_sentences
                and current_token_count + sentence_token_count > max_tokens
            ):
                chunks.append(" ".join(current_sentences))
                current_sentences = []
                current_token_count = 0

            current_sentences.append(sentence)
            current_token_count += sentence_token_count

        if current_sentences:
            chunks.append(" ".join(current_sentences))

        return chunks

    def split_long_text(
        self,
        text: str,
        max_tokens: int,
    ) -> list[str]:
        words = text.split()
        chunks: list[str] = []
        current_words: list[str] = []

        for word in words:
            candidate = " ".join(current_words + [word])

            if current_words and self.count_tokens(candidate) > max_tokens:
                chunks.append(" ".join(current_words))
                current_words = [word]
            else:
                current_words.append(word)

        if current_words:
            chunks.append(" ".join(current_words))

        return chunks

    @staticmethod
    def extract_numbers(text: str) -> list[str]:
        """
        Izdvaja pojedinacne brojeve. Opsege poput 480-500 i izraze
        poput 24/7 razdvaja na 480, 500, odnosno 24 i 7.
        """
        return re.findall(r"\d{1,3}(?:[.,]\d{3})+|\d+", text)

    @staticmethod
    def normalize_number(value: str) -> str:
        value = value.strip()

        if re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+", value):
            return re.sub(r"[.,]", "", value)

        return value.lstrip("0") or "0"

    def compare_numbers(
        self,
        source_text: str,
        summary_text: str,
    ) -> dict[str, Any]:
        """
        Proverava da li se svaki broj iz sazetka makar jednom pojavljuje
        u transkriptu. Ne prijavljuje gresku samo zato sto je isti broj
        ponovljen vise puta u sazetku.
        """
        source_numbers = {
            self.normalize_number(value)
            for value in self.extract_numbers(source_text)
        }
        summary_numbers = {
            self.normalize_number(value)
            for value in self.extract_numbers(summary_text)
        }

        added_numbers = sorted(
            summary_numbers - source_numbers,
            key=lambda item: (len(item), item),
        )

        return {
            "source_numbers": sorted(
                source_numbers,
                key=lambda item: (len(item), item),
            ),
            "summary_numbers": sorted(
                summary_numbers,
                key=lambda item: (len(item), item),
            ),
            "added_numbers": added_numbers,
            "has_invented_numbers": bool(added_numbers),
        }

    @staticmethod
    def build_chunk_prompt( #samo pravi tekst prompta koji cemo Gemmi poslati
        chunk: str,
        chunk_number: int,
        total_chunks: int,
    ) -> str:
        return f"""
Analiziraj deo {chunk_number} od ukupno {total_chunks} delova automatski
transkribovanog govora na srpskom jeziku.

Izdvoji samo informacije koje se zaista pojavljuju u tekstu.

Pravila:
- Piši na srpskom jeziku, ekavicom i isključivo latinicom.
- Ne prevodi na engleski.
- Ne izmišljaj činjenice i ne dodaj opšte znanje.
- Ne menjaj smisao rečenica.
- Sačuvaj važna imena, nazive, brojeve, datume, procente, iznose i rangove.
- Očiglednu grešku transkripcije protumači samo kada je značenje potpuno jasno.
- Kada značenje nije jasno, navedi sporni izraz u sekciji NEJASNI_DELOVI.
- Akcione stavke navedi samo kada tekst sadrži konkretnu obavezu, rok ili zadatak
  koji neko treba da izvrši posle ovog razgovora.
- Informacije o postupku upisa nisu akcione stavke same po sebi.
- Svaka stavka mora početi znakom "-".
- Ne piši uvod, završni komentar, dodatne sekcije ili Markdown naslov.
- Svaku sekciju navedi tačno jednom i tačno ovim redosledom.

[GLAVNE_TEME]
- tema

[VAZNE_INFORMACIJE]
- konkretna informacija

[BROJEVI_I_DATUMI]
- broj ili datum sa objašnjenjem

[ZAKLJUCCI]
- zaključak direktno podržan tekstom

[AKCIONE_STAVKE]
- konkretna buduća obaveza ili zadatak

[NEJASNI_DELOVI]
- nejasan ili verovatno pogrešno transkribovan izraz

Kada neka sekcija nema stavke, napiši samo njenu oznaku i pređi na sledeću.

TEKST:
{chunk}
""".strip()

    @staticmethod
    def build_overview_prompt( #ovo je drugi promt
        main_topics: list[str],
        important_information: list[str],
        numbers_and_dates: list[str],
    ) -> str:
        source_data = {
            "glavne_teme": main_topics,
            "vazne_informacije": important_information,
            "brojevi_i_datumi": numbers_and_dates,
        }

        return f"""
Na osnovu datih beleški napiši samo naslov, vrstu sadržaja i kratak pregled.

Pravila:
- Piši na srpskom jeziku, ekavicom i isključivo latinicom.
- Koristi samo informacije iz beleški.
- Ne dodaj nove činjenice, datume, rokove ili brojeve.
- Ne pretvaraj godinu statistike u godinu događaja ili roka.
- Strogo razlikuj školarinu, cenu i početnu platu; nikada ih ne spajaj.
- Ne menjaj nazive studijskih programa i ne izmišljaj nove oblasti.
- Ne koristi podebljanje, zvezdice ni Markdown.
- Kratak pregled treba da ima od 4 do 6 jasnih rečenica.
- Pregled mora da obuhvati ceo sadržaj, a ne samo jednu temu.
- Ne navodi svaki broj pojedinačno.
- Ne piši Markdown, uvod ili završni komentar.
- Koristi tačno sledeći format:

[NASLOV]
kratak informativan naslov

[VRSTA_SADRZAJA]
predavanje, sastanak, prezentacija, intervju ili drugo

[KRATAK_PREGLED]
tekst od 4 do 6 rečenica

BELEŠKE:
{json.dumps(source_data, ensure_ascii=False, indent=2)}
""".strip()

    def generate_text( #centralna funkcija koja stvarno razgovara sa Gemmom
        self,
        prompt: str,
        max_new_tokens: int,
    ) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    }
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        model_device = next(self.model.parameters()).device

        #Sve ulazne tensore prebacujemo na isti uredjaj
        inputs = {
            key: value.to(model_device)
            for key, value in inputs.items()
        }

        input_ids = inputs["input_ids"]
        vocab_size = self.model.get_input_embeddings().num_embeddings

        invalid_mask = (input_ids < 0) | (input_ids >= vocab_size)
        if invalid_mask.any():
            invalid_ids = torch.unique(input_ids[invalid_mask]).tolist()
            print(
                "Pronađeni nevažeći token ID-jevi; biće zamenjeni nulom:",
                invalid_ids,
            )
            input_ids = input_ids.clone()
            input_ids[invalid_mask] = 0
            inputs["input_ids"] = input_ids

        input_length = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.03,
                use_cache=True,
            )

        generated_tokens = generated[0, input_length:]
        output_text = self.processor.decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()

        if not output_text:
            raw_output = self.processor.decode(
                generated_tokens,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            raise ValueError(
                "Model je vratio prazan odgovor. "
                f"Sirovi izlaz: {raw_output[:500]!r}"
            )

        return self.normalize_text(output_text)

    @classmethod
    def parse_section_output(
            cls,
            text: str,
            expected_sections: list[str],
    ) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {
            section: []
            for section in expected_sections
        }

        normalized_text = cls.normalize_text(text)

        if not normalized_text:
            raise ValueError("Model je vratio prazan odgovor.")

        empty_values = {
            "nema",
            "(nema)",
            "nema stavki",
            "nema izdvojenih stavki",
        }

        # -------------------------------------------------
        # 1. Pokušaj parsiranja JSON odgovora
        # -------------------------------------------------

        json_text = normalized_text.strip()

        # Gemma ponekad obavije JSON u:
        # ```json
        # ...
        # ```
        json_text = re.sub(
            r"^```(?:json)?\s*",
            "",
            json_text,
            flags=re.IGNORECASE,
        )

        json_text = re.sub(
            r"\s*```$",
            "",
            json_text,
        )

        try:
            parsed_json = json.loads(json_text)

            # -------------------------------------------------
            # JSON može biti:
            #
            # {
            #   "GLAVNE_TEME": [...],
            #   ...
            # }
            #
            # ili:
            #
            # [
            #   {"GLAVNE_TEME": [...]},
            #   {"VAZNE_INFORMACIJE": [...]},
            #   ...
            # ]
            # -------------------------------------------------

            if isinstance(parsed_json, list):
                merged_json = {}

                for item in parsed_json:
                    if isinstance(item, dict):
                        merged_json.update(item)

                parsed_json = merged_json

            if isinstance(parsed_json, dict):
                for section in expected_sections:
                    value = parsed_json.get(
                        section,
                        [],
                    )

                    if isinstance(value, list):
                        cleaned_items: list[str] = []

                        for item in value:
                            clean_item = cls.normalize_text(
                                str(item)
                            )

                            if (
                                    clean_item
                                    and clean_item.casefold()
                                    not in empty_values
                            ):
                                cleaned_items.append(
                                    clean_item
                                )

                        result[section] = cleaned_items

                    elif isinstance(value, str):
                        clean_value = cls.normalize_text(
                            value
                        )

                        if (
                                clean_value
                                and clean_value.casefold()
                                not in empty_values
                        ):
                            result[section] = [
                                clean_value
                            ]

                if any(result.values()):
                    return result

        except json.JSONDecodeError:
            pass

        # -------------------------------------------------
        # 2. Parsiranje tekstualnog formata
        # -------------------------------------------------

        current_section: str | None = None

        for raw_line in normalized_text.splitlines():
            line = raw_line.strip()

            if not line:
                continue

            # Podržava:
            #
            # [GLAVNE_TEME]
            #
            # ali i:
            #
            # - [GLAVNE_TEME]
            #
            # kao i:
            #
            # [GLAVNE_TEME] neka tema
            #
            # ili:
            #
            # - [GLAVNE_TEME] neka tema

            section_match = re.match(
                r"^(?:[-*•]\s*)?"
                r"\[([A-Z_]+)\]"
                r"\s*:?\s*(.*)$",
                line,
            )

            if section_match:
                section_name = section_match.group(1)
                inline_content = section_match.group(2).strip()

                current_section = (
                    section_name
                    if section_name in result
                    else None
                )

                if current_section is None:
                    continue

                # Ako se sadržaj nalazi u istom redu
                # kao oznaka sekcije.
                if inline_content:
                    inline_content = re.sub(
                        r"^[-*•]\s*",
                        "",
                        inline_content,
                    ).strip()

                    if (
                            inline_content
                            and inline_content.casefold()
                            not in empty_values
                    ):
                        result[current_section].append(
                            inline_content
                        )

                continue

            # Ako još nismo pronašli poznatu sekciju,
            # ovu liniju ignorišemo.
            if current_section is None:
                continue

            # Uklanja bullet oznaku sa početka stavke.
            line = re.sub(
                r"^[-*•]\s*",
                "",
                line,
            ).strip()

            if not line:
                continue

            if line.casefold() in empty_values:
                continue

            result[current_section].append(line)

        return result

    @classmethod
    def parse_overview_output(
            cls,
            text: str,
    ) -> dict[str, str]:
        """
        Parsira završni Gemma odgovor.

        Podržava:
        [NASLOV]
        NASLOV
        [NASLOV] sadržaj
        NASLOV: sadržaj

        i isto za VRSTA_SADRZAJA i KRATAK_PREGLED.
        """

        normalized_text = cls.normalize_text(text)

        result = {
            "title": "",
            "content_type": "",
            "overview": "",
        }

        section_map = {
            "NASLOV": "title",
            "VRSTA_SADRZAJA": "content_type",
            "VRSTA SADRZAJA": "content_type",
            "KRATAK_PREGLED": "overview",
            "KRATAK PREGLED": "overview",
        }

        collected: dict[str, list[str]] = {
            "title": [],
            "content_type": [],
            "overview": [],
        }

        current_key: str | None = None

        for raw_line in normalized_text.splitlines():
            line = raw_line.strip()

            if not line:
                continue

            # Uklanja eventualni bullet sa početka.
            line = re.sub(
                r"^[-*•]\s*",
                "",
                line,
            ).strip()

            # ---------------------------------------------
            # 1. Format:
            #
            # [NASLOV]
            # [VRSTA_SADRZAJA]
            # [KRATAK_PREGLED]
            #
            # ali podržava i:
            #
            # [NASLOV] neki naslov
            # ---------------------------------------------

            bracket_match = re.match(
                r"^\["
                r"(NASLOV|VRSTA_SADRZAJA|KRATAK_PREGLED)"
                r"\]"
                r"\s*:?\s*(.*)$",
                line,
                flags=re.IGNORECASE,
            )

            if bracket_match:
                section_name = (
                    bracket_match
                    .group(1)
                    .upper()
                )

                inline_content = (
                    bracket_match
                    .group(2)
                    .strip()
                )

                current_key = section_map.get(
                    section_name
                )

                if (
                        current_key
                        and inline_content
                ):
                    collected[current_key].append(
                        inline_content
                    )

                continue

            # ---------------------------------------------
            # 2. Format bez uglastih zagrada:
            #
            # NASLOV
            # naslov
            #
            # ili:
            #
            # NASLOV: naslov
            # ---------------------------------------------

            plain_match = re.match(
                r"^"
                r"(NASLOV|"
                r"VRSTA_SADRZAJA|VRSTA SADRZAJA|"
                r"KRATAK_PREGLED|KRATAK PREGLED)"
                r"\s*:?\s*(.*)$",
                line,
                flags=re.IGNORECASE,
            )

            if plain_match:
                section_name = (
                    plain_match
                    .group(1)
                    .upper()
                )

                inline_content = (
                    plain_match
                    .group(2)
                    .strip()
                )

                current_key = section_map.get(
                    section_name
                )

                if (
                        current_key
                        and inline_content
                ):
                    collected[current_key].append(
                        inline_content
                    )

                continue

            # ---------------------------------------------
            # 3. Ako smo već u nekoj sekciji,
            # dodajemo tekst toj sekciji.
            # ---------------------------------------------

            if current_key is not None:
                collected[current_key].append(
                    line
                )

        # Spajanje linija svake sekcije.
        for key, values in collected.items():
            result[key] = " ".join(
                values
            ).strip()

        # ---------------------------------------------
        # Bezbedan fallback.
        #
        # Ako Gemma nije vratila nijednu oznaku,
        # ceo odgovor koristimo samo kao overview.
        # ---------------------------------------------

        if (
                not result["title"]
                and not result["content_type"]
                and not result["overview"]
                and normalized_text
        ):
            fallback_lines = []

            for raw_line in normalized_text.splitlines():
                line = raw_line.strip()

                if not line:
                    continue

                line = re.sub(
                    r"^[-*•]\s*",
                    "",
                    line,
                ).strip()

                if line:
                    fallback_lines.append(line)

            result["overview"] = " ".join(
                fallback_lines
            ).strip()

        return result

    def summarize_chunk(
        self,
        chunk: str,
        chunk_number: int,
        total_chunks: int,
    ) -> dict[str, Any]:
        prompt = self.build_chunk_prompt(
            chunk=chunk,
            chunk_number=chunk_number,
            total_chunks=total_chunks,
        )

        output = self.generate_text(
            prompt=prompt,
            max_new_tokens=1100,
        )

        os.makedirs("outputs/debug_raw", exist_ok=True)
        raw_path = f"outputs/debug_raw/chunk_{chunk_number}_raw.txt"
        with open(raw_path, "w", encoding="utf-8") as file:
            file.write(output)

        sections = self.parse_section_output(
            text=output,
            expected_sections=self.CHUNK_SECTIONS,
        )

        parsed_item_count = sum(len(items) for items in sections.values())
        if parsed_item_count == 0:
            raise ValueError(
                "Model je vratio tekst, ali parser nije pronašao nijednu "
                f"poznatu sekciju. Proveri fajl: {raw_path}"
            )

        return {
            "glavne_teme": sections["GLAVNE_TEME"],
            "vazne_informacije": sections["VAZNE_INFORMACIJE"],
            "brojevi_i_datumi": sections["BROJEVI_I_DATUMI"],
            "zakljucci": sections["ZAKLJUCCI"],
            "akcione_stavke": sections["AKCIONE_STAVKE"],
            "nejasni_delovi": sections["NEJASNI_DELOVI"],
        }

    @staticmethod
    def split_oversized_item(item: str, max_length: int = 260) -> list[str]:
        """Deli samo predugacke stavke na citljive delove."""
        if len(item) <= max_length:
            return [item]

        parts = re.split(r"\s*[;]\s*|,\s+(?=[A-ZČĆŽŠĐ0-9])", item)
        parts = [part.strip(" .,-") for part in parts if part.strip(" .,-")]

        if len(parts) <= 1:
            return [item]

        result: list[str] = []
        current = ""

        for part in parts:
            candidate = f"{current}, {part}" if current else part
            if current and len(candidate) > max_length:
                result.append(current.rstrip(" ,") + ".")
                current = part
            else:
                current = candidate

        if current:
            result.append(current.rstrip(" ,") + ".")

        return result

    def collect_unique_items(
        self,
        partial_summaries: list[dict[str, Any]],
        key: str,
        limit: int | None = None,
    ) -> list[str]:
        values: list[str] = []
        normalized_values: list[str] = []

        for partial in partial_summaries:
            items = partial.get(key, [])
            if not isinstance(items, list):
                continue

            for item in items:
                clean_item = self.normalize_text(str(item))
                if not clean_item:
                    continue

                for expanded_item in self.split_oversized_item(clean_item):
                    normalized = self.normalize_item_for_comparison(
                        expanded_item
                    )
                    if not normalized:
                        continue

                    # Uklanja tačne duplikate i stavke koje su gotovo u celosti
                    # već sadržane u drugoj, dužoj stavci.
                    duplicate = False
                    for existing in normalized_values:
                        shorter = min(len(normalized), len(existing))
                        if normalized == existing:
                            duplicate = True
                            break
                        if shorter >= 35 and (
                            normalized in existing or existing in normalized
                        ):
                            duplicate = True
                            break

                    if duplicate:
                        continue

                    normalized_values.append(normalized)
                    values.append(expanded_item.rstrip(" .") + ".")

                    if limit is not None and len(values) >= limit:
                        return values

        return values

    @staticmethod
    def filter_action_items(items: list[str]) -> list[str]:
        action_markers = (
            "potrebno je",
            "treba da",
            "dogovoreno je",
            "rok je",
            "zadužen",
            "zadužena",
            "obaveza",
            "poslati",
            "uraditi",
            "pripremiti",
            "dostaviti",
            "proveriti",
            "ispraviti",
            "zakazati",
            "kontaktirati",
            "javiti",
        )

        excluded_procedure_markers = (
            "prijemni",
            "upis",
            "elektronski prijaviti",
            "konačna prijava",
            "rang list",
        )

        filtered: list[str] = []
        for item in items:
            normalized = item.casefold()

            if any(
                marker in normalized
                for marker in excluded_procedure_markers
            ):
                continue

            if any(marker in normalized for marker in action_markers):
                filtered.append(item)

        return filtered

    def create_final_summary(
        self,
        partial_summaries: list[dict[str, Any]],
    ) -> SummaryResult:
        main_topics = self.collect_unique_items(
            partial_summaries,
            "glavne_teme",
            limit=8,
        )
        important_information = self.collect_unique_items(
            partial_summaries,
            "vazne_informacije",
            limit=28,
        )
        numbers_and_dates = self.collect_unique_items(
            partial_summaries,
            "brojevi_i_datumi",
            limit=24,
        )
        conclusions = self.collect_unique_items(
            partial_summaries,
            "zakljucci",
            limit=6,
        )
        unclear_parts = self.collect_unique_items(
            partial_summaries,
            "nejasni_delovi",
            limit=8,
        )
        raw_action_items = self.collect_unique_items(
            partial_summaries,
            "akcione_stavke",
            limit=10,
        )
        action_items = self.filter_action_items(raw_action_items)

        overview_prompt = self.build_overview_prompt(
            main_topics=main_topics,
            important_information=important_information,
            numbers_and_dates=numbers_and_dates,
        )
        try:
            overview_output = self.generate_text(
                prompt=overview_prompt,
                max_new_tokens=450,
            )
        except ValueError as error:
            print(
                "Upozorenje: model nije vratio završni pregled. "
                "Biće napravljen bezbedan pregled iz izdvojenih beleški."
            )
            print(str(error))
            overview_output = ""

        os.makedirs("outputs/debug_raw", exist_ok=True)
        with open(
            "outputs/debug_raw/final_overview_raw.txt",
            "w",
            encoding="utf-8",
        ) as file:
            file.write(overview_output)

        overview_data = self.parse_overview_output(overview_output)
        overview_data = {
            key: self.normalize_text(value)
            for key, value in overview_data.items()
        }

        inferred_title = (
            main_topics[0]
            if main_topics
            else "Pregled sadržaja transkripta"
        )
        title = self.normalize_text(
            overview_data["title"] or inferred_title
        ).strip(" -*#")

        combined_for_type = " ".join(
            main_topics + important_information[:5]
        ).casefold()
        if "sastanak" in combined_for_type:
            inferred_type = "sastanak"
        elif "intervju" in combined_for_type:
            inferred_type = "intervju"
        elif "predavanje" in combined_for_type or "fakultet" in combined_for_type:
            inferred_type = "predavanje"
        elif "prezent" in combined_for_type or "predstavljanje" in combined_for_type:
            inferred_type = "prezentacija"
        else:
            inferred_type = "drugo"

        content_type = self.normalize_text(
            overview_data["content_type"] or inferred_type
        ).strip(" -*#")
        short_overview = self.normalize_text(
            overview_data["overview"]
        ).strip(" -*#")

        if not short_overview:
            fallback_sentences: list[str] = []

            if main_topics:
                topics_text = ", ".join(main_topics[:4])
                fallback_sentences.append(
                    f"Sadržaj obrađuje sledeće glavne teme: {topics_text}."
                )

            for item in important_information[:4]:
                sentence = item.strip().rstrip(".!?")
                if sentence:
                    fallback_sentences.append(sentence + ".")

            if not fallback_sentences and numbers_and_dates:
                for item in numbers_and_dates[:3]:
                    sentence = item.strip().rstrip(".!?")
                    if sentence:
                        fallback_sentences.append(sentence + ".")

            short_overview = " ".join(fallback_sentences).strip()

        if not short_overview:
            short_overview = (
                "Iz transkripta su izdvojene glavne teme i važne informacije, "
                "ali model nije uspeo da formira poseban kratak pregled."
            )

        return SummaryResult(
            title=title,
            content_type=content_type,
            short_overview=short_overview,
            main_topics=main_topics,
            important_information=important_information,
            numbers_and_dates=numbers_and_dates,
            conclusions=conclusions,
            action_items=action_items,
            unclear_parts=unclear_parts,
        )

    @staticmethod
    def format_summary(summary: SummaryResult) -> str:
        def section(title: str, values: list[str]) -> str:
            lines = [title]
            if not values:
                lines.append("- Nema izdvojenih stavki.")
            else:
                lines.extend(f"- {value}" for value in values)
            return "\n".join(lines)

        sections = [
            "NASLOV",
            summary.title,
            "",
            "VRSTA SADRŽAJA",
            summary.content_type,
            "",
            "KRATAK PREGLED",
            summary.short_overview,
            "",
            section("GLAVNE TEME", summary.main_topics),
            "",
            section("VAŽNE INFORMACIJE", summary.important_information),
            "",
            section("BROJEVI I DATUMI", summary.numbers_and_dates),
            "",
            section("ZAKLJUČCI", summary.conclusions),
            "",
            section("AKCIONE STAVKE", summary.action_items),
            "",
            section(
                "NEJASNI DELOVI TRANSKRIPTA",
                summary.unclear_parts,
            ),
        ]

        return "\n".join(sections).strip()

    def summarize_transcript(
        self,
        transcript: str,
    ) -> tuple[
        SummaryResult,
        list[dict[str, Any]],
        dict[str, Any],
    ]:
        transcript = self.normalize_text(transcript)
        chunks = self.split_into_chunks(
            transcript,
            max_tokens=1800,
        )

        if not chunks:
            raise ValueError("Transkript nema sadržaj za sumarizaciju.")

        print(f"Transkript je podeljen na {len(chunks)} delova.")

        partial_summaries: list[dict[str, Any]] = []
        debug_chunks: list[dict[str, Any]] = []

        for index, chunk in enumerate(chunks, start=1):
            print(f"Sumarizacija dela {index}/{len(chunks)}...")

            partial_summary = self.summarize_chunk(
                chunk=chunk,
                chunk_number=index,
                total_chunks=len(chunks),
            )
            partial_summaries.append(partial_summary)

            debug_chunks.append(
                {
                    "chunk_number": index,
                    "source_text": chunk,
                    "source_token_count": self.count_tokens(chunk),
                    "partial_summary": partial_summary,
                }
            )

        print("Kreiranje konačnog sažetka...")
        final_summary = self.create_final_summary(partial_summaries)
        formatted_summary = self.format_summary(final_summary)

        number_check = self.compare_numbers(
            source_text=transcript,
            summary_text=formatted_summary,
        )

        return final_summary, debug_chunks, number_check


def summarize_transcript_file(
    input_path: str,
    output_path: str,
    json_output_path: str,
    debug_path: str,
) -> str:
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Transkript nije pronađen: {input_path}"
        )

    with open(input_path, "r", encoding="utf-8") as file:
        transcript = file.read().strip()

    if not transcript:
        raise ValueError("Transkript je prazan.")

    start_time = time.time()
    summarizer = GemmaTranscriptSummarizer()

    summary_result, debug_chunks, number_check = (
        summarizer.summarize_transcript(transcript)
    )
    formatted_summary = summarizer.format_summary(summary_result)

    for path in (output_path, json_output_path, debug_path):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(formatted_summary)

    with open(json_output_path, "w", encoding="utf-8") as file:
        json.dump(
            asdict(summary_result),
            file,
            ensure_ascii=False,
            indent=2,
        )

    with open(debug_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "model_name": summarizer.model_name,
                "chunks": debug_chunks,
                "number_check": number_check,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    elapsed_time = time.time() - start_time

    print()
    print(f"Sažetak je sačuvan u: {output_path}")
    print(f"JSON sažetak je sačuvan u: {json_output_path}")
    print(f"Debug podaci su sačuvani u: {debug_path}")

    if number_check["has_invented_numbers"]:
        print(
            "Upozorenje: sažetak sadrži brojeve koji nisu pronađeni "
            "u transkriptu:"
        )
        print(number_check["added_numbers"])
    else:
        print("Provera brojeva: nisu pronađeni novi brojevi.")

    print(f"Vreme sumarizacije: {elapsed_time:.2f} sekundi.")
    return formatted_summary
