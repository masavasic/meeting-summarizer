import numpy as np
import torch
import torchaudio

from sklearn.metrics.pairwise import cosine_similarity

from speechbrain.inference.speaker import (
    SpeakerRecognition,
)
from speechbrain.utils.fetching import (
    LocalStrategy,
)

from processing.realtime_diarization import (
    diarize_buffer,
)


SPEECHBRAIN_MODEL = (
    "speechbrain/spkrec-ecapa-voxceleb"
)


class BufferedSpeakerDiarizer:
    def __init__(
        self,
        buffer_seconds: float = 30.0,
        step_seconds: float = 20.0,
        min_embedding_duration: float = 1.5,
    ) -> None:

        self.buffer_seconds = buffer_seconds
        self.step_seconds = step_seconds

        self.overlap_seconds = (
            buffer_seconds
            - step_seconds
        )

        self.min_embedding_duration = (
            min_embedding_duration
        )

        # Za real-time buffer ne želimo da jedan vrlo kratak
        # pyannote artefakt napravi novog globalnog govornika.
        self.min_segment_for_embedding = 0.45
        # Prava kratka replika od ~1 s mora moći da postane speaker,
        # ali pyannote trzaj od par desetinki sekunde ne sme.
        self.min_total_speech_for_new_speaker = 0.80

        # Broj lokalnih pyannote govornika iz svakog rolling prozora.
        # Koristi se samo kao stabilna procena za FINALNI full-context prolaz.
        self.speaker_count_history: list[int] = []

        # Pragovi su namerno odvojeni:
        # - local_merge: spaja dve pyannote labele iz ISTOG buffera
        #   samo kada su veoma slične
        # - global_match: povezuje lokalnog govornika sa već poznatim
        #   govornikom kroz različite buffere.
        self.local_merge_threshold = 0.80
        # Adaptivno lokalno spajanje koristimo samo kada je jedan par
        # ubedljivo najsličniji i njihove vremenske zone se ne preklapaju.
        self.adaptive_local_merge_threshold = 0.47
        self.adaptive_local_merge_margin = 0.08
        self.max_overlap_for_same_local_speaker = 0.12
        self.global_match_threshold = 0.68
        self.ambiguous_margin = 0.035

        device = (
            "cuda:0"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            "Učitavanje SpeechBrain modela "
            "za buffered speaker matching..."
        )

        print(
            f"SpeechBrain koristi: {device}"
        )

        self.model = (
            SpeakerRecognition.from_hparams(
                source=SPEECHBRAIN_MODEL,
                savedir=(
                    "models/"
                    "spkrec-ecapa-voxceleb"
                ),
                run_opts={
                    "device": device,
                },
                local_strategy=(
                    LocalStrategy.COPY
                ),
            )
        )

        print(
            "SpeechBrain model je učitan."
        )

        # Globalni govornici tokom sastanka.
        #
        # SPEAKER_00
        # SPEAKER_01
        # ...
        self.global_speakers = {}

        # Segmenti prethodnog buffer-a,
        # već prevedeni na globalne govornike.
        #
        # Koristimo ih za povezivanje
        # govornika kroz overlap.
        self.previous_segments = []
        self.speaker_count_history = []

    # =====================================================
    # EMBEDDING
    # =====================================================

    def _normalize_embedding(
        self,
        embedding: torch.Tensor,
    ) -> torch.Tensor:

        embedding = (
            embedding
            .float()
            .reshape(-1)
        )

        norm = torch.linalg.vector_norm(
            embedding
        )

        if norm > 0:
            embedding = (
                embedding / norm
            )

        return embedding.cpu()

    def _compare_embeddings(
        self,
        first: torch.Tensor,
        second: torch.Tensor,
    ) -> float:

        first_np = (
            first.numpy()
            .reshape(1, -1)
        )

        second_np = (
            second.numpy()
            .reshape(1, -1)
        )

        similarity = (
            cosine_similarity(
                first_np,
                second_np,
            )[0][0]
        )

        return float(
            similarity
        )

    def _extract_embedding(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> torch.Tensor:

        waveform = (
            torch.from_numpy(
                audio.astype(
                    np.float32
                )
            )
            .reshape(
                1,
                -1,
            )
        )

        if sample_rate != 16000:
            waveform = (
                torchaudio
                .functional
                .resample(
                    waveform,
                    sample_rate,
                    16000,
                )
            )

        with torch.no_grad():
            embedding = (
                self.model.encode_batch(
                    waveform
                )
            )

        return (
            self._normalize_embedding(
                embedding.squeeze()
            )
        )

    # =====================================================
    # LOKALNI PYANNOTE GOVORNICI
    # =====================================================

    def _build_local_speaker_embeddings(
        self,
        audio: np.ndarray,
        sample_rate: int,
        segments: list[dict],
    ) -> dict:

        """
        Gradi stabilniji embedding po lokalnom pyannote govorniku.

        Stara verzija je potpuno odbacivala svaki turn kraći od 1.5 s.
        U realnom razgovoru je mnogo replika kraće od toga, pa bi lokalna
        labela ostala bez embeddinga i kasnije automatski postala NOVI
        globalni SPEAKER_xx.

        Ovde računamo embedding za sve upotrebljive turnove >= 0.60 s,
        a zatim pravimo duration-weighted centroid. Ako nema nijednog
        dovoljno dugog pojedinačnog turna, ali zbir govora jeste dovoljan,
        spojimo njegove kratke delove i iz njih izvučemo jedan embedding.
        """

        grouped_audio: dict[str, list[tuple[np.ndarray, float]]] = {}
        total_speech: dict[str, float] = {}

        for segment in segments:
            speaker = segment["speaker"]
            duration = max(0.0, segment["end"] - segment["start"])

            total_speech[speaker] = total_speech.get(speaker, 0.0) + duration

            start_sample = max(0, int(segment["start"] * sample_rate))
            end_sample = min(len(audio), int(segment["end"] * sample_rate))

            if end_sample <= start_sample:
                continue

            segment_audio = audio[start_sample:end_sample]

            grouped_audio.setdefault(speaker, []).append(
                (segment_audio, duration)
            )

        local_centroids: dict[str, torch.Tensor] = {}

        for speaker, chunks in grouped_audio.items():
            embeddings: list[torch.Tensor] = []
            weights: list[float] = []

            for segment_audio, duration in chunks:
                if duration < self.min_segment_for_embedding:
                    continue

                embedding = self._extract_embedding(
                    segment_audio,
                    sample_rate,
                )
                embeddings.append(embedding)
                weights.append(duration)

            # Ako imamo samo kratke replike, ali zajedno nose dovoljno
            # govora, spojimo ih. Ovo je mnogo bolje nego da speaker nema
            # embedding i da se odmah kreira novi globalni ID.
            if not embeddings and total_speech.get(speaker, 0.0) >= self.min_total_speech_for_new_speaker:
                usable = [chunk for chunk, _ in chunks if len(chunk) > 0]
                if usable:
                    combined = np.concatenate(usable).astype(np.float32)
                    embeddings.append(
                        self._extract_embedding(combined, sample_rate)
                    )
                    weights.append(float(len(combined)) / sample_rate)

            if not embeddings:
                continue

            stacked = torch.stack(embeddings)
            weight_tensor = torch.tensor(weights, dtype=torch.float32).reshape(-1, 1)
            centroid = (stacked * weight_tensor).sum(dim=0) / weight_tensor.sum()
            local_centroids[speaker] = self._normalize_embedding(centroid)

        return local_centroids

    def _speaker_total_durations(
        self,
        segments: list[dict],
    ) -> dict[str, float]:
        totals: dict[str, float] = {}

        for segment in segments:
            speaker = segment["speaker"]
            totals[speaker] = totals.get(speaker, 0.0) + max(
                0.0,
                segment["end"] - segment["start"],
            )

        return totals

    def _local_labels_temporal_overlap(
        self,
        first: str,
        second: str,
        segments: list[dict],
    ) -> float:
        total = 0.0
        first_segments = [x for x in segments if x["speaker"] == first]
        second_segments = [x for x in segments if x["speaker"] == second]

        for a in first_segments:
            for b in second_segments:
                total += max(
                    0.0,
                    min(a["end"], b["end"]) - max(a["start"], b["start"]),
                )

        return total

    def _merge_duplicate_local_speakers(
        self,
        segments: list[dict],
        local_centroids: dict[str, torch.Tensor],
    ) -> tuple[list[dict], dict[str, torch.Tensor]]:
        """
        KONZERVATIVNI MOD.

        Ne spajamo lokalne pyannote labele unutar ISTOG buffer-a.

        Razlog: na kratkim razgovorima sa 3+ stvarna govornika,
        SpeechBrain similarity između dve različite osobe može biti
        dovoljno visok da agresivni merge napravi veću grešku od
        samog pyannote-a. Bolje je privremeno imati jedan speaker ID
        više nego pogrešno spojiti dve stvarne osobe.

        Stabilizacija identiteta se radi tek kroz naredne buffer-e:
        1) vremenski overlap sa prethodnim buffer-om;
        2) SpeechBrain embedding poređenje sa već poznatim globalnim
           govornicima.
        """

        labels = sorted(local_centroids.keys())

        # I dalje ispisujemo similarity vrednosti samo za dijagnostiku,
        # ali ih NE koristimo za spajanje lokalnih labela.
        for i, first in enumerate(labels):
            for second in labels[i + 1:]:
                similarity = self._compare_embeddings(
                    local_centroids[first],
                    local_centroids[second],
                )
                temporal_overlap = self._local_labels_temporal_overlap(
                    first,
                    second,
                    segments,
                )

                print(
                    f"Local similarity {first} <-> {second}: "
                    f"{similarity:.3f} (time overlap {temporal_overlap:.2f}s)"
                )

        if len(labels) > 1:
            print(
                "Konzervativni mod: ne spajam lokalne pyannote "
                "govornike unutar istog buffer-a."
            )

        return (
            [segment.copy() for segment in segments],
            dict(local_centroids),
        )

    # =====================================================
    # GLOBALNI SPEAKER
    # =====================================================

    def _create_global_speaker(
        self,
        embedding: torch.Tensor,
    ) -> str:

        speaker_id = (
            f"SPEAKER_"
            f"{len(self.global_speakers):02d}"
        )

        self.global_speakers[
            speaker_id
        ] = {
            "centroid": (
                embedding.clone()
            ),
            "embeddings": [
                embedding.clone()
            ],
        }

        print(
            f"Novi globalni govornik: "
            f"{speaker_id}"
        )

        return speaker_id

    def _update_global_speaker(
        self,
        speaker_id: str,
        embedding: torch.Tensor,
    ) -> None:

        data = self.global_speakers[
            speaker_id
        ]

        embeddings = data[
            "embeddings"
        ]

        embeddings.append(
            embedding.clone()
        )

        # Dovoljna nam je ograničena istorija.
        if len(embeddings) > 10:
            embeddings.pop(0)

        centroid = torch.stack(
            embeddings
        ).mean(
            dim=0
        )

        centroid = (
            self._normalize_embedding(
                centroid
            )
        )

        data[
            "centroid"
        ] = centroid

    # =====================================================
    # OVERLAP POVEZIVANJE
    # =====================================================

    def _temporal_overlap(
        self,
        first_start: float,
        first_end: float,
        second_start: float,
        second_end: float,
    ) -> float:

        return max(
            0.0,
            min(
                first_end,
                second_end,
            )
            - max(
                first_start,
                second_start,
            ),
        )

    def _find_overlap_matches(
            self,
            current_segments: list[dict],
            buffer_start: float,
    ) -> dict:

        """
        Povezuje lokalne pyannote govornike
        iz trenutnog buffer-a sa globalnim
        govornicima iz prethodnog buffer-a.

        Koristimo ukupno vremensko preklapanje
        kroz overlap zonu buffer-a.
        """

        if not self.previous_segments:
            return {}

        # -------------------------------------------------
        # 1. SAKUPLJAMO LOKALNE I GLOBALNE GOVORNIKE
        # -------------------------------------------------

        local_labels = sorted(
            {
                segment["speaker"]
                for segment in current_segments
            }
        )

        previous_global_labels = sorted(
            {
                segment["speaker"]
                for segment in self.previous_segments
            }
        )

        # -------------------------------------------------
        # 2. MATRICA OVERLAP SCORE-OVA
        #
        # scores[local][global] =
        # ukupno vremensko preklapanje u sekundama
        # -------------------------------------------------

        scores = {
            local_label: {
                global_label: 0.0
                for global_label
                in previous_global_labels
            }
            for local_label
            in local_labels
        }

        for current in current_segments:

            local_label = current["speaker"]

            current_start = (
                    buffer_start
                    + current["start"]
            )

            current_end = (
                    buffer_start
                    + current["end"]
            )

            for previous in self.previous_segments:

                overlap = self._temporal_overlap(
                    current_start,
                    current_end,
                    previous["start"],
                    previous["end"],
                )

                if overlap <= 0:
                    continue

                previous_global = (
                    previous["speaker"]
                )

                scores[
                    local_label
                ][
                    previous_global
                ] += overlap

        # -------------------------------------------------
        # 3. DEBUG MATRICA
        # -------------------------------------------------

        print()
        print("Overlap matrica:")

        for local_label in local_labels:

            values = []

            for global_label in previous_global_labels:
                score = scores[
                    local_label
                ][
                    global_label
                ]

                values.append(
                    f"{global_label}={score:.2f}s"
                )

            print(
                f"  {local_label}: "
                + ", ".join(values)
            )

        # -------------------------------------------------
        # 4. PRAVIMO SVE MOGUĆE PAROVE
        # -------------------------------------------------

        candidates = []

        for local_label in local_labels:

            for global_label in previous_global_labels:

                score = scores[
                    local_label
                ][
                    global_label
                ]

                if score <= 0:
                    continue

                candidates.append(
                    (
                        score,
                        local_label,
                        global_label,
                    )
                )

        # Najjači overlap prvo.
        candidates.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        # -------------------------------------------------
        # 5. ONE-TO-ONE MATCHING
        # -------------------------------------------------

        mapping = {}

        used_local = set()
        used_global = set()

        for (
                score,
                local_label,
                global_label,
        ) in candidates:

            if local_label in used_local:
                continue

            if global_label in used_global:
                continue

            # Minimalni dokaz da postoji
            # smislen overlap.
            if score < 0.5:
                continue

            mapping[
                local_label
            ] = global_label

            used_local.add(
                local_label
            )

            used_global.add(
                global_label
            )

            print(
                f"Overlap mapping: "
                f"{local_label} -> "
                f"{global_label} "
                f"({score:.2f}s)"
            )

        return mapping
    # =====================================================
    # SPEECHBRAIN FALLBACK
    # =====================================================

    def _global_similarity_score(
        self,
        embedding: torch.Tensor,
        data: dict,
    ) -> float:
        """Poredi i sa centroidom i sa najboljim istorijskim uzorkom."""

        scores = [
            self._compare_embeddings(embedding, data["centroid"])
        ]

        for stored in data.get("embeddings", []):
            scores.append(
                self._compare_embeddings(embedding, stored)
            )

        return max(scores)

    def _match_remaining_by_embedding(
        self,
        local_centroids: dict,
        mapping: dict,
    ) -> dict:

        already_used = set(mapping.values())

        for local_speaker, embedding in local_centroids.items():
            if local_speaker in mapping:
                continue

            candidates: list[tuple[float, str]] = []

            for global_speaker, data in self.global_speakers.items():
                if global_speaker in already_used:
                    continue

                similarity = self._global_similarity_score(embedding, data)
                candidates.append((similarity, global_speaker))

                print(
                    f"Similarity {local_speaker} -> {global_speaker}: "
                    f"{similarity:.3f}"
                )

            candidates.sort(reverse=True)

            best_similarity = candidates[0][0] if candidates else -1.0
            best_global = candidates[0][1] if candidates else None
            second_similarity = candidates[1][0] if len(candidates) > 1 else -1.0

            clear_enough = (
                best_global is not None
                and best_similarity >= self.global_match_threshold
                and (
                    len(candidates) == 1
                    or best_similarity >= 0.75
                    or best_similarity - second_similarity >= self.ambiguous_margin
                )
            )

            if clear_enough:
                mapping[local_speaker] = best_global
                already_used.add(best_global)

                print(
                    f"Embedding mapping: {local_speaker} -> {best_global} "
                    f"({best_similarity:.3f}, second={second_similarity:.3f})"
                )
            else:
                new_speaker = self._create_global_speaker(embedding)
                mapping[local_speaker] = new_speaker
                already_used.add(new_speaker)

        return mapping

    def _map_short_unresolved_labels(
        self,
        segments: list[dict],
        mapping: dict,
    ) -> dict:
        """
        VAŽNO: više NE lepimo kratku labelu najbližem govorniku po vremenu.

        U testu sa tri stvarne osobe pokazalo se da kratka replika novog
        govornika može da bude odmah pored prethodne replike (gap=0), pa bi
        takvo pravilo napravilo pogrešan speaker ID. Unresolved labela ostaje
        unresolved; kasnije može da postane novi globalni govornik samo ako
        ima dovoljno ukupnog govora za pouzdan embedding.
        """

        unresolved = sorted({
            segment["speaker"]
            for segment in segments
            if segment["speaker"] not in mapping
        })

        totals = self._speaker_total_durations(segments)

        for local_label in unresolved:
            print(
                f"Kratka/neodlucena lokalna labela {local_label}: "
                f"{totals.get(local_label, 0.0):.2f}s - ne lepim je "
                f"automatski drugom govorniku."
            )

        return mapping

    # =====================================================
    # GLAVNA OBRADA JEDNOG BUFFER-A
    # =====================================================

    def process_buffer(
        self,
        audio: np.ndarray,
        buffer_start: float,
        sample_rate: int = 16000,
        final: bool = False,
        commit_start: float | None = None,
        commit_end: float | None = None,
    ) -> list[dict]:

        if audio.size == 0:
            return []

        print()
        print("=" * 60)

        print(
            f"BUFFER "
            f"{buffer_start:.1f}s - "
            f"{buffer_start + len(audio) / sample_rate:.1f}s"
        )

        print("=" * 60)

        # -------------------------------------------------
        # 1. PYANNOTE
        # -------------------------------------------------

        local_segments = (
            diarize_buffer(
                audio,
                sample_rate=(
                    sample_rate
                ),
            )
        )

        if not local_segments:
            print(
                "Pyannote nije pronašao govor."
            )

            return []

        local_labels = sorted(
            {
                segment[
                    "speaker"
                ]
                for segment
                in local_segments
            }
        )

        print()
        print(
            "Lokalni pyannote govornici:",
            local_labels,
        )

        # Jedan prozor može povremeno over/under-clusterovati. Zato ne
        # verujemo jednom broju, već pamtimo istoriju i kasnije uzimamo
        # medijanu svih rolling procena.
        if local_labels:
            self.speaker_count_history.append(len(local_labels))


        print("Lokalni segmenti pre mapiranja:")
        for segment in local_segments:
            print(
                f"  [{segment['start']:.2f}s - {segment['end']:.2f}s] "
                f"{segment['speaker']}"
            )

        # -------------------------------------------------
        # 2. SPEECHBRAIN CENTROIDI
        # -------------------------------------------------

        local_centroids = (
            self
            ._build_local_speaker_embeddings(
                audio,
                sample_rate,
                local_segments,
            )
        )

        # Konzervativno: NE spajamo lokalne pyannote labele unutar
        # istog buffer-a. Stabilizacija se radi tek kroz naredne buffer-e.
        local_segments, local_centroids = (
            self._merge_duplicate_local_speakers(
                local_segments,
                local_centroids,
            )
        )

        local_labels = sorted(
            {segment["speaker"] for segment in local_segments}
        )

        # -------------------------------------------------
        # 3. OVERLAP MAPPING
        # -------------------------------------------------

        mapping = (
            self._find_overlap_matches(
                local_segments,
                buffer_start,
            )
        )

        # -------------------------------------------------
        # 4. EMBEDDING FALLBACK
        # -------------------------------------------------

        mapping = (
            self
            ._match_remaining_by_embedding(
                local_centroids,
                mapping,
            )
        )

        # Labelu bez embeddinga ne lepimo automatski najbližem govorniku.
        # To može biti prava kratka replika nove osobe.
        mapping = self._map_short_unresolved_labels(
            local_segments,
            mapping,
        )

        totals = self._speaker_total_durations(local_segments)

        # Tek ako labela nosi dovoljno stvarnog govora dozvoljavamo da
        # napravi nov globalni speaker. Time sprečavamo eksploziju ID-jeva.
        for local_label in local_labels:
            if local_label in mapping:
                continue

            total_speech = totals.get(local_label, 0.0)

            if total_speech < self.min_total_speech_for_new_speaker:
                print(
                    f"Ignorišem nepouzdanu kratku speaker labelu "
                    f"{local_label} ({total_speech:.2f}s)."
                )
                continue

            speaker_segments = [
                item for item in local_segments
                if item["speaker"] == local_label
            ]
            audio_parts = []
            for item in speaker_segments:
                start_sample = max(0, int(item["start"] * sample_rate))
                end_sample = min(len(audio), int(item["end"] * sample_rate))
                if end_sample > start_sample:
                    audio_parts.append(audio[start_sample:end_sample])

            if not audio_parts:
                continue

            speaker_audio = np.concatenate(audio_parts).astype(np.float32)
            embedding = self._extract_embedding(speaker_audio, sample_rate)
            global_speaker = self._create_global_speaker(embedding)
            mapping[local_label] = global_speaker

        # -------------------------------------------------
        # 5. GLOBALNI SEGMENTI
        # -------------------------------------------------

        global_segments = []

        for segment in local_segments:

            local_speaker = (
                segment["speaker"]
            )

            global_speaker = (
                mapping.get(
                    local_speaker
                )
            )

            if global_speaker is None:
                continue

            global_segments.append(
                {
                    "start": (
                        buffer_start
                        + segment["start"]
                    ),
                    "end": (
                        buffer_start
                        + segment["end"]
                    ),
                    "speaker": (
                        global_speaker
                    ),
                    "local_speaker": (
                        local_speaker
                    ),
                }
            )

        # -------------------------------------------------
        # 6. AŽURIRANJE GLOBALNIH PROFILA
        # -------------------------------------------------

        for (
            local_speaker,
            global_speaker,
        ) in mapping.items():

            if (
                local_speaker
                not in local_centroids
            ):
                continue

            self._update_global_speaker(
                global_speaker,
                local_centroids[
                    local_speaker
                ],
            )

        # VAŽNO: za overlap mapping narednog buffer-a čuvamo CEO
        # trenutni rezultat, uključujući poslednjih 10 s. Međutim,
        # korisniku / transkripciji ne vraćamo taj overlap odmah.
        #
        # Primer:
        #   buffer 0-30  -> vraćamo samo 0-20
        #   buffer 20-50 -> vraćamo samo 20-40
        #
        # Poslednjih 10 s svakog punog buffer-a ostaje "pending" i
        # biće potvrđeno sledećim prozorom, gde pyannote ima bolji
        # kontekst. Tako nema duplih niti konfliktnih intervala.
        self.previous_segments = [
            item.copy()
            for item
            in global_segments
        ]

        buffer_duration = len(audio) / sample_rate
        buffer_end = buffer_start + buffer_duration

        # Rolling real-time prozor može biti 30 s, 50 s ili do 60 s.
        # Zato caller eksplicitno kaže koji deo prozora sme da postane
        # finalan. Ostatak ostaje samo kao kontekst za speaker clustering.
        if commit_start is None:
            commit_start = buffer_start

        if commit_end is None:
            is_partial_final = (
                buffer_duration < self.buffer_seconds - 0.05
            )
            if final or is_partial_final:
                commit_end = buffer_end
            else:
                commit_end = min(
                    buffer_start + self.step_seconds,
                    buffer_end,
                )

        commit_start = max(buffer_start, float(commit_start))
        commit_end = min(buffer_end, float(commit_end))

        committed_segments = []

        for item in global_segments:
            clipped_start = max(
                item["start"],
                commit_start,
            )
            clipped_end = min(
                item["end"],
                commit_end,
            )

            if clipped_end <= clipped_start:
                continue

            committed = item.copy()
            committed["start"] = clipped_start
            committed["end"] = clipped_end
            committed_segments.append(committed)

        print()
        print(
            "Finalizovan deo buffer-a: "
            f"{commit_start:.2f}s - {commit_end:.2f}s"
        )

        print()
        print("Globalni speaker mapping:")

        for (
            local_speaker,
            global_speaker,
        ) in mapping.items():

            print(
                f"  {local_speaker} "
                f"-> {global_speaker}"
            )

        return committed_segments

    # =====================================================
    # RESET
    # =====================================================

    def estimate_speaker_count(self) -> int | None:
        """Robusna procena broja govornika iz rolling prozora.

        U ovom pipeline-u pyannote cesce napravi jedan prolazni klaster
        VISKA nego sto izgubi pravog govornika. Ako se rolling procene
        razlikuju samo za 1 (npr. [4, 3, 4, 4]), biramo manji broj.
        To sprecava da jedan lazni split postane finalni govornik.

        Ako je raspon veci od 1, koristimo mode, a kod izjednacenja
        opet biramo manju vrednost.
        """
        if not self.speaker_count_history:
            return None

        from collections import Counter

        history = [int(x) for x in self.speaker_count_history if int(x) > 0]
        if not history:
            return None

        lo = min(history)
        hi = max(history)

        # Najcesci realni problem je +1 lazni speaker u pojedinim prozorima.
        if hi - lo <= 1:
            estimate = lo
        else:
            counts = Counter(history)
            max_frequency = max(counts.values())
            candidates = [
                speaker_count
                for speaker_count, frequency in counts.items()
                if frequency == max_frequency
            ]
            estimate = min(candidates)

        return max(1, min(8, int(estimate)))

    def reset(
        self,
    ) -> None:

        self.global_speakers.clear()

        self.previous_segments = []