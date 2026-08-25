import os

import numpy as np
import torch
import torchaudio

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

from speechbrain.inference.speaker import (
    SpeakerRecognition,
)
from speechbrain.utils.fetching import (
    LocalStrategy,
)


MODEL_NAME = (
    "speechbrain/spkrec-ecapa-voxceleb"
)


class RealtimeSpeakerMatcher:
    def __init__(self) -> None:
        device = (
            "cuda:0"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            f"Speaker embedding koristi: {device}"
        )

        self.model = (
            SpeakerRecognition.from_hparams(
                source=MODEL_NAME,
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

        # -------------------------------------------------
        # GLOBALNI GOVORNICI
        #
        # Ovo stanje traje tokom celog sastanka.
        #
        # Tako možemo da prepoznamo:
        #
        # A -> B -> A
        #
        # kao:
        #
        # SPEAKER_00 -> SPEAKER_01 -> SPEAKER_00
        # -------------------------------------------------

        self.speakers: dict[
            str,
            dict,
        ] = {}

        # -------------------------------------------------
        # CLUSTERING PARAMETRI
        # -------------------------------------------------

        # Cosine distance:
        #
        # distance = 1 - similarity
        #
        # Npr:
        #
        # similarity 0.80 -> distance 0.20
        # similarity 0.30 -> distance 0.70
        #
        # Na našim testovima isti glas je često bio
        # oko 0.7-0.9, dok su različiti glasovi
        # često padali mnogo niže.
        self.cluster_distance_threshold = 0.42

        # Lokalni cluster mora imati bar ovoliko
        # overlapping prozora da bismo ga smatrali
        # dovoljno stabilnim.
        self.min_cluster_windows = 2

        # Kada lokalni cluster poredimo sa
        # govornicima iz ranijih VAD segmenata,
        # ova sličnost je potrebna da bismo rekli
        # da je to isti već poznati govornik.
        self.global_match_threshold = 0.58

        # Profil poznatog govornika ažuriramo
        # samo kada smo dovoljno sigurni.
        self.global_update_threshold = 0.70

        print(
            "SpeechBrain speaker model "
            "je uspešno učitan."
        )

    # =====================================================
    # EMBEDDING
    # =====================================================

    def normalize_embedding(
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

    def extract_embedding(
        self,
        audio_path: str,
    ) -> torch.Tensor:
        if not os.path.exists(
            audio_path
        ):
            raise FileNotFoundError(
                "Audio fajl nije pronađen: "
                f"{audio_path}"
            )

        waveform, sample_rate = (
            torchaudio.load(
                audio_path
            )
        )

        if waveform.shape[0] > 1:
            waveform = waveform.mean(
                dim=0,
                keepdim=True,
            )

        if sample_rate != 16000:
            waveform = (
                torchaudio.functional.resample(
                    waveform,
                    sample_rate,
                    16000,
                )
            )

        embedding = (
            self.model.encode_batch(
                waveform
            )
        )

        return self.normalize_embedding(
            embedding.squeeze()
        )

    def extract_embedding_from_array(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> torch.Tensor:
        if audio.size == 0:
            raise ValueError(
                "Audio segment je prazan."
            )

        waveform = torch.from_numpy(
            audio.astype(
                np.float32
            )
        ).reshape(
            1,
            -1,
        )

        if sample_rate != 16000:
            waveform = (
                torchaudio.functional.resample(
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

        return self.normalize_embedding(
            embedding.squeeze()
        )

    # =====================================================
    # SIMILARITY
    # =====================================================

    def compare_embeddings(
        self,
        first_embedding: torch.Tensor,
        second_embedding: torch.Tensor,
    ) -> float:
        first = (
            first_embedding
            .numpy()
            .reshape(
                1,
                -1,
            )
        )

        second = (
            second_embedding
            .numpy()
            .reshape(
                1,
                -1,
            )
        )

        similarity = (
            cosine_similarity(
                first,
                second,
            )[0][0]
        )

        return float(
            similarity
        )

    # =====================================================
    # GLOBAL SPEAKER PROFILES
    # =====================================================

    def _create_speaker(
        self,
        embedding: torch.Tensor,
    ) -> str:
        speaker_id = (
            f"SPEAKER_"
            f"{len(self.speakers):02d}"
        )

        self.speakers[
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
            f"Novi govornik: "
            f"{speaker_id}"
        )

        return speaker_id

    def _update_speaker(
        self,
        speaker_id: str,
        embedding: torch.Tensor,
    ) -> None:
        speaker_data = (
            self.speakers[
                speaker_id
            ]
        )

        stored_embeddings = (
            speaker_data[
                "embeddings"
            ]
        )

        stored_embeddings.append(
            embedding.clone()
        )

        # Ne čuvamo beskonačno mnogo
        # embeddinga.
        if (
            len(stored_embeddings)
            > 12
        ):
            stored_embeddings.pop(
                0
            )

        centroid = torch.stack(
            stored_embeddings
        ).mean(
            dim=0
        )

        centroid = (
            self.normalize_embedding(
                centroid
            )
        )

        speaker_data[
            "centroid"
        ] = centroid

    def _find_best_global_speaker(
        self,
        embedding: torch.Tensor,
    ) -> tuple[
        str | None,
        float,
    ]:
        best_speaker = None
        best_similarity = -1.0

        for (
            speaker_id,
            speaker_data,
        ) in self.speakers.items():

            similarities = []

            centroid_similarity = (
                self.compare_embeddings(
                    embedding,
                    speaker_data[
                        "centroid"
                    ],
                )
            )

            similarities.append(
                centroid_similarity
            )

            # Poredimo i sa istorijskim
            # embeddingima.
            for stored_embedding in (
                speaker_data[
                    "embeddings"
                ]
            ):
                similarity = (
                    self.compare_embeddings(
                        embedding,
                        stored_embedding,
                    )
                )

                similarities.append(
                    similarity
                )

            speaker_similarity = max(
                similarities
            )

            if (
                speaker_similarity
                > best_similarity
            ):
                best_similarity = (
                    speaker_similarity
                )

                best_speaker = (
                    speaker_id
                )

        return (
            best_speaker,
            best_similarity,
        )

    def match_speaker(
        self,
        embedding: torch.Tensor,
    ) -> str:
        """
        Jednostavan globalni matching.

        Koristi se kao fallback i za povezivanje
        cluster centroida sa govornicima koji su
        već pronađeni u ranijim VAD segmentima.
        """

        embedding = (
            self.normalize_embedding(
                embedding
            )
        )

        if not self.speakers:
            return self._create_speaker(
                embedding
            )

        (
            best_speaker,
            best_similarity,
        ) = (
            self._find_best_global_speaker(
                embedding
            )
        )

        print(
            "Globalna sličnost: "
            f"{best_similarity:.3f}"
        )

        if (
            best_speaker is not None
            and best_similarity
            >= self.global_match_threshold
        ):
            if (
                best_similarity
                >= self.global_update_threshold
            ):
                self._update_speaker(
                    best_speaker,
                    embedding,
                )

            return best_speaker

        return self._create_speaker(
            embedding
        )

    # =====================================================
    # LOCAL CLUSTERING
    # =====================================================

    def _cluster_embeddings(
        self,
        embeddings: list[
            torch.Tensor
        ],
    ) -> np.ndarray:
        """
        Grupisanje embeddinga unutar jednog
        VAD govornog segmenta.

        Ne moramo unapred da znamo broj
        govornika.
        """

        if not embeddings:
            return np.array(
                [],
                dtype=np.int32,
            )

        if len(embeddings) == 1:
            return np.array(
                [0],
                dtype=np.int32,
            )

        matrix = np.stack(
            [
                embedding.numpy()
                for embedding
                in embeddings
            ]
        )

        # Novije sklearn verzije koriste
        # metric="cosine".
        #
        # Ostavljamo fallback za starije
        # verzije gde se argument zvao
        # affinity.
        try:
            clustering = (
                AgglomerativeClustering(
                    n_clusters=None,
                    metric="cosine",
                    linkage="average",
                    distance_threshold=(
                        self
                        .cluster_distance_threshold
                    ),
                )
            )

        except TypeError:
            clustering = (
                AgglomerativeClustering(
                    n_clusters=None,
                    affinity="cosine",
                    linkage="average",
                    distance_threshold=(
                        self
                        .cluster_distance_threshold
                    ),
                )
            )

        labels = (
            clustering.fit_predict(
                matrix
            )
        )

        return labels.astype(
            np.int32
        )

    def _repair_small_clusters(
        self,
        labels: np.ndarray,
        embeddings: list[
            torch.Tensor
        ],
    ) -> np.ndarray:
        """
        Transition prozori mogu da naprave
        mali zaseban cluster.

        Cluster sa samo jednim prozorom ne
        prihvatamo automatski kao nov glas.

        Pokušavamo da ga pripišemo najbližem
        stabilnijem clusteru.
        """

        if len(labels) < 2:
            return labels

        repaired = labels.copy()

        unique_labels = list(
            np.unique(
                labels
            )
        )

        counts = {
            int(label): int(
                np.sum(
                    labels == label
                )
            )
            for label
            in unique_labels
        }

        stable_labels = [
            label
            for label in unique_labels
            if counts[
                int(label)
            ] >= self.min_cluster_windows
        ]

        # Ako nijedan nije stabilan,
        # nemamo dovoljno informacija za
        # popravljanje.
        if not stable_labels:
            return repaired

        cluster_centroids = {}

        for label in stable_labels:
            indexes = np.where(
                labels == label
            )[0]

            centroid = torch.stack(
                [
                    embeddings[index]
                    for index
                    in indexes
                ]
            ).mean(
                dim=0
            )

            cluster_centroids[
                int(label)
            ] = (
                self.normalize_embedding(
                    centroid
                )
            )

        for label in unique_labels:
            label = int(label)

            if (
                counts[label]
                >= self.min_cluster_windows
            ):
                continue

            indexes = np.where(
                labels == label
            )[0]

            for index in indexes:
                embedding = (
                    embeddings[
                        index
                    ]
                )

                best_label = None
                best_similarity = -1.0

                for (
                    stable_label,
                    centroid,
                ) in (
                    cluster_centroids.items()
                ):
                    similarity = (
                        self.compare_embeddings(
                            embedding,
                            centroid,
                        )
                    )

                    if (
                        similarity
                        > best_similarity
                    ):
                        best_similarity = (
                            similarity
                        )

                        best_label = (
                            stable_label
                        )

                if best_label is not None:
                    repaired[
                        index
                    ] = best_label

        return repaired

    def _build_local_cluster_centroids(
        self,
        labels: np.ndarray,
        embeddings: list[
            torch.Tensor
        ],
    ) -> dict[
        int,
        torch.Tensor,
    ]:
        centroids = {}

        for label in np.unique(
            labels
        ):
            label_int = int(
                label
            )

            indexes = np.where(
                labels == label
            )[0]

            cluster_embeddings = [
                embeddings[index]
                for index in indexes
            ]

            centroid = torch.stack(
                cluster_embeddings
            ).mean(
                dim=0
            )

            centroids[
                label_int
            ] = (
                self.normalize_embedding(
                    centroid
                )
            )

        return centroids

    # =====================================================
    # WINDOW ANALYSIS
    # =====================================================

    def analyze_speech_windows(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        window_seconds: float = 3.0,
        step_seconds: float = 1.0,
    ) -> list[dict]:
        """
        1. Sečemo segment na overlapping prozore.
        2. Vadimo SpeechBrain embedding za svaki.
        3. Clustering radimo NAD SVIM prozorima.
        4. Lokalni cluster povezujemo sa globalnim
           SPEAKER_XX identitetom.
        """

        if audio.size == 0:
            return []

        window_size = int(
            window_seconds
            * sample_rate
        )

        step_size = int(
            step_seconds
            * sample_rate
        )

        window_data = []
        embeddings = []

        # -------------------------------------------------
        # 1. PROZORI + EMBEDDINGS
        # -------------------------------------------------

        for start_sample in range(
            0,
            len(audio),
            step_size,
        ):
            end_sample = min(
                start_sample
                + window_size,
                len(audio),
            )

            window = audio[
                start_sample:
                end_sample
            ]

            duration = (
                len(window)
                / sample_rate
            )

            # Prekratko za pouzdan
            # speaker embedding.
            if duration < 1.5:
                continue

            rms = float(
                np.sqrt(
                    np.mean(
                        np.square(
                            window
                        )
                    )
                )
            )

            if rms < 0.001:
                continue

            embedding = (
                self
                .extract_embedding_from_array(
                    window,
                    sample_rate=(
                        sample_rate
                    ),
                )
            )

            window_data.append(
                {
                    "start": (
                        start_sample
                        / sample_rate
                    ),
                    "end": (
                        end_sample
                        / sample_rate
                    ),
                }
            )

            embeddings.append(
                embedding
            )

        if not embeddings:
            return []

        # -------------------------------------------------
        # 2. CLUSTERING
        # -------------------------------------------------

        local_labels = (
            self._cluster_embeddings(
                embeddings
            )
        )

        print(
            "Lokalni clustering:"
        )

        unique_before = np.unique(
            local_labels
        )

        print(
            "  Broj sirovih klastera: "
            f"{len(unique_before)}"
        )

        # -------------------------------------------------
        # 3. UKLANJANJE MALIH TRANSITION KLASTERA
        # -------------------------------------------------

        local_labels = (
            self._repair_small_clusters(
                local_labels,
                embeddings,
            )
        )

        unique_after = np.unique(
            local_labels
        )

        print(
            "  Broj stabilnih klastera: "
            f"{len(unique_after)}"
        )

        # -------------------------------------------------
        # 4. CENTROID SVAKOG LOKALNOG KLASTERA
        # -------------------------------------------------

        local_centroids = (
            self
            ._build_local_cluster_centroids(
                local_labels,
                embeddings,
            )
        )

        # -------------------------------------------------
        # 5. LOKALNI CLUSTER -> GLOBALNI SPEAKER
        # -------------------------------------------------

        cluster_to_speaker = {}

        for local_label in sorted(
            local_centroids.keys()
        ):
            centroid = (
                local_centroids[
                    local_label
                ]
            )

            speaker_id = (
                self.match_speaker(
                    centroid
                )
            )

            cluster_to_speaker[
                local_label
            ] = speaker_id

            print(
                f"  Cluster "
                f"{local_label} -> "
                f"{speaker_id}"
            )

        # -------------------------------------------------
        # 6. FINALNI WINDOW RESULTS
        # -------------------------------------------------

        results = []

        for (
            item,
            local_label,
        ) in zip(
            window_data,
            local_labels,
        ):
            speaker_id = (
                cluster_to_speaker[
                    int(local_label)
                ]
            )

            results.append(
                {
                    "start": (
                        item["start"]
                    ),
                    "end": (
                        item["end"]
                    ),
                    "speaker": (
                        speaker_id
                    ),
                }
            )

        return results

    # =====================================================
    # GROUPING
    # =====================================================

    def group_speaker_windows(
        self,
        results: list[dict],
    ) -> list[dict]:
        if not results:
            return []

        if len(results) == 1:
            return [
                {
                    "start": (
                        results[0][
                            "start"
                        ]
                    ),
                    "end": (
                        results[0][
                            "end"
                        ]
                    ),
                    "speaker": (
                        results[0][
                            "speaker"
                        ]
                    ),
                }
            ]

        grouped = []

        current_speaker = (
            results[0][
                "speaker"
            ]
        )

        current_start = (
            results[0][
                "start"
            ]
        )

        for index in range(
            1,
            len(results),
        ):
            previous = (
                results[
                    index - 1
                ]
            )

            current = (
                results[
                    index
                ]
            )

            if (
                current["speaker"]
                == current_speaker
            ):
                continue

            previous_center = (
                previous["start"]
                + previous["end"]
            ) / 2.0

            current_center = (
                current["start"]
                + current["end"]
            ) / 2.0

            boundary = (
                previous_center
                + current_center
            ) / 2.0

            grouped.append(
                {
                    "start": (
                        current_start
                    ),
                    "end": (
                        boundary
                    ),
                    "speaker": (
                        current_speaker
                    ),
                }
            )

            current_speaker = (
                current["speaker"]
            )

            current_start = (
                boundary
            )

        grouped.append(
            {
                "start": (
                    current_start
                ),
                "end": (
                    results[-1][
                        "end"
                    ]
                ),
                "speaker": (
                    current_speaker
                ),
            }
        )

        return grouped

    # =====================================================
    # SMOOTHING
    # =====================================================

    def smooth_speaker_groups(
        self,
        groups: list[dict],
        min_duration: float = 3.0,
    ) -> list[dict]:
        if len(groups) < 2:
            return groups

        smoothed = [
            group.copy()
            for group
            in groups
        ]

        changed = True

        while changed:
            changed = False

            for index in range(
                len(smoothed)
            ):
                group = (
                    smoothed[
                        index
                    ]
                )

                duration = (
                    group["end"]
                    - group["start"]
                )

                if (
                    duration
                    >= min_duration
                ):
                    continue

                previous_group = (
                    smoothed[
                        index - 1
                    ]
                    if index > 0
                    else None
                )

                next_group = (
                    smoothed[
                        index + 1
                    ]
                    if (
                        index
                        < len(smoothed) - 1
                    )
                    else None
                )

                # A B A, gde je B vrlo kratak:
                #
                # pripajamo B nazad A.
                if (
                    previous_group
                    is not None
                    and next_group
                    is not None
                    and previous_group[
                        "speaker"
                    ]
                    == next_group[
                        "speaker"
                    ]
                ):
                    previous_group[
                        "end"
                    ] = (
                        next_group[
                            "end"
                        ]
                    )

                    del smoothed[
                        index:
                        index + 2
                    ]

                    changed = True
                    break

                previous_duration = (
                    previous_group[
                        "end"
                    ]
                    - previous_group[
                        "start"
                    ]
                    if previous_group
                    is not None
                    else -1
                )

                next_duration = (
                    next_group[
                        "end"
                    ]
                    - next_group[
                        "start"
                    ]
                    if next_group
                    is not None
                    else -1
                )

                if (
                    previous_group
                    is not None
                    and previous_duration
                    >= next_duration
                ):
                    previous_group[
                        "end"
                    ] = (
                        group[
                            "end"
                        ]
                    )

                    del smoothed[
                        index
                    ]

                    changed = True
                    break

                if (
                    next_group
                    is not None
                ):
                    next_group[
                        "start"
                    ] = (
                        group[
                            "start"
                        ]
                    )

                    del smoothed[
                        index
                    ]

                    changed = True
                    break

        return smoothed

    # =====================================================
    # RESET
    # =====================================================

    def reset(
        self,
    ) -> None:
        self.speakers.clear()


# =========================================================
# STANDALONE TEST
# =========================================================

if __name__ == "__main__":
    matcher = (
        RealtimeSpeakerMatcher()
    )

    audio_path = (
        "audio/"
        "Kineska veštačka inteligencija "
        "plaši SAD "
        "I RTS jutarnji program.wav"
    )

    waveform, sample_rate = (
        torchaudio.load(
            audio_path
        )
    )

    if waveform.shape[0] > 1:
        waveform = waveform.mean(
            dim=0,
            keepdim=True,
        )

    if sample_rate != 16000:
        waveform = (
            torchaudio.functional.resample(
                waveform,
                sample_rate,
                16000,
            )
        )

        sample_rate = 16000

    audio = (
        waveform
        .squeeze()
        .numpy()
        .astype(
            np.float32
        )
    )

    results = (
        matcher.analyze_speech_windows(
            audio,
            sample_rate=(
                sample_rate
            ),
            window_seconds=3.0,
            step_seconds=1.0,
        )
    )

    grouped_results = (
        matcher.group_speaker_windows(
            results
        )
    )

    grouped_results = (
        matcher.smooth_speaker_groups(
            grouped_results,
            min_duration=3.0,
        )
    )

    print()
    print("=" * 60)
    print(
        "GRUPISANI REZULTATI"
    )
    print("=" * 60)

    for item in grouped_results:
        print(
            f"["
            f"{item['start']:.1f}s - "
            f"{item['end']:.1f}s"
            f"] "
            f"{item['speaker']}"
        )

    print()
    print(
        "Ukupan broj globalnih "
        "govornika:",
        len(
            matcher.speakers
        ),
    )