# src/segment.py
"""
Семантическая сегментация транскрипции
"""
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
import nltk
from tqdm import tqdm

# Загрузка punkt для сегментации предложений
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class TranscriptSegmenter:
    """Сегментация транскрипции на смысловые блоки"""

    def __init__(
            self,
            model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            device: str = "cuda",
            cache_dir: str = "models/sentence_transformers"
    ):
        """
        Args:
            model_name: название модели для эмбеддингов
            device: cuda или cpu
            cache_dir: директория для кэширования моделей
        """
        print(f"[INFO] Loading sentence transformer: {model_name}")
        self.model = SentenceTransformer(model_name, device=device, cache_folder=cache_dir)
        print(f"[✓] Model loaded on {device}")

        self.device = device

    def split_into_sentences(self, text: str) -> List[str]:
        """Разбивка текста на предложения"""
        sentences = nltk.sent_tokenize(text, language='russian')
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    def compute_embeddings(self, sentences: List[str]) -> np.ndarray:
        """Вычисление эмбеддингов для предложений"""
        print(f"[INFO] Computing embeddings for {len(sentences)} sentences")
        embeddings = self.model.encode(
            sentences,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings

    def segment_by_similarity(
            self,
            sentences: List[str],
            embeddings: np.ndarray,
            threshold: float = 0.7,
            min_segment_sentences: int = 3,
            max_segment_sentences: int = 30
    ) -> List[List[int]]:
        """
        Сегментация по семантической схожести

        Returns:
            List[List[int]]: список сегментов, каждый сегмент - список индексов предложений
        """
        print(f"[INFO] Segmenting by similarity (threshold={threshold})")

        if len(sentences) == 0:
            return []

        segments = []
        current_segment = [0]

        for i in range(1, len(sentences)):
            # Сравниваем текущее предложение с предыдущим
            sim = cosine_similarity(
                embeddings[i].reshape(1, -1),
                embeddings[i - 1].reshape(1, -1)
            )[0][0]

            # Если схожесть высокая - продолжаем сегмент
            if sim >= threshold and len(current_segment) < max_segment_sentences:
                current_segment.append(i)
            else:
                # Начинаем новый сегмент
                if len(current_segment) >= min_segment_sentences:
                    segments.append(current_segment)
                    current_segment = [i]
                else:
                    # Слишком короткий сегмент - присоединяем к текущему
                    current_segment.append(i)

        # Добавляем последний сегмент
        if len(current_segment) >= min_segment_sentences:
            segments.append(current_segment)
        elif segments:
            # Присоединяем к предыдущему сегменту
            segments[-1].extend(current_segment)
        else:
            segments.append(current_segment)

        print(f"[✓] Created {len(segments)} segments")
        return segments

    def segment_by_clustering(
            self,
            sentences: List[str],
            embeddings: np.ndarray,
            num_segments: int = None,
            distance_threshold: float = 1.5
    ) -> List[List[int]]:
        """
        Сегментация через иерархическую кластеризацию
        """
        print(f"[INFO] Segmenting by clustering")

        if num_segments is None:
            # Автоматическое определение числа сегментов
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=distance_threshold,
                linkage='ward'
            )
        else:
            clustering = AgglomerativeClustering(
                n_clusters=num_segments,
                linkage='ward'
            )

        labels = clustering.fit_predict(embeddings)

        # Группировка по кластерам
        segments = {}
        for idx, label in enumerate(labels):
            if label not in segments:
                segments[label] = []
            segments[label].append(idx)

        # Сортировка сегментов по первому предложению
        segments_list = [segments[k] for k in sorted(segments.keys())]

        print(f"[✓] Created {len(segments_list)} segments via clustering")
        return segments_list

    def map_sentences_to_timestamps(
            self,
            sentences: List[str],
            transcript_segments: List[Dict]
    ) -> List[Dict]:
        """
        Сопоставление предложений с временными метками из транскрипции
        """
        print("[INFO] Mapping sentences to timestamps")

        sentence_timestamps = []
        full_transcript = " ".join([seg["text"] for seg in transcript_segments])

        char_to_time = []  # Маппинг позиции символа на время
        current_pos = 0

        for seg in transcript_segments:
            seg_text = seg["text"] + " "
            for _ in seg_text:
                char_to_time.append({
                    "start": seg["start"],
                    "end": seg["end"]
                })
            current_pos += len(seg_text)

        # Для каждого предложения найдём его позицию в полном тексте
        search_pos = 0
        for sent in sentences:
            try:
                pos = full_transcript.index(sent, search_pos)
                search_pos = pos + len(sent)

                # Получаем временные метки
                if pos < len(char_to_time):
                    start_time = char_to_time[pos]["start"]
                    end_pos = min(pos + len(sent), len(char_to_time) - 1)
                    end_time = char_to_time[end_pos]["end"]

                    sentence_timestamps.append({
                        "text": sent,
                        "start": start_time,
                        "end": end_time
                    })
                else:
                    sentence_timestamps.append({
                        "text": sent,
                        "start": 0,
                        "end": 0
                    })
            except ValueError:
                # Предложение не найдено - используем приблизительные метки
                sentence_timestamps.append({
                    "text": sent,
                    "start": 0,
                    "end": 0
                })

        return sentence_timestamps

    def create_semantic_segments(
            self,
            transcript_path: Path,
            method: str = "similarity",
            **kwargs
    ) -> Dict:
        """
        Полный процесс сегментации

        Args:
            transcript_path: путь к transcript_raw.json
            method: "similarity" или "clustering"
            **kwargs: параметры для методов сегментации

        Returns:
            Dict с сегментами
        """
        print(f"\n{'=' * 60}")
        print("[INFO] Starting semantic segmentation")
        print(f"{'=' * 60}\n")

        # Загрузка транскрипции
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript = json.load(f)

        full_text = transcript["full_text"]
        transcript_segments = transcript["segments"]

        # Разбивка на предложения
        sentences = self.split_into_sentences(full_text)
        print(f"[INFO] Split into {len(sentences)} sentences")

        # Вычисление эмбеддингов
        embeddings = self.compute_embeddings(sentences)

        # Сегментация
        if method == "similarity":
            segment_indices = self.segment_by_similarity(
                sentences, embeddings, **kwargs
            )
        elif method == "clustering":
            segment_indices = self.segment_by_clustering(
                sentences, embeddings, **kwargs
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        # Маппинг на временные метки
        sentence_timestamps = self.map_sentences_to_timestamps(
            sentences, transcript_segments
        )

        # Формирование финальных сегментов
        final_segments = []
        for i, indices in enumerate(segment_indices):
            segment_sentences = [sentence_timestamps[idx] for idx in indices]

            segment_text = " ".join([s["text"] for s in segment_sentences])
            start_time = segment_sentences[0]["start"]
            end_time = segment_sentences[-1]["end"]

            final_segments.append({
                "id": i,
                "start": start_time,
                "end": end_time,
                "duration": end_time - start_time,
                "num_sentences": len(segment_sentences),
                "text": segment_text,
                "sentences": segment_sentences
            })

        result = {
            "method": method,
            "num_segments": len(final_segments),
            "total_sentences": len(sentences),
            "segments": final_segments
        }

        print(f"\n[✓] Segmentation complete!")
        print(f"[INFO] Created {len(final_segments)} semantic segments")
        print(f"[INFO] Average segment length: {len(sentences) / len(final_segments):.1f} sentences")

        return result

    def save_segments(self, segments: Dict, output_dir: Path):
        """Сохранение сегментов"""
        # JSON формат
        json_path = output_dir / "segments_semantic.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)
        print(f"[✓] Saved: {json_path}")

        # Читаемый формат
        txt_path = output_dir / "segments_summary.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"Семантическая сегментация\n")
            f.write(f"Метод: {segments['method']}\n")
            f.write(f"Количество сегментов: {segments['num_segments']}\n")
            f.write(f"{'=' * 60}\n\n")

            for seg in segments["segments"]:
                f.write(f"Сегмент {seg['id'] + 1}\n")
                f.write(f"Время: {self._format_time(seg['start'])} - {self._format_time(seg['end'])}\n")
                f.write(f"Длительность: {seg['duration']:.1f}с\n")
                f.write(f"Предложений: {seg['num_sentences']}\n")
                f.write(f"{'-' * 60}\n")
                f.write(f"{seg['text']}\n")
                f.write(f"\n{'=' * 60}\n\n")
        print(f"[✓] Saved: {txt_path}")

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Форматирование времени"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def main():
    """Пример использования"""
    import argparse

    parser = argparse.ArgumentParser(description="Semantic segmentation of transcript")
    parser.add_argument("transcript", help="Path to transcript_raw.json")
    parser.add_argument("--method", default="similarity", choices=["similarity", "clustering"])
    parser.add_argument("--threshold", type=float, default=0.7, help="Similarity threshold")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")

    args = parser.parse_args()

    transcript_path = Path(args.transcript)
    output_dir = transcript_path.parent

    # Создание сегментатора
    segmenter = TranscriptSegmenter(device=args.device)

    # Сегментация
    segments = segmenter.create_semantic_segments(
        transcript_path=transcript_path,
        method=args.method,
        threshold=args.threshold
    )

    # Сохранение
    segmenter.save_segments(segments, output_dir)

    # Обновление checkpoint
    checkpoint_path = output_dir / "checkpoint.json"
    if checkpoint_path.exists():
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            checkpoint = json.load(f)

        checkpoint["stage"] = "segmentation_complete"
        checkpoint["files"]["segments_json"] = str(output_dir / "segments_semantic.json")
        checkpoint["files"]["segments_txt"] = str(output_dir / "segments_summary.txt")

        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)

    print(f"\n[SUCCESS] Segmentation complete!")


if __name__ == "__main__":
    main()