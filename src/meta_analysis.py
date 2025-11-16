# src/meta_analysis.py (продолжение)
"""
Мета-анализ: финальная суммаризация и ключевые тезисы
"""
import json
from pathlib import Path
from typing import List, Dict
from collections import Counter
import re


class MetaAnalyzer:
    """Анализ всего видео и извлечение ключевых инсайтов"""

    def __init__(self):
        pass

    def extract_key_sentences(
            self,
            segments: List[Dict],
            num_sentences: int = 10
    ) -> List[Dict]:
        """
        Извлечение ключевых предложений из всех сегментов
        Используем TF-IDF подход
        """
        print(f"[INFO] Extracting {num_sentences} key sentences")

        # Собираем все предложения
        all_sentences = []
        for seg in segments:
            sentences = seg["text"].split('.')
            for sent in sentences:
                sent = sent.strip()
                if len(sent) > 30:  # Минимальная длина предложения
                    all_sentences.append({
                        "text": sent,
                        "segment_id": seg["id"],
                        "timestamp": seg["start"]
                    })

        # Простой scoring: длина + позиция в сегменте
        scored_sentences = []
        for i, sent_data in enumerate(all_sentences):
            score = len(sent_data["text"].split())  # Длина предложения
            score *= (1 / (i + 1) ** 0.1)  # Небольшой бонус за раннее появление

            scored_sentences.append({
                **sent_data,
                "score": score
            })

        # Сортируем по score
        top_sentences = sorted(
            scored_sentences,
            key=lambda x: x["score"],
            reverse=True
        )[:num_sentences]

        # Сортируем по timestamp для хронологии
        top_sentences = sorted(top_sentences, key=lambda x: x["timestamp"])

        return top_sentences

    def extract_topics(self, text: str, num_topics: int = 5) -> List[str]:
        """
        Извлечение основных тем через частотный анализ
        """
        # Стоп-слова для русского языка
        stop_words = {
            'это', 'как', 'так', 'и', 'в', 'на', 'с', 'для', 'по', 'от',
            'что', 'все', 'был', 'быть', 'к', 'а', 'то', 'за', 'из', 'или',
            'у', 'о', 'же', 'не', 'мы', 'вы', 'они', 'он', 'она', 'оно'
        }

        # Токенизация и очистка
        words = re.findall(r'\b[а-яё]{4,}\b', text.lower())
        words = [w for w in words if w not in stop_words]

        # Подсчёт частоты
        word_counts = Counter(words)
        topics = [word for word, _ in word_counts.most_common(num_topics)]

        return topics

    def create_structured_summary(
            self,
            summaries_data: Dict
    ) -> Dict:
        """
        Создание структурированной финальной суммаризации
        """
        print("[INFO] Creating structured summary")

        meta_summary = summaries_data["meta_summary"]
        key_points = summaries_data["key_points"]
        segments = summaries_data["segments"]

        # Извлекаем ключевые предложения
        key_sentences = self.extract_key_sentences(segments)

        # Анализ тематики (простой подсчёт частых слов)
        all_text = " ".join([seg["summary"] for seg in segments])
        topics = self.extract_topics(all_text)

        # Статистика
        total_duration = sum(seg["end"] - seg["start"] for seg in segments)
        total_words = sum(len(seg["text"].split()) for seg in segments)

        structured = {
            "overview": meta_summary,
            "key_points": key_points,
            "key_sentences": [
                {
                    "text": s["text"],
                    "timestamp": self._format_time(s["timestamp"]),
                    "segment_id": s["segment_id"]
                }
                for s in key_sentences
            ],
            "main_topics": topics,
            "statistics": {
                "total_duration_seconds": total_duration,
                "total_words": total_words,
                "num_segments": len(segments),
                "avg_segment_duration": total_duration / len(segments),
                "words_per_minute": (total_words / total_duration) * 60 if total_duration > 0 else 0
            }
        }

        return structured

    def generate_timeline(self, segments: List[Dict]) -> List[Dict]:
        """
        Создание временной шкалы с ключевыми моментами
        """
        print("[INFO] Generating timeline")

        timeline = []
        for seg in segments:
            timeline.append({
                "timestamp": self._format_time(seg["start"]),
                "timestamp_seconds": seg["start"],
                "title": seg["summary"][:100] + "..." if len(seg["summary"]) > 100 else seg["summary"],
                "duration": seg["end"] - seg["start"]
            })

        return timeline

    def process_summaries_file(
            self,
            summaries_path: Path,
            output_dir: Path = None
    ) -> Dict:
        """
        Полный процесс мета-анализа
        """
        print(f"\n{'=' * 60}")
        print("[INFO] Starting meta-analysis")
        print(f"{'=' * 60}\n")

        # Загрузка суммаризаций
        with open(summaries_path, 'r', encoding='utf-8') as f:
            summaries_data = json.load(f)

        # Создание структурированной суммаризации
        structured_summary = self.create_structured_summary(summaries_data)

        # Создание временной шкалы
        timeline = self.generate_timeline(summaries_data["segments"])
        structured_summary["timeline"] = timeline

        # Сохранение
        if output_dir is None:
            output_dir = summaries_path.parent

        self.save_analysis(structured_summary, output_dir)

        return structured_summary

    def save_analysis(self, analysis: Dict, output_dir: Path):
        """Сохранение результатов мета-анализа"""

        # JSON формат
        json_path = output_dir / "final_summary.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)
        print(f"[✓] Saved: {json_path}")

        # TXT формат (финальная суммаризация)
        txt_path = output_dir / "final_summary.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("ФИНАЛЬНАЯ СУММАРИЗАЦИЯ ВИДЕО\n")
            f.write("=" * 70 + "\n\n")

            # Обзор
            f.write("ОБЗОР:\n")
            f.write("-" * 70 + "\n")
            f.write(f"{analysis['overview']}\n\n")

            # Статистика
            stats = analysis['statistics']
            f.write("СТАТИСТИКА:\n")
            f.write("-" * 70 + "\n")
            f.write(f"Длительность: {self._format_time(stats['total_duration_seconds'])}\n")
            f.write(f"Количество слов: {stats['total_words']}\n")
            f.write(f"Сегментов: {stats['num_segments']}\n")
            f.write(f"Темп речи: {stats['words_per_minute']:.0f} слов/мин\n\n")

            # Основные темы
            f.write("ОСНОВНЫЕ ТЕМЫ:\n")
            f.write("-" * 70 + "\n")
            for i, topic in enumerate(analysis['main_topics'], 1):
                f.write(f"{i}. {topic}\n")
            f.write("\n")

            # Ключевые тезисы
            f.write("КЛЮЧЕВЫЕ ТЕЗИСЫ:\n")
            f.write("-" * 70 + "\n")
            for i, point in enumerate(analysis['key_points'], 1):
                f.write(f"{i}. {point}\n")
            f.write("\n")

            # Ключевые цитаты
            f.write("КЛЮЧЕВЫЕ ЦИТАТЫ:\n")
            f.write("-" * 70 + "\n")
            for i, sent in enumerate(analysis['key_sentences'], 1):
                f.write(f"{i}. [{sent['timestamp']}] {sent['text']}\n\n")

            # Временная шкала
            f.write("\n" + "=" * 70 + "\n")
            f.write("ВРЕМЕННАЯ ШКАЛА:\n")
            f.write("=" * 70 + "\n\n")
            for item in analysis['timeline']:
                f.write(f"[{item['timestamp']}] {item['title']}\n")
                f.write(f"  Длительность: {item['duration']:.0f}с\n\n")

        print(f"[✓] Saved: {txt_path}")

        # Key points отдельно
        kp_path = output_dir / "key_points.json"
        key_points_data = {
            "key_points": analysis['key_points'],
            "key_sentences": analysis['key_sentences'],
            "main_topics": analysis['main_topics']
        }
        with open(kp_path, 'w', encoding='utf-8') as f:
            json.dump(key_points_data, f, ensure_ascii=False, indent=2)
        print(f"[✓] Saved: {kp_path}")

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

    parser = argparse.ArgumentParser(description="Meta-analysis of video summaries")
    parser.add_argument("summaries", help="Path to summaries_per_segment.json")

    args = parser.parse_args()

    summaries_path = Path(args.summaries)
    output_dir = summaries_path.parent

    # Создание анализатора
    analyzer = MetaAnalyzer()

    # Мета-анализ
    analysis = analyzer.process_summaries_file(summaries_path, output_dir)

    # Обновление checkpoint
    checkpoint_path = output_dir / "checkpoint.json"
    if checkpoint_path.exists():
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            checkpoint = json.load(f)

        checkpoint["stage"] = "meta_analysis_complete"
        checkpoint["files"]["final_summary_json"] = str(output_dir / "final_summary.json")
        checkpoint["files"]["final_summary_txt"] = str(output_dir / "final_summary.txt")
        checkpoint["files"]["key_points"] = str(output_dir / "key_points.json")

        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)

    print(f"\n[SUCCESS] Meta-analysis complete!")
    print(f"[INFO] Overview: {analysis['overview'][:100]}...")


if __name__ == "__main__":
    main()