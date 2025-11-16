# src/meta_analysis.py
"""
Мета-анализ: финальная суммаризация и ключевые тезисы
Phase 2: Интеграция LLM и KeyBERT для качественного анализа
"""
import json
from pathlib import Path
from typing import List, Dict, Optional
from collections import Counter
import re


class MetaAnalyzer:
    """Анализ всего видео и извлечение ключевых инсайтов"""

    def __init__(
            self,
            use_llm: bool = False,
            llm_provider: Optional['LLMProvider'] = None,
            use_keybert: bool = False
    ):
        """
        Args:
            use_llm: использовать LLM для overview и key points
            llm_provider: провайдер LLM (если None, создаётся автоматически)
            use_keybert: использовать KeyBERT для извлечения ключевых слов
        """
        self.use_llm = use_llm
        self.use_keybert = use_keybert

        # Инициализация LLM
        if use_llm and llm_provider is None:
            try:
                from .llm_provider import LLMProvider, LLMConfig
                print("[INFO] Initializing LLM provider for meta-analysis")
                config = LLMConfig()
                self.llm = LLMProvider(config)
            except Exception as e:
                print(f"[WARN] Failed to initialize LLM: {e}")
                print("[WARN] Falling back to rule-based approach")
                self.use_llm = False
                self.llm = None
        else:
            self.llm = llm_provider

        # Инициализация KeyBERT
        if use_keybert:
            try:
                from keybert import KeyBERT
                print("[INFO] Initializing KeyBERT for keyword extraction")
                self.keybert = KeyBERT()
            except Exception as e:
                print(f"[WARN] Failed to initialize KeyBERT: {e}")
                print("[WARN] Falling back to TF-IDF approach")
                self.use_keybert = False
                self.keybert = None
        else:
            self.keybert = None

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

    def extract_topics_keybert(self, text: str, num_topics: int = 5) -> List[str]:
        """
        Извлечение основных тем через KeyBERT (Phase 2)
        """
        try:
            # Извлечение ключевых слов
            keywords = self.keybert.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 2),  # Uni/bigrams
                stop_words='russian',
                top_n=num_topics * 2,  # Берём больше для фильтрации
                diversity=0.7  # Разнообразие ключевых слов
            )

            # Фильтруем и возвращаем топ-N
            topics = [keyword for keyword, _ in keywords]
            return topics[:num_topics]

        except Exception as e:
            print(f"[WARN] KeyBERT extraction failed: {e}, falling back to TF-IDF")
            return self.extract_topics_tfidf(text, num_topics)

    def extract_topics_tfidf(self, text: str, num_topics: int = 5) -> List[str]:
        """
        Извлечение основных тем через TF-IDF (fallback)
        """
        # Расширенный список стоп-слов для русского языка
        stop_words = {
            # Местоимения
            'я', 'ты', 'он', 'она', 'оно', 'мы', 'вы', 'они',
            'его', 'её', 'их', 'мой', 'твой', 'свой', 'наш', 'ваш',
            'этот', 'тот', 'такой', 'этого', 'того', 'этом', 'том',
            'кто', 'что', 'какой', 'который', 'чей', 'сколько',

            # Предлоги
            'в', 'на', 'с', 'к', 'по', 'за', 'из', 'от', 'у', 'о', 'об',
            'про', 'для', 'без', 'до', 'при', 'через', 'над', 'под', 'между',

            # Союзы
            'и', 'а', 'но', 'или', 'да', 'что', 'чтобы', 'если', 'когда',
            'как', 'так', 'потому', 'поэтому', 'также', 'тоже', 'либо',

            # Частицы
            'не', 'ни', 'бы', 'ли', 'же', 'ведь', 'вот', 'вон', 'даже',
            'уже', 'еще', 'ещё', 'только', 'лишь', 'просто',

            # Глаголы-связки и вспомогательные
            'быть', 'был', 'была', 'было', 'были', 'есть', 'будет', 'будут',
            'стать', 'стал', 'стала', 'стало', 'стали',

            # Общие слова
            'это', 'все', 'весь', 'всё', 'сам', 'самый', 'самого', 'самое',
            'другой', 'другого', 'другое', 'один', 'одного', 'одно',
            'два', 'три', 'несколько', 'много', 'мало', 'такое', 'такого',

            # Слова-паразиты из разговорной речи
            'ну', 'вот', 'как-то', 'типа', 'короче', 'значит', 'понимаете',
            'знаете', 'смотрите', 'слушайте', 'скажем', 'допустим',
            'собственно', 'фактически', 'практически', 'буквально',

            # Указательные и вопросительные
            'где', 'куда', 'откуда', 'когда', 'почему', 'зачем', 'сколько',
            'здесь', 'там', 'тут', 'туда', 'сюда', 'тогда', 'теперь', 'сейчас',

            # Дополнительные частые слова без смысловой нагрузки
            'может', 'можно', 'нужно', 'надо', 'должен', 'должно', 'должны',
            'хочу', 'хотел', 'могу', 'мог', 'говорить', 'сказать',
            'делать', 'сделать', 'давать', 'дать', 'иметь',

            # Слова из вашего примера
            'какой', 'этого', 'если', 'друг', 'человек', 'любой', 'просто',
            'достаточно', 'какие', 'более', 'менее', 'очень', 'совсем'
        }

        # Токенизация и очистка
        words = re.findall(r'\b[а-яё]{4,}\b', text.lower())
        words = [w for w in words if w not in stop_words]

        # Подсчёт частоты
        word_counts = Counter(words)

        # Фильтруем слова с частотой меньше 2 (слишком редкие)
        filtered_counts = {word: count for word, count in word_counts.items() if count >= 2}

        # Берём топ-N наиболее частых
        topics = [word for word, _ in Counter(filtered_counts).most_common(num_topics * 3)]

        # Дополнительная фильтрация: убираем слова короче 5 букв (они часто бессмысленны)
        topics = [word for word in topics if len(word) >= 5]

        return topics[:num_topics]

    def extract_topics(self, text: str, num_topics: int = 5) -> List[str]:
        """
        Извлечение основных тем (выбирает метод автоматически)
        """
        if self.use_keybert and self.keybert is not None:
            return self.extract_topics_keybert(text, num_topics)
        else:
            return self.extract_topics_tfidf(text, num_topics)

    def create_structured_summary(
            self,
            summaries_data: Dict
    ) -> Dict:
        """
        Создание структурированной финальной суммаризации
        Phase 2: Использует LLM для качественного overview и key points
        """
        print("[INFO] Creating structured summary")

        segments = summaries_data["segments"]

        # Собираем суммаризации сегментов
        segment_summaries = [seg["summary"] for seg in segments]

        # Генерация overview и key points
        if self.use_llm and self.llm is not None:
            print("[INFO] Using LLM for overview and key points extraction")
            try:
                # Используем LLM для качественного анализа
                meta_summary = self.llm.generate_overview(segment_summaries)
                key_points = self.llm.extract_key_points(segment_summaries, num_points=5)
            except Exception as e:
                print(f"[WARN] LLM generation failed: {e}, using fallback")
                # Fallback к старым данным
                meta_summary = summaries_data.get("meta_summary", "")
                key_points = summaries_data.get("key_points", [])
        else:
            # Используем старые данные из T5 суммаризации
            meta_summary = summaries_data.get("meta_summary", "")
            key_points = summaries_data.get("key_points", [])

        # Извлекаем ключевые предложения
        key_sentences = self.extract_key_sentences(segments)

        # Анализ тематики (KeyBERT или TF-IDF)
        all_text = " ".join(segment_summaries)
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