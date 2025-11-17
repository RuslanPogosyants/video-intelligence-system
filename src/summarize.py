# src/summarize.py (продолжение)
"""
Суммаризация сегментов транскрипции
"""
import json
import torch
from pathlib import Path
from typing import List, Dict
from transformers import T5ForConditionalGeneration, T5Tokenizer
from tqdm import tqdm


class SegmentSummarizer:
    """Суммаризация текстовых сегментов"""

    def __init__(
            self,
            model_name: str = "cointegrated/rut5-base-absum",
            device: str = "auto",
            max_input_length: int = 600,
            max_output_length: int = 150,
            cache_dir: str = "models/summarization"
    ):
        """
        Args:
            model_name: название модели (rut5-base-absum, FRED-T5-large, etc.)
            device: cuda, cpu, auto
            max_input_length: максимальная длина входа в токенах
            max_output_length: максимальная длина суммаризации
            cache_dir: директория для кэширования моделей
        """
        print(f"[INFO] Loading summarization model: {model_name}")

        # Определение устройства
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[INFO] Using device: {self.device}")

        # Загрузка модели и токенизатора
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir
            )
            self.model = T5ForConditionalGeneration.from_pretrained(
                model_name,
                cache_dir=cache_dir
            ).to(self.device)

            print(f"[✓] Model loaded successfully")

            # Проверка VRAM
            if self.device == "cuda":
                memory_allocated = torch.cuda.memory_allocated(0) / 1024 ** 3
                print(f"[INFO] VRAM allocated: {memory_allocated:.2f} GB")

        except Exception as e:
            print(f"[✗] Failed to load model: {e}")
            raise

        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

        # Перевод модели в режим eval
        self.model.eval()

    @staticmethod
    def preprocess_text(text: str) -> str:
        """
        Предобработка текста: очистка от мусора в разговорной речи

        Args:
            text: исходный текст из транскрипции

        Returns:
            Очищенный текст
        """
        import re

        # 1. Удаление звуков-заполнителей (filler sounds)
        filler_patterns = [
            r'\b(?:ммм|эээ|ааа|эмм|хмм|угу|ага)\b',  # Основные звуки
            r'\b(?:ну|вот|как бы|типа|короче|значит)\b',  # Слова-паразиты (частые)
            r'\b(?:понимаете|знаете|смотрите|слушайте)\b',  # Обращения без смысла
        ]

        for pattern in filler_patterns:
            text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)

        # 2. Удаление повторяющихся слов (это это это -> это)
        text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)

        # 3. Удаление множественных знаков препинания (...... -> .)
        text = re.sub(r'\.{2,}', '.', text)
        text = re.sub(r',{2,}', ',', text)
        text = re.sub(r'\?{2,}', '?', text)
        text = re.sub(r'!{2,}', '!', text)

        # 4. Удаление неполных предложений (заканчивающихся на "...", но оставляем если это середина)
        text = re.sub(r'\.{3,}\s*$', '.', text)

        # 5. Очистка множественных пробелов
        text = re.sub(r'\s+', ' ', text)

        # 6. Удаление пробелов перед знаками препинания
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)

        # 7. Добавление пробела после знаков препинания (если его нет)
        text = re.sub(r'([.,!?;:])(\w)', r'\1 \2', text)

        # 8. Удаление пробелов в начале и конце
        text = text.strip()

        return text

    def summarize_text(
            self,
            text: str,
            num_beams: int = 4,
            length_penalty: float = 1.0,
            no_repeat_ngram_size: int = 3
    ) -> str:
        """
        Суммаризация одного текста

        Args:
            text: входной текст
            num_beams: количество лучей для beam search
            length_penalty: штраф за длину (>1 - длиннее, <1 - короче)
            no_repeat_ngram_size: предотвращение повторов n-грамм

        Returns:
            Суммаризированный текст
        """
        # Токенизация
        inputs = self.tokenizer(
            text,
            max_length=self.max_input_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        ).to(self.device)

        # Генерация
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=self.max_output_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                early_stopping=True,
                do_sample=False
            )

        # Декодирование
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary.strip()

    def summarize_segments(
            self,
            segments: List[Dict],
            min_text_length: int = 100,
            show_progress: bool = True
    ) -> List[Dict]:
        """
        Суммаризация списка сегментов

        Args:
            segments: список сегментов с полем 'text'
            min_text_length: минимальная длина текста для суммаризации
            show_progress: показывать прогресс-бар

        Returns:
            Список сегментов с добавленным полем 'summary'
        """
        print(f"[INFO] Summarizing {len(segments)} segments")

        summarized_segments = []
        iterator = tqdm(segments, desc="Summarizing") if show_progress else segments

        for segment in iterator:
            text = segment["text"]
            summary = ""  # ВАЖНО: инициализация по умолчанию

            # Предобработка текста (очистка от мусора)
            try:
                text_cleaned = self.preprocess_text(text)
            except Exception as e:
                print(f"[WARN] Error preprocessing segment {segment.get('id', '?')}: {e}")
                text_cleaned = text  # Fallback на оригинал

            # Пропускаем слишком короткие тексты
            if len(text_cleaned) < min_text_length:
                summary = text_cleaned if text_cleaned else text
                if show_progress:
                    print(f"[WARN] Segment {segment.get('id', '?')} too short ({len(text_cleaned)} chars), skipping summarization")
            else:
                try:
                    summary = self.summarize_text(text_cleaned)
                except Exception as e:
                    print(f"[✗] Error summarizing segment {segment.get('id', '?')}: {e}")
                    # Fallback: первые 200 символов или весь текст
                    summary = text_cleaned[:200] + "..." if len(text_cleaned) > 200 else text_cleaned

            # Добавляем суммаризацию к сегменту
            segment_with_summary = segment.copy()
            segment_with_summary["summary"] = summary
            segment_with_summary["original_length"] = len(text)
            segment_with_summary["summary_length"] = len(summary)
            segment_with_summary["compression_ratio"] = len(text) / len(summary) if summary and len(summary) > 0 else 1.0

            summarized_segments.append(segment_with_summary)

        print(f"[✓] Summarization complete!")

        # Статистика
        total_original = sum(s["original_length"] for s in summarized_segments)
        total_summary = sum(s["summary_length"] for s in summarized_segments)
        avg_compression = total_original / total_summary if total_summary > 0 else 0

        print(f"[INFO] Original length: {total_original} chars")
        print(f"[INFO] Summary length: {total_summary} chars")
        print(f"[INFO] Compression ratio: {avg_compression:.2f}x")

        return summarized_segments

    def create_meta_summary(
            self,
            segment_summaries: List[str],
            max_length: int = 300
    ) -> str:
        """
        Создание мета-суммаризации (суммаризация всех суммаризаций)

        Args:
            segment_summaries: список суммаризаций сегментов
            max_length: максимальная длина мета-суммаризации

        Returns:
            Мета-суммаризация
        """
        print(f"[INFO] Creating meta-summary from {len(segment_summaries)} summaries")

        # Объединяем все суммаризации
        combined = " ".join(segment_summaries)

        # Если объединённый текст слишком длинный, разбиваем на части
        max_chunk = self.max_input_length * 4  # Примерно 2400 символов

        if len(combined) <= max_chunk:
            # Суммаризируем напрямую
            meta_summary = self.summarize_text(combined)
        else:
            # Разбиваем на части и суммаризируем каждую
            chunks = []
            words = combined.split()
            current_chunk = []
            current_length = 0

            for word in words:
                current_length += len(word) + 1
                if current_length > max_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [word]
                    current_length = len(word)
                else:
                    current_chunk.append(word)

            if current_chunk:
                chunks.append(" ".join(current_chunk))

            print(f"[INFO] Split into {len(chunks)} chunks for meta-summarization")

            # Суммаризируем каждую часть
            chunk_summaries = []
            for i, chunk in enumerate(chunks):
                print(f"[INFO] Summarizing chunk {i + 1}/{len(chunks)}")
                summary = self.summarize_text(chunk)
                chunk_summaries.append(summary)

            # Финальная суммаризация
            combined_summaries = " ".join(chunk_summaries)
            meta_summary = self.summarize_text(combined_summaries)

        print(f"[✓] Meta-summary created ({len(meta_summary)} chars)")
        return meta_summary

    def extract_key_points(
            self,
            segments: List[Dict],
            num_points: int = 5
    ) -> List[str]:
        """
        Извлечение ключевых тезисов из сегментов

        Args:
            segments: список сегментов с суммаризациями
            num_points: количество ключевых тезисов

        Returns:
            Список ключевых тезисов
        """
        print(f"[INFO] Extracting {num_points} key points")

        # Простая эвристика: берём самые длинные сегменты
        sorted_segments = sorted(
            segments,
            key=lambda s: s.get("original_length", 0),
            reverse=True
        )

        key_points = []
        for seg in sorted_segments[:num_points]:
            # Берём первое предложение суммаризации
            summary = seg.get("summary", "")
            first_sentence = summary.split(".")[0] + "."
            key_points.append(first_sentence)

        return key_points

    def process_segments_file(
            self,
            segments_path: Path,
            output_dir: Path = None
    ) -> Dict:
        """
        Полный процесс суммаризации из файла сегментов

        Args:
            segments_path: путь к segments_semantic.json
            output_dir: директория для сохранения (по умолчанию та же)

        Returns:
            Dict с результатами суммаризации
        """
        print(f"\n{'=' * 60}")
        print("[INFO] Starting summarization process")
        print(f"{'=' * 60}\n")

        # Загрузка сегментов
        with open(segments_path, 'r', encoding='utf-8') as f:
            segments_data = json.load(f)

        segments = segments_data["segments"]

        # Суммаризация сегментов
        summarized_segments = self.summarize_segments(segments)

        # Создание мета-суммаризации
        segment_summaries = [s["summary"] for s in summarized_segments]
        meta_summary = self.create_meta_summary(segment_summaries)

        # Извлечение ключевых тезисов
        key_points = self.extract_key_points(summarized_segments)

        # Формирование результата
        result = {
            "num_segments": len(summarized_segments),
            "meta_summary": meta_summary,
            "key_points": key_points,
            "segments": summarized_segments
        }

        # Сохранение
        if output_dir is None:
            output_dir = segments_path.parent

        self.save_summaries(result, output_dir)

        return result

    def save_summaries(self, summaries: Dict, output_dir: Path):
        """Сохранение суммаризаций"""
        # JSON формат
        json_path = output_dir / "summaries_per_segment.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summaries, f, ensure_ascii=False, indent=2)
        print(f"[✓] Saved: {json_path}")

        # Читаемый формат
        txt_path = output_dir / "summaries_readable.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("СУММАРИЗАЦИЯ ВИДЕО\n")
            f.write(f"{'=' * 60}\n\n")

            # Мета-суммаризация
            f.write("ОБЩАЯ СУММАРИЗАЦИЯ:\n")
            f.write(f"{'-' * 60}\n")
            f.write(f"{summaries['meta_summary']}\n\n")

            # Ключевые тезисы
            f.write("КЛЮЧЕВЫЕ ТЕЗИСЫ:\n")
            f.write(f"{'-' * 60}\n")
            for i, point in enumerate(summaries['key_points'], 1):
                f.write(f"{i}. {point}\n")
            f.write(f"\n{'=' * 60}\n\n")

            # Суммаризации по сегментам
            f.write("СУММАРИЗАЦИЯ ПО СЕГМЕНТАМ:\n")
            f.write(f"{'=' * 60}\n\n")

            for seg in summaries["segments"]:
                f.write(f"Сегмент {seg['id'] + 1}\n")
                f.write(f"Время: {self._format_time(seg['start'])} - {self._format_time(seg['end'])}\n")
                f.write(f"Compression: {seg['compression_ratio']:.1f}x\n")
                f.write(f"{'-' * 60}\n")
                f.write(f"СУММАРИЗАЦИЯ:\n{seg['summary']}\n\n")
                f.write(f"ОРИГИНАЛЬНЫЙ ТЕКСТ:\n{seg['text'][:300]}...\n")
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

    parser = argparse.ArgumentParser(description="Summarize transcript segments")
    parser.add_argument("segments", help="Path to segments_semantic.json")
    parser.add_argument("--model", default="cointegrated/rut5-base-absum",
                        help="Model name")
    parser.add_argument("--device", default="auto", help="Device (cuda/cpu/auto)")
    parser.add_argument("--max-input", type=int, default=600, help="Max input length")
    parser.add_argument("--max-output", type=int, default=150, help="Max output length")

    args = parser.parse_args()

    segments_path = Path(args.segments)
    output_dir = segments_path.parent

    # Создание суммаризатора
    summarizer = SegmentSummarizer(
        model_name=args.model,
        device=args.device,
        max_input_length=args.max_input,
        max_output_length=args.max_output
    )

    # Суммаризация
    summaries = summarizer.process_segments_file(segments_path, output_dir)

    # Обновление checkpoint
    checkpoint_path = output_dir / "checkpoint.json"
    if checkpoint_path.exists():
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            checkpoint = json.load(f)

        checkpoint["stage"] = "summarization_complete"
        checkpoint["files"]["summaries_json"] = str(output_dir / "summaries_per_segment.json")
        checkpoint["files"]["summaries_txt"] = str(output_dir / "summaries_readable.txt")

        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)

    print(f"\n[SUCCESS] Summarization complete!")
    print(f"[INFO] Meta-summary: {summaries['meta_summary'][:100]}...")


if __name__ == "__main__":
    main()