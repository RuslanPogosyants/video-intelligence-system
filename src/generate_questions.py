# src/generate_questions.py
"""
Генерация вопросов по содержимому видео
Phase 2: Интеграция LLM для качественной генерации вопросов
"""
import json
import torch
from pathlib import Path
from typing import List, Dict, Optional
from transformers import T5ForConditionalGeneration, T5Tokenizer
import random


class QuestionGenerator:
    """Генерация вопросов для самопроверки"""

    def __init__(
            self,
            model_name: str = "cointegrated/rut5-base-absum",
            device: str = "auto",
            use_model: bool = True,
            use_llm: bool = False,
            llm_provider: Optional['LLMProvider'] = None
    ):
        """
        Args:
            model_name: модель для генерации (можно использовать ту же, что для суммаризации)
            device: cuda, cpu, auto
            use_model: использовать T5 модель или rule-based подход
            use_llm: использовать LLM (GigaChat) для качественной генерации
            llm_provider: провайдер LLM (если None, создаётся автоматически)
        """
        self.use_model = use_model
        self.use_llm = use_llm

        # Инициализация LLM (Phase 2)
        if use_llm and llm_provider is None:
            try:
                from .llm_provider import LLMProvider, LLMConfig
                print("[INFO] Initializing LLM provider for question generation")
                config = LLMConfig()
                self.llm = LLMProvider(config)
            except Exception as e:
                print(f"[WARN] Failed to initialize LLM: {e}")
                print("[WARN] Falling back to T5/rule-based approach")
                self.use_llm = False
                self.llm = None
        else:
            self.llm = llm_provider

        # Инициализация T5 (fallback)
        if use_model and not use_llm:
            print(f"[INFO] Loading question generation model: {model_name}")

            if device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = device

            print(f"[INFO] Using device: {self.device}")

            try:
                self.tokenizer = T5Tokenizer.from_pretrained(
                    model_name,
                    cache_dir="models/summarization"
                )
                self.model = T5ForConditionalGeneration.from_pretrained(
                    model_name,
                    cache_dir="models/summarization"
                ).to(self.device)
                self.model.eval()
                print(f"[✓] Model loaded successfully")
            except Exception as e:
                print(f"[WARN] Failed to load model: {e}")
                print(f"[INFO] Falling back to rule-based approach")
                self.use_model = False
        else:
            if not use_llm:
                print("[INFO] Using rule-based question generation")

    def generate_question_from_text(
            self,
            text: str,
            question_type: str = "what"
    ) -> str:
        """
        Генерация вопроса из текста с помощью модели
        """
        if not self.use_model:
            return self._generate_rule_based_question(text, question_type)

        # Промпт для генерации вопроса
        prompt = f"Создай вопрос по тексту: {text}"

        inputs = self.tokenizer(
            prompt,
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=100,
                num_beams=3,
                temperature=0.8,
                do_sample=True,
                top_k=50
            )

        question = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return question.strip()

    def _generate_rule_based_question(
            self,
            text: str,
            question_type: str = "what"
    ) -> str:
        """
        Простая генерация вопросов по правилам
        """
        # Берём первое предложение
        sentences = text.split('.')
        if not sentences:
            return "Что обсуждается в данном фрагменте?"

        first_sentence = sentences[0].strip()

        # Шаблоны вопросов
        templates = {
            "what": [
                f"Что говорится о {self._extract_topic(first_sentence)}?",
                f"Какие аспекты рассматриваются в связи с {self._extract_topic(first_sentence)}?",
                "Что является основной темой данного фрагмента?"
            ],
            "why": [
                f"Почему важно понимать {self._extract_topic(first_sentence)}?",
                "Какие причины приводятся для объяснения данного явления?",
                "Почему автор акцентирует внимание на этом аспекте?"
            ],
            "how": [
                f"Как работает {self._extract_topic(first_sentence)}?",
                "Каким образом это связано с основной темой?",
                "Как применить эти знания на практике?"
            ],
            "when": [
                "Когда это происходит?",
                "В каких ситуациях это актуально?",
                "В какой момент это становится важным?"
            ]
        }

        return random.choice(templates.get(question_type, templates["what"]))

    def _extract_topic(self, sentence: str) -> str:
        """Извлечение темы из предложения (упрощённо)"""
        words = sentence.split()
        # Берём существительные из середины предложения
        if len(words) > 3:
            return " ".join(words[2:min(5, len(words))])
        return "этом"

    def generate_questions_from_segments(
            self,
            segments: List[Dict],
            num_questions_per_segment: int = 2,
            difficulty_levels: List[str] = None
    ) -> List[Dict]:
        """
        Генерация вопросов из сегментов

        Args:
            segments: список сегментов с суммаризациями
            num_questions_per_segment: количество вопросов на сегмент
            difficulty_levels: уровни сложности ["easy", "medium", "hard"]

        Returns:
            Список вопросов с метаданными
        """
        if difficulty_levels is None:
            difficulty_levels = ["easy", "medium", "hard"]

        print(f"[INFO] Generating questions from {len(segments)} segments")

        all_questions = []
        question_types = ["what", "why", "how", "when"]

        for seg in segments:
            # Используем суммаризацию для генерации вопросов
            text = seg.get("summary", seg.get("text", ""))

            if len(text) < 50:  # Пропускаем слишком короткие
                continue

            for i in range(num_questions_per_segment):
                # Выбираем тип вопроса и сложность
                q_type = random.choice(question_types)
                difficulty = difficulty_levels[i % len(difficulty_levels)]

                try:
                    question = self.generate_question_from_text(text, q_type)

                    all_questions.append({
                        "id": len(all_questions),
                        "question": question,
                        "segment_id": seg["id"],
                        "timestamp": seg["start"],
                        "difficulty": difficulty,
                        "type": q_type,
                        "context": text[:200] + "..."
                    })
                except Exception as e:
                    print(f"[WARN] Failed to generate question for segment {seg['id']}: {e}")
                    continue

        print(f"[✓] Generated {len(all_questions)} questions")
        return all_questions

    def generate_key_questions(
            self,
            key_points: List[str],
            num_questions: int = 10
    ) -> List[Dict]:
        """
        Генерация ключевых вопросов на основе тезисов
        """
        print(f"[INFO] Generating {num_questions} key questions from key points")

        questions = []

        for i, point in enumerate(key_points[:num_questions]):
            # Генерируем вопрос по каждому тезису
            question = self._generate_rule_based_question(point, "what")

            questions.append({
                "id": i,
                "question": question,
                "key_point": point,
                "difficulty": "hard",  # Ключевые вопросы — сложные
                "type": "conceptual"
            })

        return questions

    def process_summaries_file(
            self,
            summaries_path: Path,
            num_questions: int = 20,
            output_dir: Path = None
    ) -> Dict:
        """
        Полный процесс генерации вопросов
        Phase 2: Использует LLM для качественной генерации
        """
        print(f"\n{'=' * 60}")
        print("[INFO] Starting question generation")
        print(f"{'=' * 60}\n")

        # Загрузка суммаризаций
        with open(summaries_path, 'r', encoding='utf-8') as f:
            summaries_data = json.load(f)

        segments = summaries_data["segments"]
        key_points = summaries_data.get("key_points", [])

        # PHASE 2: Использование LLM для генерации всех вопросов сразу
        if self.use_llm and self.llm is not None:
            print("[INFO] Using LLM for question generation")
            try:
                # Собираем суммаризации
                summaries = [seg["summary"] for seg in segments]

                # Генерируем вопросы через LLM (качественно!)
                llm_questions = self.llm.generate_questions(
                    summaries,
                    num_questions=num_questions,
                    difficulty_mix=True
                )

                # Форматируем вопросы с таймкодами
                all_questions = []
                for i, q in enumerate(llm_questions):
                    # Пытаемся найти релевантный сегмент
                    relevant_segment = segments[i % len(segments)]

                    all_questions.append({
                        "id": i,
                        "question": q["question"],
                        "difficulty": q["difficulty"],
                        "segment_id": relevant_segment["id"],
                        "timestamp": relevant_segment["start"],
                        "type": "llm_generated"
                    })

            except Exception as e:
                print(f"[WARN] LLM question generation failed: {e}, using fallback")
                # Fallback к T5/rule-based
                segment_questions = self.generate_questions_from_segments(
                    segments,
                    num_questions_per_segment=max(1, num_questions // len(segments))
                )
                key_questions = self.generate_key_questions(
                    key_points,
                    num_questions=min(10, num_questions // 2)
                )
                all_questions = segment_questions + key_questions
                random.shuffle(all_questions)

        else:
            # Оригинальный подход (T5 или rule-based)
            segment_questions = self.generate_questions_from_segments(
                segments,
                num_questions_per_segment=max(1, num_questions // len(segments))
            )

            # Генерация ключевых вопросов
            key_questions = self.generate_key_questions(
                key_points,
                num_questions=min(10, num_questions // 2)
            )

            # Объединение и перемешивание
            all_questions = segment_questions + key_questions
            random.shuffle(all_questions)

        # Переназначение ID после перемешивания
        for i, q in enumerate(all_questions):
            q["id"] = i

        # Статистика
        difficulty_counts = {}
        type_counts = {}
        for q in all_questions:
            diff = q.get("difficulty", "unknown")
            q_type = q.get("type", "unknown")
            difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1
            type_counts[q_type] = type_counts.get(q_type, 0) + 1

        result = {
            "total_questions": len(all_questions),
            "questions": all_questions,
            "statistics": {
                "by_difficulty": difficulty_counts,
                "by_type": type_counts
            }
        }

        print(f"\n[✓] Question generation complete!")
        print(f"[INFO] Total questions: {len(all_questions)}")
        print(f"[INFO] By difficulty: {difficulty_counts}")

        # Сохранение
        if output_dir is None:
            output_dir = summaries_path.parent

        self.save_questions(result, output_dir)

        return result

    def save_questions(self, questions_data: Dict, output_dir: Path):
        """Сохранение вопросов"""

        # JSON формат
        json_path = output_dir / "questions.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(questions_data, f, ensure_ascii=False, indent=2)
        print(f"[✓] Saved: {json_path}")

        # TXT формат (читаемый)
        txt_path = output_dir / "questions_formatted.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("ВОПРОСЫ ДЛЯ САМОПРОВЕРКИ\n")
            f.write("=" * 70 + "\n\n")

            # Статистика
            stats = questions_data["statistics"]
            f.write("СТАТИСТИКА:\n")
            f.write("-" * 70 + "\n")
            f.write(f"Всего вопросов: {questions_data['total_questions']}\n")
            f.write(f"По сложности: {stats['by_difficulty']}\n")
            f.write(f"По типу: {stats['by_type']}\n\n")

            # Вопросы по уровням сложности
            for difficulty in ["easy", "medium", "hard"]:
                difficulty_questions = [
                    q for q in questions_data["questions"]
                    if q.get("difficulty") == difficulty
                ]

                if difficulty_questions:
                    level_name = {
                        "easy": "ЛЁГКИЕ",
                        "medium": "СРЕДНИЕ",
                        "hard": "СЛОЖНЫЕ"
                    }[difficulty]

                    f.write(f"\n{level_name} ВОПРОСЫ:\n")
                    f.write("=" * 70 + "\n\n")

                    for q in difficulty_questions:
                        f.write(f"Вопрос {q['id'] + 1}:\n")
                        f.write(f"{q['question']}\n")

                        if "timestamp" in q:
                            f.write(f"Таймкод: {self._format_time(q['timestamp'])}\n")

                        if "context" in q:
                            f.write(f"Контекст: {q['context']}\n")

                        f.write("\n" + "-" * 70 + "\n\n")

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

    parser = argparse.ArgumentParser(description="Generate questions from video content")
    parser.add_argument("summaries", help="Path to summaries_per_segment.json")
    parser.add_argument("--num-questions", type=int, default=20, help="Number of questions")
    parser.add_argument("--use-model", action="store_true", help="Use T5 model (slower)")
    parser.add_argument("--use-llm", action="store_true", help="Use LLM (GigaChat) for quality (Phase 2)")
    parser.add_argument("--device", default="auto", help="Device (cuda/cpu/auto)")

    args = parser.parse_args()

    summaries_path = Path(args.summaries)
    output_dir = summaries_path.parent

    # Создание генератора
    generator = QuestionGenerator(
        device=args.device,
        use_model=args.use_model,
        use_llm=args.use_llm
    )

    # Генерация вопросов
    questions = generator.process_summaries_file(
        summaries_path,
        num_questions=args.num_questions,
        output_dir=output_dir
    )

    # Обновление checkpoint
    checkpoint_path = output_dir / "checkpoint.json"
    if checkpoint_path.exists():
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            checkpoint = json.load(f)

        checkpoint["stage"] = "question_generation_complete"
        checkpoint["files"]["questions_json"] = str(output_dir / "questions.json")
        checkpoint["files"]["questions_txt"] = str(output_dir / "questions_formatted.txt")

        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)

    print(f"\n[SUCCESS] Question generation complete!")


if __name__ == "__main__":
    main()