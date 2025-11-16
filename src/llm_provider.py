# src/llm_provider.py
"""
LLM Provider для качественной обработки образовательного контента
Поддержка: GigaChat (Sberbank)
Phase 3: Кэширование и генерация ответов к вопросам
"""
import os
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class LLMConfig:
    """Конфигурация LLM"""
    provider: str = "gigachat"  # gigachat, openai, claude
    model: str = "GigaChat"  # GigaChat, GigaChat-Pro, GigaChat-Plus
    api_key: Optional[str] = None
    temperature: float = 0.3  # Низкая для консистентности
    max_tokens: int = 2000
    verify_ssl: bool = False  # Для GigaChat часто нужно False
    scope: str = "GIGACHAT_API_PERS"  # Для физлиц
    use_cache: bool = True  # Phase 3: Включить кэширование


class GigaChatProvider:
    """Провайдер для работы с GigaChat API"""

    def __init__(self, config: LLMConfig):
        """
        Инициализация GigaChat провайдера

        Args:
            config: конфигурация LLM
        """
        self.config = config

        # Импорт библиотеки GigaChat
        try:
            from gigachat import GigaChat
            self.GigaChat = GigaChat
        except ImportError:
            raise ImportError(
                "GigaChat library not installed. "
                "Install with: pip install gigachat"
            )

        # Получение API ключа
        self.api_key = config.api_key or os.getenv("GIGACHAT_CREDENTIALS")
        if not self.api_key:
            raise ValueError(
                "GigaChat API key not found. "
                "Set GIGACHAT_CREDENTIALS environment variable or pass api_key"
            )

        print(f"[INFO] Initializing GigaChat provider")
        print(f"[INFO] Model: {config.model}")
        print(f"[INFO] Scope: {config.scope}")

    def chat(
            self,
            prompt: str,
            system_prompt: Optional[str] = None,
            temperature: Optional[float] = None
    ) -> str:
        """
        Отправка запроса к GigaChat

        Args:
            prompt: пользовательский промпт
            system_prompt: системный промпт (опционально)
            temperature: температура генерации (опционально)

        Returns:
            Ответ модели
        """
        temp = temperature if temperature is not None else self.config.temperature

        try:
            with self.GigaChat(
                    credentials=self.api_key,
                    verify_ssl_certs=self.config.verify_ssl,
                    scope=self.config.scope,
                    model=self.config.model
            ) as giga:
                # Формирование сообщений
                messages = []

                if system_prompt:
                    messages.append({
                        "role": "system",
                        "content": system_prompt
                    })

                messages.append({
                    "role": "user",
                    "content": prompt
                })

                # Запрос к API
                response = giga.chat(
                    messages=messages,
                    temperature=temp,
                    max_tokens=self.config.max_tokens
                )

                return response.choices[0].message.content

        except Exception as e:
            print(f"[ERROR] GigaChat request failed: {e}")
            raise


class LLMProvider:
    """Универсальный провайдер для работы с разными LLM"""

    def __init__(self, config: LLMConfig = None):
        """
        Инициализация провайдера

        Args:
            config: конфигурация LLM (если None, используются значения по умолчанию)
        """
        self.config = config or LLMConfig()

        # Выбор провайдера
        if self.config.provider == "gigachat":
            self.provider = GigaChatProvider(self.config)
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")

        # Phase 3: Инициализация кэша
        if self.config.use_cache:
            try:
                from .llm_cache import LLMCache
                self.cache = LLMCache(enabled=True)
                print("[INFO] LLM caching enabled")
            except Exception as e:
                print(f"[WARN] Failed to initialize cache: {e}")
                self.cache = None
        else:
            self.cache = None

    def _chat_with_cache(
            self,
            prompt: str,
            system_prompt: Optional[str] = None,
            temperature: Optional[float] = None
    ) -> str:
        """
        Отправка запроса с кэшированием (Phase 3)

        Args:
            prompt: пользовательский промпт
            system_prompt: системный промпт
            temperature: температура

        Returns:
            Ответ модели (из кэша или новый)
        """
        # Формируем конфигурацию для кэша
        cache_config = {
            "model": self.config.model,
            "temperature": temperature or self.config.temperature,
            "system_prompt": system_prompt or ""
        }

        # Проверяем кэш
        if self.cache:
            cached_response = self.cache.get(prompt, cache_config)
            if cached_response:
                return cached_response

        # Вызываем API
        response = self.provider.chat(prompt, system_prompt, temperature)

        # Сохраняем в кэш
        if self.cache:
            # Примерная оценка токенов (грубая)
            tokens_estimate = len(prompt.split()) + len(response.split())
            self.cache.set(prompt, cache_config, response, tokens=tokens_estimate)

        return response

    def generate_overview(self, summaries: List[str], metadata: Dict = None) -> str:
        """
        Генерация общего обзора лекции

        Args:
            summaries: список суммаризаций сегментов
            metadata: метаданные (длительность, количество сегментов и т.д.)

        Returns:
            Общий обзор лекции
        """
        print("[INFO] Generating lecture overview with LLM...")

        # Объединяем суммаризации
        combined_text = "\n\n".join([
            f"Сегмент {i + 1}: {summary}"
            for i, summary in enumerate(summaries)
        ])

        # Системный промпт
        system_prompt = """Ты — экспертный ассистент для анализа образовательного контента.
Твоя задача — создать краткий, но информативный обзор всей лекции на основе суммаризаций её частей.

Требования к обзору:
- Объём: 3-5 предложений
- Структура: О чём лекция, какие основные темы рассматриваются, какая цель
- Стиль: академический, ясный, без воды
- Язык: русский
- НЕ используй фразы типа "в этой лекции", "лектор рассказывает" - пиши прямо о содержании"""

        # Пользовательский промпт
        user_prompt = f"""На основе следующих суммаризаций сегментов лекции создай общий обзор:

{combined_text}

Обзор лекции:"""

        # Генерация с кэшированием
        overview = self._chat_with_cache(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.3
        )

        return overview.strip()

    def extract_key_points(self, summaries: List[str], num_points: int = 5) -> List[str]:
        """
        Извлечение ключевых тезисов

        Args:
            summaries: список суммаризаций сегментов
            num_points: количество тезисов

        Returns:
            Список ключевых тезисов
        """
        print(f"[INFO] Extracting {num_points} key points with LLM...")

        # Объединяем суммаризации
        combined_text = "\n\n".join([
            f"Часть {i + 1}: {summary}"
            for i, summary in enumerate(summaries)
        ])

        # Системный промпт
        system_prompt = """Ты — эксперт по анализу образовательного контента.
Твоя задача — выделить наиболее важные тезисы из лекции.

Требования к тезисам:
- Каждый тезис — это законченная мысль (1-2 предложения)
- Тезисы должны быть независимыми и неповторяющимися
- Тезисы отражают СУТЬ, а не детали
- Формулировки чёткие и конкретные
- Без вводных слов типа "лектор говорит о том, что..."

Формат ответа:
1. [Первый тезис]
2. [Второй тезис]
...и так далее"""

        # Пользовательский промпт
        user_prompt = f"""Из следующего содержания лекции выдели {num_points} самых важных тезисов:

{combined_text}

Ключевые тезисы:"""

        # Генерация с кэшированием
        response = self._chat_with_cache(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.3
        )

        # Парсинг тезисов
        key_points = []
        for line in response.strip().split('\n'):
            line = line.strip()
            # Удаляем нумерацию (1., 2., - и т.д.)
            if line and (line[0].isdigit() or line.startswith('-')):
                # Убираем номер и точку/тире
                point = line.lstrip('0123456789.-) ').strip()
                if point:
                    key_points.append(point)

        return key_points[:num_points]

    def generate_questions(
            self,
            summaries: List[str],
            num_questions: int = 10,
            difficulty_mix: bool = True,
            with_answers: bool = True  # Phase 3: Генерация ответов
    ) -> List[Dict[str, str]]:
        """
        Генерация вопросов для самопроверки

        Args:
            summaries: список суммаризаций сегментов
            num_questions: количество вопросов
            difficulty_mix: генерировать вопросы разной сложности (easy, medium, hard)
            with_answers: генерировать ответы к вопросам (Phase 3)

        Returns:
            Список вопросов с метаданными (и ответами, если with_answers=True)
        """
        print(f"[INFO] Generating {num_questions} questions with LLM...")

        # Объединяем суммаризации
        combined_text = "\n\n".join([
            f"Раздел {i + 1}: {summary}"
            for i, summary in enumerate(summaries)
        ])

        # Системный промпт
        if with_answers:
            system_prompt = """Ты — эксперт по созданию образовательных материалов.
Твоя задача — создать вопросы для самопроверки с правильными ответами и объяснениями.

Используй таксономию Блума для разных уровней сложности:
- EASY (базовый уровень): вопросы на запоминание, понимание фактов
- MEDIUM (применение): вопросы на анализ, применение концепций
- HARD (синтез): вопросы на оценку, синтез, критическое мышление

Требования к вопросам:
- Вопросы должны быть конкретными и однозначными
- Избегай вопросов типа "да/нет"
- Вопросы должны проверять понимание, а не просто факты
- Ответы должны быть краткими и точными
- Объяснения должны помогать понять суть

Формат ответа (строго соблюдай):
[EASY] Вопрос?
ОТВЕТ: Краткий правильный ответ.
ОБЪЯСНЕНИЕ: Пояснение, почему это правильный ответ.

[MEDIUM] Вопрос?
ОТВЕТ: Краткий правильный ответ.
ОБЪЯСНЕНИЕ: Пояснение.

...и так далее"""
        else:
            system_prompt = """Ты — эксперт по созданию образовательных материалов.
Твоя задача — создать вопросы для самопроверки по содержанию лекции.

Используй таксономию Блума для разных уровней сложности:
- EASY (базовый уровень): вопросы на запоминание, понимание фактов
- MEDIUM (применение): вопросы на анализ, применение концепций
- HARD (синтез): вопросы на оценку, синтез, критическое мышление

Формат ответа:
[EASY] Вопрос?
[MEDIUM] Вопрос?
[HARD] Вопрос?"""

        # Пользовательский промпт
        difficulty_instruction = ""
        if difficulty_mix:
            easy_count = num_questions // 3
            medium_count = num_questions // 3
            hard_count = num_questions - easy_count - medium_count
            difficulty_instruction = f"""
Создай {num_questions} вопросов в таком соотношении:
- {easy_count} вопросов уровня EASY
- {medium_count} вопросов уровня MEDIUM
- {hard_count} вопросов уровня HARD"""
        else:
            difficulty_instruction = f"Создай {num_questions} вопросов разного уровня сложности."

        user_prompt = f"""На основе следующего содержания лекции создай вопросы для самопроверки:

{combined_text}

{difficulty_instruction}

Вопросы:"""

        # Генерация с кэшированием
        response = self._chat_with_cache(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.5  # Чуть выше для разнообразия
        )

        # Парсинг вопросов
        questions = []
        current_question = None
        current_answer = None
        current_explanation = None

        for line in response.strip().split('\n'):
            line = line.strip()
            if not line:
                continue

            # Поиск уровня сложности
            if any(tag in line.upper() for tag in ["[EASY]", "[MEDIUM]", "[HARD]"]):
                # Сохраняем предыдущий вопрос, если есть
                if current_question:
                    questions.append({
                        "question": current_question["text"],
                        "difficulty": current_question["difficulty"],
                        "answer": current_answer,
                        "explanation": current_explanation
                    })

                # Начинаем новый вопрос
                difficulty = "medium"
                if "[EASY]" in line.upper():
                    difficulty = "easy"
                elif "[HARD]" in line.upper():
                    difficulty = "hard"

                question_text = line.split(']', 1)[1].strip()
                question_text = question_text.lstrip('0123456789.-) ').strip()

                current_question = {"text": question_text, "difficulty": difficulty}
                current_answer = None
                current_explanation = None

            elif line.upper().startswith("ОТВЕТ:"):
                current_answer = line.split(':', 1)[1].strip()

            elif line.upper().startswith("ОБЪЯСНЕНИЕ:"):
                current_explanation = line.split(':', 1)[1].strip()

        # Добавляем последний вопрос
        if current_question:
            questions.append({
                "question": current_question["text"],
                "difficulty": current_question["difficulty"],
                "answer": current_answer,
                "explanation": current_explanation
            })

        return questions[:num_questions]


def main():
    """Пример использования"""
    # Конфигурация
    config = LLMConfig(
        provider="gigachat",
        model="GigaChat",
        api_key=os.getenv("GIGACHAT_CREDENTIALS"),
        temperature=0.3,
        use_cache=True  # Phase 3: Включить кэш
    )

    # Инициализация провайдера
    llm = LLMProvider(config)

    # Пример суммаризаций
    example_summaries = [
        "В начале лекции рассматриваются основные принципы объектно-ориентированного программирования.",
        "Далее объясняется концепция инкапсуляции и её применение на практике.",
        "В заключение обсуждаются паттерны проектирования и их использование."
    ]

    # Генерация overview
    print("\n=== OVERVIEW ===")
    overview = llm.generate_overview(example_summaries)
    print(overview)

    # Извлечение ключевых тезисов
    print("\n=== KEY POINTS ===")
    key_points = llm.extract_key_points(example_summaries, num_points=3)
    for i, point in enumerate(key_points, 1):
        print(f"{i}. {point}")

    # Генерация вопросов с ответами (Phase 3)
    print("\n=== QUESTIONS WITH ANSWERS ===")
    questions = llm.generate_questions(example_summaries, num_questions=3, with_answers=True)
    for q in questions:
        print(f"\n[{q['difficulty'].upper()}] {q['question']}")
        if q.get('answer'):
            print(f"  Ответ: {q['answer']}")
        if q.get('explanation'):
            print(f"  Объяснение: {q['explanation']}")

    # Статистика кэша
    if llm.cache:
        stats = llm.cache.get_stats()
        print(f"\n=== CACHE STATS ===")
        print(f"Entries: {stats['total_entries']}")
        print(f"Estimated tokens saved: {stats['estimated_tokens_saved']}")


if __name__ == "__main__":
    main()
