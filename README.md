# 🎬 Video Intelligence System

Комплексная система интеллектуального анализа образовательных видео с использованием AI/ML.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![AI](https://img.shields.io/badge/AI-Powered-purple.svg)

## 🌟 Возможности

- ✅ **Транскрибация** - Whisper (faster-whisper) для преобразования видео/аудио в текст
- ✅ **Семантическая сегментация** - разбивка на смысловые блоки
- ✅ **Суммаризация** - T5 модель для создания краткого содержания
- ✅ **Мета-анализ** - извлечение ключевых тезисов и временной шкалы
- ✅ **Извлечение терминов** - NER и глоссарий
- ✅ **Генерация вопросов** - автоматические вопросы для самопроверки
- ✅ **Поиск статей** - релевантные материалы по темам
- ✅ **Экспорт** - HTML/PDF отчеты
- ✅ **Веб-интерфейс** - красивый Bootstrap UI для просмотра результатов

## 🚀 Быстрый старт

### 1. Установка зависимостей

```bash
# Создать виртуальное окружение
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate  # Windows

# Установить зависимости
pip install -r requirements.txt

# Загрузить SpaCy модель для русского языка
python -m spacy download ru_core_news_lg
```

### 2. Обработка видео

```bash
# Полный пайплайн (рекомендуется)
python -m src.cli process-all your_video.mp4 --language ru

# Или отдельные этапы:
python -m src.cli transcribe your_video.mp4
python -m src.cli segment artifacts/video_TIMESTAMP/transcript_raw.json
python -m src.cli summarize artifacts/video_TIMESTAMP/segments_semantic.json
# ... и т.д.
```

### 3. Запуск веб-интерфейса

```bash
python web_app.py
```

Откройте браузер: **http://localhost:5000**

## 📊 Архитектура системы

```
┌─────────────────────────────────────────────────────────────┐
│                    VIDEO INTELLIGENCE SYSTEM                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │   1. Transcribe   │  ← Whisper (faster-whisper)
                    │   Video → Text    │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │   2. Segment      │  ← Sentence Transformers
                    │   Semantic Split  │     + Clustering
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │   3. Summarize    │  ← T5 (rut5-base-absum)
                    │   Each Segment    │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │  4. Meta-Analysis │  ← Key points extraction
                    │   Final Summary   │     + Timeline
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ 5. Extract Terms  │  ← SpaCy NER
                    │   NER + Glossary  │     + Frequency analysis
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ 6. Gen Questions  │  ← Rule-based + ML
                    │  Self-assessment  │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ 7. Search Articles│  ← Wikipedia API
                    │  Related content  │     + Web scraping
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │   8. Export       │  ← Jinja2 + WeasyPrint
                    │   HTML/PDF Report │
                    └───────────────────┘
```

## 🛠️ Технологический стек

| Категория | Технологии |
|-----------|-----------|
| **ML/DL** | PyTorch, Transformers, faster-whisper |
| **NLP** | SpaCy, NLTK, Sentence Transformers |
| **Web** | Flask, Bootstrap 5, Jinja2 |
| **Audio/Video** | yt-dlp, pydub, ffmpeg |
| **Export** | WeasyPrint, Markdown |
| **Utils** | Click, tqdm, BeautifulSoup |

## 📂 Структура проекта

```
video-intelligence-system/
├── src/
│   ├── cli.py              # CLI интерфейс
│   ├── transcribe.py       # Транскрибация (Whisper)
│   ├── segment.py          # Сегментация
│   ├── summarize.py        # Суммаризация (T5)
│   ├── meta_analysis.py    # Мета-анализ
│   ├── extract_terms.py    # Извлечение терминов (SpaCy)
│   ├── generate_questions.py  # Генерация вопросов
│   ├── search_articles.py  # Поиск статей
│   └── export.py           # Экспорт (HTML/PDF)
├── templates/
│   ├── base.html           # Базовый шаблон
│   ├── index.html          # Список видео
│   └── session.html        # Детальный просмотр
├── config/
│   └── config.yaml         # Конфигурация
├── artifacts/              # Результаты обработки (локально)
├── models/                 # Модели ML (локально)
├── web_app.py              # Flask веб-приложение
└── requirements.txt        # Зависимости
```

## ⚙️ Конфигурация

Настройки в `config/config.yaml`:

```yaml
system:
  device: "cuda"  # cuda, cpu, auto
  fallback_to_cpu: true

transcription:
  model: "base"  # tiny, base, small, medium, large
  language: "ru"

segmentation:
  method: "semantic"
  min_segment_length: 60

summarization:
  model: "cointegrated/rut5-base-absum"
  max_input_length: 600
```

## 📸 Скриншоты

### Главная страница
Список обработанных видео с основной информацией

### Детальный просмотр
- **Обзор** - общая суммаризация и ключевые тезисы
- **Транскрипция** - полный текст с таймкодами
- **Сегменты** - смысловые блоки с суммаризацией
- **Термины** - глоссарий и именованные сущности
- **Вопросы** - автогенерированные вопросы для самопроверки
- **Статьи** - релевантные материалы

## 🎯 Примеры использования

### Обработка YouTube видео

```bash
python -m src.cli process-all https://youtube.com/watch?v=VIDEO_ID --language ru
```

### Обработка локального файла

```bash
python -m src.cli process-all lecture.mp4 --model medium --device cuda
```

### Только транскрибация

```bash
python -m src.cli transcribe lecture.wav --model base --language ru
```

### Проверка статуса обработки

```bash
python -m src.cli status artifacts/video_20241116_120000
```

## 🔧 Требования

- Python 3.8+
- FFmpeg (для обработки аудио/видео)
- CUDA (опционально, для ускорения на GPU)
- 8GB+ RAM (рекомендуется 16GB)
- 10GB+ свободного места (для моделей)

## 📝 Лицензия

MIT License - свободное использование для любых целей

## 🤝 Поддержка

Нашли баг? Есть предложения? Создайте Issue на GitHub!

## 🌟 Roadmap

- [ ] Поддержка других языков
- [ ] Интеграция с API (OpenAI, Anthropic)
- [ ] Batch processing
- [ ] Real-time transcription
- [ ] Docker контейнер
- [ ] Telegram бот
- [ ] Облачное хранилище результатов

---

**Создано с ❤️ и AI**
