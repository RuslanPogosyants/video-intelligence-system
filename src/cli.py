# src/cli.py
"""
Единый CLI-интерфейс для всей системы
"""
import click
import json
from pathlib import Path
import sys

# Импорт всех модулей
from src.transcribe import VideoTranscriber
from src.segment import TranscriptSegmenter
from src.summarize import SegmentSummarizer
from src.meta_analysis import MetaAnalyzer
from src.extract_terms import TermExtractor
from src.generate_questions import QuestionGenerator
from src.search_articles import ArticleSearcher
from src.export import ReportExporter


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """
    Система интеллектуального анализа образовательных видео

    Используйте команды для выполнения отдельных этапов или 'process-all' для полного пайплайна.
    """
    pass


@cli.command()
@click.argument('video', type=str)
@click.option('--model', default='base', help='Модель Whisper (tiny/base/small/medium/large)')
@click.option('--language', default='ru', help='Язык видео')
@click.option('--device', default='auto', help='Устройство (cuda/cpu/auto)')
@click.option('--output-dir', default='artifacts', help='Директория для результатов')
def transcribe(video, model, language, device, output_dir):
    """Транскрибация видео"""
    click.echo(f"🎬 Транскрибация видео: {video}")

    transcriber = VideoTranscriber(
        model_size=model,
        device=device,
        output_dir=output_dir
    )

    output_path = transcriber.process_video(video, language)
    click.echo(f"✅ Готово! Результаты: {output_path}")


@cli.command()
@click.argument('transcript', type=click.Path(exists=True))
@click.option('--method', default='similarity', type=click.Choice(['similarity', 'clustering']))
@click.option('--threshold', default=0.7, type=float, help='Порог схожести')
@click.option('--device', default='auto', help='Устройство')
def segment(transcript, method, threshold, device):
    """Сегментация транскрипции"""
    click.echo(f"📊 Сегментация: {transcript}")

    transcript_path = Path(transcript)
    segmenter = TranscriptSegmenter(device=device)

    segments = segmenter.create_semantic_segments(
        transcript_path,
        method=method,
        threshold=threshold
    )

    segmenter.save_segments(segments, transcript_path.parent)
    click.echo(f"✅ Создано {len(segments['segments'])} сегментов")


@cli.command()
@click.argument('segments', type=click.Path(exists=True))
@click.option('--model', default='cointegrated/rut5-base-absum', help='Модель суммаризации')
@click.option('--device', default='auto', help='Устройство')
def summarize(segments, model, device):
    """Суммаризация сегментов"""
    click.echo(f"📝 Суммаризация: {segments}")

    segments_path = Path(segments)
    summarizer = SegmentSummarizer(
        model_name=model,
        device=device
    )

    summaries = summarizer.process_segments_file(segments_path)
    click.echo(f"✅ Создано {len(summaries['segments'])} суммаризаций")


@cli.command()
@click.argument('summaries', type=click.Path(exists=True))
@click.option('--use-llm', is_flag=True, help='Использовать LLM (GigaChat) для overview и key points (Phase 2)')
@click.option('--use-keybert', is_flag=True, help='Использовать KeyBERT для извлечения ключевых слов (Phase 2)')
def meta_analyze(summaries, use_llm, use_keybert):
    """Мета-анализ и ключевые тезисы"""
    click.echo(f"🔍 Мета-анализ: {summaries}")

    if use_llm:
        click.echo("   🤖 Использую LLM (GigaChat) для качественного анализа")
    if use_keybert:
        click.echo("   🔑 Использую KeyBERT для извлечения ключевых слов")

    summaries_path = Path(summaries)
    analyzer = MetaAnalyzer(use_llm=use_llm, use_keybert=use_keybert)

    analysis = analyzer.process_summaries_file(summaries_path)
    click.echo(f"✅ Мета-анализ завершён")


@cli.command()
@click.argument('transcript', type=click.Path(exists=True))
@click.option('--model', default='ru_core_news_lg', help='SpaCy модель')
def extract_terms(transcript, model):
    """Извлечение терминов и сущностей"""
    click.echo(f"📚 Извлечение терминов: {transcript}")

    transcript_path = Path(transcript)
    extractor = TermExtractor(model_name=model)

    results = extractor.process_transcript(transcript_path)
    click.echo(f"✅ Извлечено {results['statistics']['total_terms']} терминов")


@cli.command()
@click.argument('summaries', type=click.Path(exists=True))
@click.option('--num-questions', default=20, type=int, help='Количество вопросов')
@click.option('--use-model', is_flag=True, help='Использовать T5 модель')
@click.option('--use-llm', is_flag=True, help='Использовать LLM (GigaChat) для качественной генерации (Phase 2)')
def generate_questions(summaries, num_questions, use_model, use_llm):
    """Генерация вопросов"""
    click.echo(f"❓ Генерация вопросов: {summaries}")

    if use_llm:
        click.echo("   🤖 Использую LLM (GigaChat) для качественных вопросов")

    summaries_path = Path(summaries)
    generator = QuestionGenerator(use_model=use_model, use_llm=use_llm)

    questions = generator.process_summaries_file(summaries_path, num_questions)
    click.echo(f"✅ Создано {questions['total_questions']} вопросов")


@cli.command()
@click.argument('terms', type=click.Path(exists=True))
@click.option('--enable-scraping', is_flag=True, help='Включить веб-скрейпинг')
@click.option('--max-articles', default=10, type=int, help='Максимум статей')
def search_articles(terms, enable_scraping, max_articles):
    """Поиск релевантных статей"""
    click.echo(f"🔎 Поиск статей: {terms}")

    if not enable_scraping:
        click.echo("⚠️  Веб-скрейпинг отключен (используйте --enable-scraping)")

    terms_path = Path(terms)
    searcher = ArticleSearcher(
        enable_scraping=enable_scraping,
        max_articles=max_articles
    )

    articles = searcher.process_terms_file(terms_path)
    click.echo(f"✅ Найдено {articles['total_articles']} статей")


@cli.command()
@click.argument('artifacts_dir', type=click.Path(exists=True))
@click.option('--no-pdf', is_flag=True, help='Не генерировать PDF')
def export_report(artifacts_dir, no_pdf):
    """Экспорт отчёта (HTML/PDF)"""
    click.echo(f"📄 Экспорт отчёта: {artifacts_dir}")

    artifacts_path = Path(artifacts_dir)
    exporter = ReportExporter()

    results = exporter.export_all(artifacts_path)

    click.echo(f"✅ HTML: {results.get('html', 'N/A')}")
    if 'pdf' in results and not no_pdf:
        click.echo(f"✅ PDF: {results['pdf']}")


@cli.command()
@click.argument('video', type=str)
@click.option('--model', default='base', help='Модель Whisper')
@click.option('--language', default='ru', help='Язык')
@click.option('--device', default='auto', help='Устройство')
@click.option('--output-dir', default='artifacts', help='Директория результатов')
@click.option('--enable-scraping', is_flag=True, help='Включить поиск статей')
@click.option('--skip-questions', is_flag=True, help='Пропустить генерацию вопросов')
@click.option('--skip-articles', is_flag=True, help='Пропустить поиск статей')
@click.option('--use-llm', is_flag=True, help='Использовать LLM (GigaChat) для качественного анализа (Phase 2)')
@click.option('--use-keybert', is_flag=True, help='Использовать KeyBERT для ключевых слов (Phase 2)')
def process_all(video, model, language, device, output_dir, enable_scraping, skip_questions, skip_articles, use_llm, use_keybert):
    """
    ПОЛНЫЙ ПАЙПЛАЙН: транскрибация → сегментация → суммаризация →
    анализ → термины → вопросы → статьи → экспорт
    """
    click.echo("🚀 Запуск полного пайплайна обработки видео\n")
    click.echo("=" * 60)

    try:
        # Этап 1: Транскрибация
        click.echo("\n[1/8] 🎬 Транскрибация...")
        transcriber = VideoTranscriber(model_size=model, device=device, output_dir=output_dir)
        output_path = transcriber.process_video(video, language)
        transcript_path = output_path / "transcript_raw.json"

        # Этап 2: Сегментация
        click.echo("\n[2/8] 📊 Сегментация...")
        segmenter = TranscriptSegmenter(device=device)
        segments = segmenter.create_semantic_segments(transcript_path)
        segmenter.save_segments(segments, output_path)
        segments_path = output_path / "segments_semantic.json"

        # Этап 3: Суммаризация
        click.echo("\n[3/8] 📝 Суммаризация...")
        summarizer = SegmentSummarizer(device=device)
        summaries = summarizer.process_segments_file(segments_path)
        summaries_path = output_path / "summaries_per_segment.json"

        # Этап 4: Мета-анализ
        click.echo("\n[4/8] 🔍 Мета-анализ...")
        if use_llm:
            click.echo("   🤖 Использую LLM (GigaChat) для качественного анализа")
        if use_keybert:
            click.echo("   🔑 Использую KeyBERT для ключевых слов")
        analyzer = MetaAnalyzer(use_llm=use_llm, use_keybert=use_keybert)
        analysis = analyzer.process_summaries_file(summaries_path)

        # Этап 5: Извлечение терминов
        click.echo("\n[5/8] 📚 Извлечение терминов...")
        extractor = TermExtractor()
        terms = extractor.process_transcript(transcript_path)
        terms_path = output_path / "terms_and_entities.json"

        # Этап 6: Генерация вопросов (опционально)
        if not skip_questions:
            click.echo("\n[6/8] ❓ Генерация вопросов...")
            if use_llm:
                click.echo("   🤖 Использую LLM (GigaChat) для качественных вопросов")
            generator = QuestionGenerator(use_model=False, use_llm=use_llm)
            questions = generator.process_summaries_file(summaries_path)
        else:
            click.echo("\n[6/8] ⏭️  Пропускаем генерацию вопросов")

        # Этап 7: Поиск статей (опционально)
        if not skip_articles:
            click.echo("\n[7/8] 🔎 Поиск статей...")
            searcher = ArticleSearcher(enable_scraping=enable_scraping)
            articles = searcher.process_terms_file(terms_path)
        else:
            click.echo("\n[7/8] ⏭️  Пропускаем поиск статей")

        # Этап 8: Экспорт
        click.echo("\n[8/8] 📄 Экспорт отчёта...")
        exporter = ReportExporter()
        results = exporter.export_all(output_path)

        # Финал
        click.echo("\n" + "=" * 60)
        click.echo("✅ ПАЙПЛАЙН ЗАВЕРШЁН УСПЕШНО!")
        click.echo("=" * 60)
        click.echo(f"\n📁 Все результаты сохранены в: {output_path}")
        click.echo(f"🌐 HTML отчёт: {results.get('html', 'N/A')}")
        if 'pdf' in results:
            click.echo(f"📄 PDF отчёт: {results['pdf']}")

        # Краткая статистика
        with open(output_path / "final_summary.json", 'r', encoding='utf-8') as f:
            final = json.load(f)

        stats = final.get('statistics', {})
        click.echo(f"\n📊 Статистика:")
        click.echo(f"   • Длительность: {stats.get('total_duration_seconds', 0) / 60:.1f} минут")
        click.echo(f"   • Сегментов: {stats.get('num_segments', 0)}")
        click.echo(f"   • Слов: {stats.get('total_words', 0)}")

        return 0

    except Exception as e:
        click.echo(f"\n❌ ОШИБКА: {e}", err=True)
        import traceback
        traceback.print_exc()
        return 1


@cli.command()
@click.argument('artifacts_dir', type=click.Path(exists=True))
def status(artifacts_dir):
    """Проверить статус обработки"""
    artifacts_path = Path(artifacts_dir)
    checkpoint_path = artifacts_path / "checkpoint.json"

    if not checkpoint_path.exists():
        click.echo("❌ Checkpoint не найден")
        return

    with open(checkpoint_path, 'r', encoding='utf-8') as f:
        checkpoint = json.load(f)

    click.echo(f"📍 Статус: {checkpoint.get('stage', 'unknown')}")
    click.echo(f"📁 Директория: {checkpoint.get('output_path', 'N/A')}")

    files = checkpoint.get('files', {})
    click.echo("\n📄 Созданные файлы:")
    for name, path in files.items():
        exists = "✅" if Path(path).exists() else "❌"
        click.echo(f"   {exists} {name}: {Path(path).name}")


if __name__ == '__main__':
    sys.exit(cli())