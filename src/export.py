# src/export.py
"""
Экспорт результатов в HTML и PDF
"""
import json
from pathlib import Path
from typing import Dict
from jinja2 import Template
import markdown
from datetime import datetime


class ReportExporter:
    """Экспорт результатов анализа"""

    def __init__(self):
        self.html_template = self._create_html_template()

    def _create_html_template(self) -> Template:
        """Создание HTML шаблона"""
        template_str = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding: 20px;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-radius: 8px;
        }

        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 30px;
            font-size: 2.5em;
        }

        h2 {
            color: #34495e;
            margin-top: 40px;
            margin-bottom: 20px;
            font-size: 1.8em;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }

        h3 {
            color: #555;
            margin-top: 25px;
            margin-bottom: 15px;
            font-size: 1.3em;
        }

        .meta-info {
            background: #ecf0f1;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 30px;
        }

        .meta-info p {
            margin: 5px 0;
        }

        .overview {
            background: #e8f4f8;
            padding: 20px;
            border-left: 4px solid #3498db;
            margin: 20px 0;
            font-size: 1.1em;
        }

        .key-points {
            background: #fff9e6;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
        }

        .key-points ul {
            list-style: none;
            padding-left: 0;
        }

        .key-points li {
            padding: 10px;
            margin: 5px 0;
            background: white;
            border-left: 3px solid #f39c12;
            padding-left: 15px;
        }

        .segment {
            margin: 25px 0;
            padding: 20px;
            background: #fafafa;
            border-radius: 5px;
            border: 1px solid #e0e0e0;
        }

        .segment-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }

        .timestamp {
            color: #3498db;
            font-weight: bold;
            font-family: monospace;
        }

        .summary {
            background: white;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
            border-left: 3px solid #27ae60;
        }

        .glossary-term {
            background: #f8f9fa;
            padding: 10px;
            margin: 8px 0;
            border-radius: 4px;
            border-left: 3px solid #9b59b6;
        }

        .question {
            background: #fff5f5;
            padding: 15px;
            margin: 15px 0;
            border-radius: 5px;
            border-left: 3px solid #e74c3c;
        }

        .difficulty {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 3px;
            font-size: 0.85em;
            font-weight: bold;
        }

        .difficulty.easy { background: #2ecc71; color: white; }
        .difficulty.medium { background: #f39c12; color: white; }
        .difficulty.hard { background: #e74c3c; color: white; }

        .article {
            background: white;
            padding: 15px;
            margin: 15px 0;
            border-radius: 5px;
            border: 1px solid #ddd;
        }

        .article a {
            color: #3498db;
            text-decoration: none;
            font-weight: bold;
        }

        .article a:hover {
            text-decoration: underline;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }

        .stat-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }

        .stat-value {
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }

        .stat-label {
            font-size: 0.9em;
            opacity: 0.9;
        }

        .footer {
            margin-top: 50px;
            padding-top: 20px;
            border-top: 2px solid #ecf0f1;
            text-align: center;
            color: #7f8c8d;
        }

        @media print {
            body {
                background: white;
                padding: 0;
            }
            .container {
                box-shadow: none;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>{{ title }}</h1>

        <div class="meta-info">
            <p><strong>Дата создания:</strong> {{ date }}</p>
            <p><strong>Длительность видео:</strong> {{ duration }}</p>
            <p><strong>Количество сегментов:</strong> {{ num_segments }}</p>
        </div>

        {% if overview %}
        <h2>Общий обзор</h2>
        <div class="overview">
            {{ overview }}
        </div>
        {% endif %}

        {% if statistics %}
        <h2>Статистика</h2>
        <div class="stats-grid">
            {% for stat in statistics %}
            <div class="stat-card">
                <div class="stat-value">{{ stat.value }}</div>
                <div class="stat-label">{{ stat.label }}</div>
            </div>
            {% endfor %}
        </div>
        {% endif %}

        {% if key_points %}
        <h2>Ключевые тезисы</h2>
        <div class="key-points">
            <ul>
            {% for point in key_points %}
                <li>{{ point }}</li>
            {% endfor %}
            </ul>
        </div>
        {% endif %}

        {% if segments %}
        <h2>Суммаризация по сегментам</h2>
        {% for segment in segments %}
        <div class="segment">
            <div class="segment-header">
                <h3>Сегмент {{ segment.id + 1 }}</h3>
                <span class="timestamp">{{ segment.timestamp }}</span>
            </div>
            <div class="summary">
                {{ segment.summary }}
            </div>
        </div>
        {% endfor %}
        {% endif %}

        {% if glossary %}
        <h2>Глоссарий терминов</h2>
        {% for term in glossary %}
        <div class="glossary-term">
            <strong>{{ term.term }}</strong> (встречается {{ term.frequency }} раз)
        </div>
        {% endfor %}
        {% endif %}

        {% if questions %}
        <h2>Вопросы для самопроверки</h2>
        {% for question in questions %}
        <div class="question">
            <span class="difficulty {{ question.difficulty }}">{{ question.difficulty }}</span>
            <p><strong>{{ question.question }}</strong></p>
            {% if question.timestamp %}
            <p><small>Таймкод: {{ question.timestamp }}</small></p>
            {% endif %}
        </div>
        {% endfor %}
        {% endif %}

        {% if articles %}
        <h2>Рекомендуемые материалы</h2>
        {% for article in articles %}
        <div class="article">
            <h4><a href="{{ article.url }}" target="_blank">{{ article.title }}</a></h4>
            <p><small>Источник: {{ article.source }}</small></p>
            {% if article.snippet %}
            <p>{{ article.snippet }}</p>
            {% endif %}
        </div>
        {% endfor %}
        {% endif %}

        <div class="footer">
            <p>Создано системой интеллектуального анализа видео</p>
            <p>{{ date }}</p>
        </div>
    </div>
</body>
</html>
        """
        return Template(template_str)

    def load_all_data(self, artifacts_dir: Path) -> Dict:
        """Загрузка всех данных из артефактов"""
        data = {}

        # Финальная суммаризация
        final_summary_path = artifacts_dir / "final_summary.json"
        if final_summary_path.exists():
            with open(final_summary_path, 'r', encoding='utf-8') as f:
                data["final_summary"] = json.load(f)

        # Суммаризации сегментов
        summaries_path = artifacts_dir / "summaries_per_segment.json"
        if summaries_path.exists():
            with open(summaries_path, 'r', encoding='utf-8') as f:
                data["summaries"] = json.load(f)

        # Термины
        terms_path = artifacts_dir / "terms_and_entities.json"
        if terms_path.exists():
            with open(terms_path, 'r', encoding='utf-8') as f:
                data["terms"] = json.load(f)

        # Вопросы
        questions_path = artifacts_dir / "questions.json"
        if questions_path.exists():
            with open(questions_path, 'r', encoding='utf-8') as f:
                data["questions"] = json.load(f)

        # Статьи
        articles_path = artifacts_dir / "related_articles.json"
        if articles_path.exists():
            with open(articles_path, 'r', encoding='utf-8') as f:
                data["articles"] = json.load(f)

        return data

    def prepare_template_data(self, data: Dict) -> Dict:
        """Подготовка данных для шаблона"""
        template_data = {
            "title": "Анализ образовательного видео",
            "date": datetime.now().strftime("%d.%m.%Y %H:%M"),
            "duration": "N/A",
            "num_segments": 0,
            "overview": "",
            "key_points": [],
            "statistics": [],
            "segments": [],
            "glossary": [],
            "questions": [],
            "articles": []
        }

        # Финальная суммаризация
        if "final_summary" in data:
            fs = data["final_summary"]
            template_data["overview"] = fs.get("overview", "")
            template_data["key_points"] = fs.get("key_points", [])

            stats = fs.get("statistics", {})
            template_data["duration"] = self._format_time(stats.get("total_duration_seconds", 0))
            template_data["num_segments"] = stats.get("num_segments", 0)

            template_data["statistics"] = [
                {"value": stats.get("total_words", 0), "label": "Слов в транскрипции"},
                {"value": stats.get("num_segments", 0), "label": "Сегментов"},
                {"value": f"{stats.get('words_per_minute', 0):.0f}", "label": "Слов/мин"},
            ]

        # Сегменты
        if "summaries" in data:
            summaries = data["summaries"]
            for seg in summaries.get("segments", [])[:10]:  # Топ-10 сегментов
                template_data["segments"].append({
                    "id": seg["id"],
                    "timestamp": self._format_time(seg["start"]),
                    "summary": seg["summary"]
                })

        # Глоссарий
        if "terms" in data:
            terms = data["terms"]
            template_data["glossary"] = terms["glossary"]["technical_terms"][:20]  # Топ-20

        # Вопросы
        if "questions" in data:
            questions = data["questions"]
            for q in questions.get("questions", [])[:15]:  # Топ-15
                template_data["questions"].append({
                    "question": q["question"],
                    "difficulty": q.get("difficulty", "medium"),
                    "timestamp": self._format_time(q["timestamp"]) if "timestamp" in q else None
                })

        # Статьи
        if "articles" in data:
            articles = data["articles"]
            template_data["articles"] = articles.get("articles", [])[:10]  # Топ-10

        return template_data

    def export_html(self, artifacts_dir: Path) -> Path:
        """Экспорт в HTML"""
        print("[INFO] Generating HTML report...")

        # Загрузка данных
        data = self.load_all_data(artifacts_dir)
        template_data = self.prepare_template_data(data)

        # Рендеринг шаблона
        html_content = self.html_template.render(**template_data)

        # Сохранение
        html_path = artifacts_dir / "report.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"[✓] HTML report saved: {html_path}")
        return html_path

    def export_pdf(self, html_path: Path) -> Path:
        """Экспорт HTML в PDF"""
        print("[INFO] Generating PDF report...")

        try:
            from weasyprint import HTML

            pdf_path = html_path.parent / "report.pdf"
            HTML(filename=str(html_path)).write_pdf(str(pdf_path))

            print(f"[✓] PDF report saved: {pdf_path}")
            return pdf_path

        except ImportError:
            print("[WARN] WeasyPrint not installed. Skipping PDF generation.")
            print("[INFO] Install with: pip install weasyprint")
            return None
        except Exception as e:
            print(f"[WARN] PDF generation failed: {e}")
            return None

    def export_all(self, artifacts_dir: Path) -> Dict[str, Path]:
        """Экспорт в HTML (PDF отключен)"""
        print(f"\n{'=' * 60}")
        print("[INFO] Starting report generation")
        print(f"{'=' * 60}\n")

        results = {}

        # HTML
        html_path = self.export_html(artifacts_dir)
        results["html"] = html_path

        # PDF отключен - пользователь может сохранить через браузер
        print("\n[INFO] PDF generation is disabled (to avoid WeasyPrint dependency issues)")
        print("[INFO] You can save as PDF using your browser: File > Print > Save as PDF")

        print(f"\n[✓] Report generation complete!")
        return results

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Форматирование времени"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours}ч {minutes}м {secs}с"
        elif minutes > 0:
            return f"{minutes}м {secs}с"
        else:
            return f"{secs}с"


def main():
    """Пример использования"""
    import argparse

    parser = argparse.ArgumentParser(description="Export analysis results to HTML/PDF")
    parser.add_argument("artifacts_dir", help="Path to artifacts directory")
    parser.add_argument("--no-pdf", action="store_true", help="Skip PDF generation")

    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir)

    # Создание экспортёра
    exporter = ReportExporter()

    # Экспорт
    results = exporter.export_all(artifacts_dir)

    # Обновление checkpoint
    checkpoint_path = artifacts_dir / "checkpoint.json"
    if checkpoint_path.exists():
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            checkpoint = json.load(f)

        checkpoint["stage"] = "export_complete"
        checkpoint["files"]["report_html"] = str(results.get("html", ""))
        # PDF generation disabled

        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)

    print(f"\n[SUCCESS] Export complete!")
    print(f"[INFO] HTML: {results.get('html', 'N/A')}")
    print(f"[INFO] PDF: Disabled (use browser to save as PDF)")


if __name__ == "__main__":
    main()