#!/usr/bin/env python3
"""
Веб-интерфейс для Video Intelligence System
"""
from flask import Flask, render_template, jsonify, send_from_directory
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Optional

app = Flask(__name__)
app.config['ARTIFACTS_DIR'] = Path('artifacts')
app.config['JSON_AS_ASCII'] = False


def get_all_sessions() -> List[Dict]:
    """Получить список всех обработанных видео"""
    artifacts_dir = app.config['ARTIFACTS_DIR']

    if not artifacts_dir.exists():
        return []

    sessions = []

    for session_dir in sorted(artifacts_dir.iterdir(), reverse=True):
        if not session_dir.is_dir():
            continue

        # Пропускаем служебные директории
        if session_dir.name in ['checkpoints', 'latest']:
            continue

        checkpoint_path = session_dir / 'checkpoint.json'

        if checkpoint_path.exists():
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)

            # Определяем статус обработки
            stage = checkpoint.get('stage', 'unknown')

            # Загружаем базовую информацию
            info = {
                'id': session_dir.name,
                'path': str(session_dir),
                'stage': stage,
                'timestamp': checkpoint.get('timestamp', 0),
                'video_source': checkpoint.get('video_source', 'N/A'),
                'files': checkpoint.get('files', {})
            }

            # Добавляем статистику если есть финальная суммаризация
            final_summary_path = session_dir / 'final_summary.json'
            if final_summary_path.exists():
                with open(final_summary_path, 'r', encoding='utf-8') as f:
                    summary = json.load(f)
                    info['statistics'] = summary.get('statistics', {})

            sessions.append(info)

    return sessions


def load_session_data(session_id: str) -> Optional[Dict]:
    """Загрузить все данные сессии"""
    session_dir = app.config['ARTIFACTS_DIR'] / session_id

    if not session_dir.exists():
        return None

    data = {
        'id': session_id,
        'path': str(session_dir)
    }

    # Checkpoint
    checkpoint_path = session_dir / 'checkpoint.json'
    if checkpoint_path.exists():
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            data['checkpoint'] = json.load(f)

    # Транскрипция
    transcript_path = session_dir / 'transcript_raw.json'
    if transcript_path.exists():
        with open(transcript_path, 'r', encoding='utf-8') as f:
            data['transcript'] = json.load(f)

    # Сегменты
    segments_path = session_dir / 'segments_semantic.json'
    if segments_path.exists():
        with open(segments_path, 'r', encoding='utf-8') as f:
            data['segments'] = json.load(f)

    # Суммаризация
    summaries_path = session_dir / 'summaries_per_segment.json'
    if summaries_path.exists():
        with open(summaries_path, 'r', encoding='utf-8') as f:
            data['summaries'] = json.load(f)

    # Финальная суммаризация
    final_summary_path = session_dir / 'final_summary.json'
    if final_summary_path.exists():
        with open(final_summary_path, 'r', encoding='utf-8') as f:
            data['final_summary'] = json.load(f)

    # Термины
    terms_path = session_dir / 'terms_and_entities.json'
    if terms_path.exists():
        with open(terms_path, 'r', encoding='utf-8') as f:
            data['terms'] = json.load(f)

    # Вопросы
    questions_path = session_dir / 'questions.json'
    if questions_path.exists():
        with open(questions_path, 'r', encoding='utf-8') as f:
            data['questions'] = json.load(f)

    # Статьи
    articles_path = session_dir / 'related_articles.json'
    if articles_path.exists():
        with open(articles_path, 'r', encoding='utf-8') as f:
            data['articles'] = json.load(f)

    return data


@app.template_filter('format_time')
def format_time(seconds):
    """Форматирование времени"""
    if not seconds:
        return "N/A"

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"


@app.template_filter('format_datetime')
def format_datetime(timestamp):
    """Форматирование даты и времени"""
    if not timestamp:
        return "N/A"

    dt = datetime.fromtimestamp(timestamp)
    return dt.strftime("%d.%m.%Y %H:%M")


@app.route('/')
def index():
    """Главная страница со списком обработанных видео"""
    sessions = get_all_sessions()
    return render_template('index.html', sessions=sessions)


@app.route('/session/<session_id>')
def session_view(session_id):
    """Детальный просмотр результатов обработки"""
    data = load_session_data(session_id)

    if not data:
        return "Session not found", 404

    return render_template('session.html', data=data)


@app.route('/api/sessions')
def api_sessions():
    """API: список сессий"""
    sessions = get_all_sessions()
    return jsonify(sessions)


@app.route('/api/session/<session_id>')
def api_session(session_id):
    """API: данные конкретной сессии"""
    data = load_session_data(session_id)

    if not data:
        return jsonify({'error': 'Session not found'}), 404

    return jsonify(data)


@app.route('/static/<path:filename>')
def serve_static(filename):
    """Отдача статических файлов"""
    return send_from_directory('static', filename)


if __name__ == '__main__':
    print("=" * 60)
    print("Video Intelligence System - Web Interface")
    print("=" * 60)
    print(f"Artifacts directory: {app.config['ARTIFACTS_DIR']}")
    print(f"Starting server at http://localhost:5000")
    print("=" * 60)

    app.run(debug=True, host='0.0.0.0', port=5000)
