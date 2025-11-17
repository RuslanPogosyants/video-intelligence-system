#!/usr/bin/env python3
"""
Веб-интерфейс для Video Intelligence System
"""
from flask import Flask, render_template, jsonify, send_from_directory, request, redirect, url_for
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Optional
import os
import sys
import subprocess
import threading
import time

app = Flask(__name__)
app.config['ARTIFACTS_DIR'] = Path('artifacts')
app.config['UPLOAD_FOLDER'] = Path('artifacts')
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max
app.config['JSON_AS_ASCII'] = False

# Словарь для отслеживания активных процессов обработки
processing_tasks = {}


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


def get_available_audio_files() -> List[Dict]:
    """Получить список доступных аудио файлов"""
    artifacts_dir = app.config['ARTIFACTS_DIR']

    if not artifacts_dir.exists():
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        return []

    audio_files = []

    # Ищем wav, mp3, mp4 файлы
    for ext in ['*.wav', '*.mp3', '*.mp4', '*.m4a']:
        for file_path in artifacts_dir.rglob(ext):
            # Пропускаем файлы внутри обработанных папок
            if 'video_' in str(file_path.parent):
                continue

            stat = file_path.stat()
            audio_files.append({
                'name': file_path.name,
                'path': str(file_path.relative_to(artifacts_dir)),
                'full_path': str(file_path),
                'size': stat.st_size,
                'size_mb': round(stat.st_size / (1024 * 1024), 2),
                'modified': stat.st_mtime,
                'ext': file_path.suffix
            })

    # Сортируем по дате изменения (новые первые)
    audio_files.sort(key=lambda x: x['modified'], reverse=True)

    return audio_files


@app.route('/process')
def process_page():
    """Страница обработки новых файлов"""
    audio_files = get_available_audio_files()
    return render_template('process.html', audio_files=audio_files)


@app.route('/api/audio-files')
def api_audio_files():
    """API: список доступных аудио файлов"""
    files = get_available_audio_files()
    return jsonify(files)


@app.route('/upload', methods=['POST'])
def upload_file():
    """Загрузка нового файла"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Проверяем расширение
    allowed_extensions = {'.wav', '.mp3', '.mp4', '.m4a', '.avi', '.mkv'}
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        return jsonify({'error': f'File type not allowed. Allowed: {", ".join(allowed_extensions)}'}), 400

    # Сохраняем файл
    filename = file.filename
    file_path = app.config['UPLOAD_FOLDER'] / filename

    # Если файл уже существует, добавляем timestamp
    if file_path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name_without_ext = file_path.stem
        filename = f"{name_without_ext}_{timestamp}{file_ext}"
        file_path = app.config['UPLOAD_FOLDER'] / filename

    file.save(str(file_path))

    return jsonify({
        'success': True,
        'filename': filename,
        'path': str(file_path)
    })


def run_processing(file_path: str, task_id: str, options: Dict):
    """Запуск обработки в отдельном потоке"""
    try:
        processing_tasks[task_id]['status'] = 'running'
        processing_tasks[task_id]['stage'] = 'Подготовка...'
        processing_tasks[task_id]['progress'] = 5

        # Формируем команду
        cmd = [
            sys.executable, '-m', 'src.cli', 'process-all',
            file_path,
            '--language', options.get('language', 'ru'),
            '--model', options.get('model', 'base'),
            '--device', options.get('device', 'auto'),
        ]

        # Phase 3: Добавляем флаги для LLM/KeyBERT/Answers
        if options.get('use_llm'):
            cmd.append('--use-llm')

        if options.get('use_keybert'):
            cmd.append('--use-keybert')

        if options.get('with_answers'):
            cmd.append('--with-answers')

        if options.get('skip_questions'):
            cmd.append('--skip-questions')

        if options.get('skip_articles'):
            cmd.append('--skip-articles')

        processing_tasks[task_id]['stage'] = 'Запуск обработки...'
        processing_tasks[task_id]['progress'] = 10

        # Запускаем процесс
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        processing_tasks[task_id]['process'] = process

        # Читаем вывод и обновляем прогресс
        output_lines = []
        error_lines = []

        # Читаем stdout
        if process.stdout:
            for line in process.stdout:
                line_stripped = line.strip()
                output_lines.append(line_stripped)

                # Также ловим строки с ошибками
                if 'error' in line_stripped.lower() or 'exception' in line_stripped.lower():
                    error_lines.append(line_stripped)

                # Обновляем статус на основе вывода
                if 'Транскрибация' in line:
                    processing_tasks[task_id]['stage'] = 'Транскрибация...'
                    processing_tasks[task_id]['progress'] = 20
                elif 'Сегментация' in line:
                    processing_tasks[task_id]['stage'] = 'Сегментация...'
                    processing_tasks[task_id]['progress'] = 35
                elif 'Суммаризация' in line or 'Summarizing' in line:
                    processing_tasks[task_id]['stage'] = 'Суммаризация...'
                    processing_tasks[task_id]['progress'] = 50
                elif 'Мета-анализ' in line:
                    processing_tasks[task_id]['stage'] = 'Мета-анализ...'
                    processing_tasks[task_id]['progress'] = 65
                elif 'Извлечение терминов' in line:
                    processing_tasks[task_id]['stage'] = 'Извлечение терминов...'
                    processing_tasks[task_id]['progress'] = 75
                elif 'Генерация вопросов' in line:
                    processing_tasks[task_id]['stage'] = 'Генерация вопросов...'
                    processing_tasks[task_id]['progress'] = 85
                elif 'Поиск статей' in line:
                    processing_tasks[task_id]['stage'] = 'Поиск статей...'
                    processing_tasks[task_id]['progress'] = 90
                elif 'Экспорт' in line:
                    processing_tasks[task_id]['stage'] = 'Экспорт отчёта...'
                    processing_tasks[task_id]['progress'] = 95
                elif 'ЗАВЕРШЁН' in line or 'SUCCESS' in line:
                    processing_tasks[task_id]['stage'] = 'Завершено!'
                    processing_tasks[task_id]['progress'] = 100

        # Ждём завершения и читаем stderr
        return_code = process.wait()
        stderr_output = ""
        if process.stderr:
            stderr_output = process.stderr.read()

        if return_code == 0:
            processing_tasks[task_id]['status'] = 'completed'
            processing_tasks[task_id]['stage'] = 'Обработка завершена успешно!'
            processing_tasks[task_id]['progress'] = 100
            processing_tasks[task_id]['output'] = '\n'.join(output_lines)
        else:
            # Формируем подробное сообщение об ошибке
            error_msg = f"Процесс завершился с кодом {return_code}\n\n"
            if stderr_output:
                error_msg += f"STDERR:\n{stderr_output}\n\n"
            if error_lines:
                error_msg += f"Ошибки из лога:\n" + "\n".join(error_lines[-10:])  # Последние 10 строк с ошибками

            processing_tasks[task_id]['status'] = 'error'
            processing_tasks[task_id]['stage'] = 'Ошибка обработки'
            processing_tasks[task_id]['error'] = error_msg
            processing_tasks[task_id]['output'] = '\n'.join(output_lines)

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        processing_tasks[task_id]['status'] = 'error'
        processing_tasks[task_id]['stage'] = 'Ошибка при выполнении'
        processing_tasks[task_id]['error'] = f"Exception: {str(e)}\n\nTraceback:\n{error_trace}"


@app.route('/api/process/start', methods=['POST'])
def start_processing():
    """Запуск обработки файла"""
    data = request.get_json()

    if not data or 'file_path' not in data:
        return jsonify({'error': 'No file path provided'}), 400

    file_path = data['file_path']

    # Проверяем существование файла
    if not Path(file_path).exists():
        return jsonify({'error': 'File not found'}), 404

    # Создаём уникальный ID задачи
    task_id = f"task_{int(time.time() * 1000)}"

    # Опции обработки
    options = {
        'language': data.get('language', 'ru'),
        'model': data.get('model', 'base'),
        'device': data.get('device', 'auto'),
        'skip_questions': data.get('skip_questions', False),
        'skip_articles': data.get('skip_articles', False),
        # Phase 3: Новые опции
        'use_llm': data.get('use_llm', True),  # По умолчанию включено
        'use_keybert': data.get('use_keybert', True),  # По умолчанию включено
        'with_answers': data.get('with_answers', True),  # По умолчанию включено
    }

    # Инициализируем задачу
    processing_tasks[task_id] = {
        'status': 'pending',
        'stage': 'Инициализация...',
        'progress': 0,
        'file_path': file_path,
        'started_at': time.time(),
        'options': options
    }

    # Запускаем в отдельном потоке
    thread = threading.Thread(
        target=run_processing,
        args=(file_path, task_id, options)
    )
    thread.daemon = True
    thread.start()

    return jsonify({
        'success': True,
        'task_id': task_id,
        'message': 'Processing started'
    })


@app.route('/api/process/status/<task_id>')
def processing_status(task_id):
    """Получить статус обработки"""
    if task_id not in processing_tasks:
        return jsonify({'error': 'Task not found'}), 404

    task = processing_tasks[task_id]

    return jsonify({
        'task_id': task_id,
        'status': task['status'],
        'stage': task['stage'],
        'progress': task['progress'],
        'started_at': task['started_at'],
        'elapsed': time.time() - task['started_at'],
        'error': task.get('error'),
        'output': task.get('output')
    })


@app.route('/api/process/cancel/<task_id>', methods=['POST'])
def cancel_processing(task_id):
    """Отменить обработку"""
    if task_id not in processing_tasks:
        return jsonify({'error': 'Task not found'}), 404

    task = processing_tasks[task_id]

    if 'process' in task:
        task['process'].terminate()
        task['status'] = 'cancelled'
        task['stage'] = 'Отменено пользователем'

    return jsonify({'success': True})


if __name__ == '__main__':
    print("=" * 60)
    print("Video Intelligence System - Web Interface")
    print("=" * 60)
    print(f"Artifacts directory: {app.config['ARTIFACTS_DIR']}")
    print(f"Starting server at http://localhost:5000")
    print("=" * 60)

    app.run(debug=True, host='0.0.0.0', port=5000)
