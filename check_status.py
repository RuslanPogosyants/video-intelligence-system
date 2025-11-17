#!/usr/bin/env python3
"""Проверка статуса обработки"""
import requests
import sys

task_id = "task_1763402857975"

try:
    response = requests.get(f"http://localhost:5000/api/process/status/{task_id}")
    if response.status_code == 200:
        data = response.json()
        print("=" * 60)
        print(f"Статус задачи: {task_id}")
        print("=" * 60)
        print(f"Статус: {data.get('status')}")
        print(f"Этап: {data.get('stage')}")
        print(f"Прогресс: {data.get('progress')}%")
        print(f"Времени прошло: {data.get('elapsed', 0):.1f} сек")
        print("=" * 60)

        if data.get('error'):
            print("\nОШИБКА:")
            print(data['error'])
            print("=" * 60)

        if data.get('output'):
            print("\nПОСЛЕДНИЙ ВЫВОД:")
            lines = data['output'].split('\n')
            print('\n'.join(lines[-20:]))  # Последние 20 строк
            print("=" * 60)
    else:
        print(f"Ошибка: {response.status_code}")
except Exception as e:
    print(f"Не удалось подключиться: {e}")
    print("Убедитесь что веб-сервер запущен на localhost:5000")
