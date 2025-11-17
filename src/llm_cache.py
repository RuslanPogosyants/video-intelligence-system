# src/llm_cache.py
"""
Кэширование LLM запросов для экономии токенов и времени
"""
import json
import hashlib
from pathlib import Path
from typing import Optional, Any
from datetime import datetime, timedelta


class LLMCache:
    """Кэш для LLM запросов"""

    def __init__(
            self,
            cache_dir: str = ".llm_cache",
            ttl_days: int = 30,
            enabled: bool = True
    ):
        """
        Args:
            cache_dir: директория для хранения кэша
            ttl_days: время жизни кэша в днях
            enabled: включен ли кэш
        """
        self.cache_dir = Path(cache_dir)
        self.ttl_days = ttl_days
        self.enabled = enabled

        if self.enabled:
            self.cache_dir.mkdir(exist_ok=True)
            print(f"[INFO] LLM Cache enabled: {self.cache_dir}")

    def _get_cache_key(self, prompt: str, config: dict) -> str:
        """
        Генерация ключа кэша на основе промпта и конфигурации

        Args:
            prompt: текст промпта
            config: конфигурация LLM (модель, температура и т.д.)

        Returns:
            Хэш ключ для кэша
        """
        # Сериализуем промпт + конфигурацию
        cache_data = {
            "prompt": prompt,
            "config": config
        }

        # Создаем MD5 хэш
        data_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Путь к файлу кэша"""
        return self.cache_dir / f"{cache_key}.json"

    def get(self, prompt: str, config: dict) -> Optional[str]:
        """
        Получить результат из кэша

        Args:
            prompt: текст промпта
            config: конфигурация LLM

        Returns:
            Закэшированный ответ или None
        """
        if not self.enabled:
            return None

        cache_key = self._get_cache_key(prompt, config)
        cache_path = self._get_cache_path(cache_key)

        if not cache_path.exists():
            return None

        try:
            # Читаем кэш
            with open(cache_path, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)

            # Проверяем TTL
            cached_time = datetime.fromisoformat(cached_data["timestamp"])
            expiry_time = cached_time + timedelta(days=self.ttl_days)

            if datetime.now() > expiry_time:
                # Кэш устарел
                cache_path.unlink()
                print(f"[CACHE] Expired: {cache_key[:8]}...")
                return None

            # Возвращаем закэшированный ответ
            print(f"[CACHE] Hit: {cache_key[:8]}... (saved ~{cached_data.get('tokens', 0)} tokens)")
            return cached_data["response"]

        except Exception as e:
            print(f"[WARN] Cache read error: {e}")
            return None

    def set(self, prompt: str, config: dict, response: str, tokens: int = 0):
        """
        Сохранить результат в кэш

        Args:
            prompt: текст промпта
            config: конфигурация LLM
            response: ответ модели
            tokens: количество использованных токенов (опционально)
        """
        if not self.enabled:
            return

        cache_key = self._get_cache_key(prompt, config)
        cache_path = self._get_cache_path(cache_key)

        try:
            # Сохраняем в кэш
            cached_data = {
                "prompt": prompt,
                "config": config,
                "response": response,
                "tokens": tokens,
                "timestamp": datetime.now().isoformat()
            }

            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cached_data, f, ensure_ascii=False, indent=2)

            print(f"[CACHE] Saved: {cache_key[:8]}... (~{tokens} tokens)")

        except Exception as e:
            print(f"[WARN] Cache write error: {e}")

    def clear(self):
        """Очистить весь кэш"""
        if not self.enabled:
            return

        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
            count += 1

        print(f"[CACHE] Cleared {count} entries")

    def get_stats(self) -> dict:
        """Получить статистику кэша"""
        if not self.enabled:
            return {"enabled": False}

        total_files = len(list(self.cache_dir.glob("*.json")))
        total_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.json"))
        total_tokens = 0

        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    total_tokens += data.get("tokens", 0)
            except:
                pass

        return {
            "enabled": True,
            "total_entries": total_files,
            "total_size_mb": total_size / (1024 * 1024),
            "estimated_tokens_saved": total_tokens
        }


def main():
    """Пример использования"""
    # Создаем кэш
    cache = LLMCache(cache_dir=".llm_cache", ttl_days=30)

    # Пример промпта и конфига
    prompt = "Создай обзор лекции про Python"
    config = {
        "model": "GigaChat",
        "temperature": 0.3,
        "max_tokens": 2000
    }

    # Проверяем кэш
    cached_response = cache.get(prompt, config)
    if cached_response:
        print(f"Из кэша: {cached_response}")
    else:
        # Имитация вызова LLM
        response = "Python - это высокоуровневый язык программирования..."
        cache.set(prompt, config, response, tokens=150)

    # Статистика
    stats = cache.get_stats()
    print(f"\nСтатистика кэша: {stats}")


if __name__ == "__main__":
    main()
