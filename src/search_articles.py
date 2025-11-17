# src/search_articles.py
"""
Поиск релевантных статей и материалов
"""
import json
import time
import requests
from pathlib import Path
from typing import List, Dict
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import numpy as np


class ArticleSearcher:
    """Поиск релевантных статей по темам видео"""

    def __init__(
            self,
            enable_scraping: bool = True,
            rate_limit_delay: int = 2,
            max_articles: int = 10
    ):
        """
        Args:
            enable_scraping: разрешить веб-скрейпинг
            rate_limit_delay: задержка между запросами (сек)
            max_articles: максимальное количество статей
        """
        self.enable_scraping = enable_scraping
        self.rate_limit_delay = rate_limit_delay
        self.max_articles = max_articles

        if enable_scraping:
            print("[INFO] Web scraping enabled")
            print(f"[INFO] Rate limit: {rate_limit_delay}s between requests")
        else:
            print("[WARN] Web scraping disabled (use --enable-scraping to enable)")

        # Загрузка модели для семантического ранжирования
        print("[INFO] Loading sentence transformer for ranking...")
        self.model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            cache_folder="models/sentence_transformers"
        )
        print("[✓] Model loaded")

    def search_google_scholar(
            self,
            query: str,
            num_results: int = 5
    ) -> List[Dict]:
        """
        Поиск в Google Scholar (без scraping, только через API если есть)
        Заглушка - для production нужен API ключ
        """
        print(f"[INFO] Searching Google Scholar: {query}")

        # В реальности здесь был бы вызов API
        # Например: https://serpapi.com/google-scholar-api

        # Заглушка
        return [{
            "title": f"Academic article about {query}",
            "url": "https://scholar.google.com/",
            "source": "Google Scholar",
            "snippet": f"Research on {query}...",
            "year": 2023
        }]

    def search_wikipedia(
            self,
            query: str,
            lang: str = "ru"
    ) -> List[Dict]:
        """
        Поиск в Wikipedia через API
        """
        print(f"[INFO] Searching Wikipedia: {query}")

        api_url = f"https://{lang}.wikipedia.org/w/api.php"

        # Поиск статей
        search_params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": 5,
            "format": "json"
        }

        # Правильный User-Agent для Wikipedia API
        headers = {
            'User-Agent': 'VideoIntelligenceSystem/1.0 (Educational project; https://github.com/video-intelligence) Python-requests'
        }

        try:
            # Делаем запрос с правильным User-Agent
            response = requests.get(
                api_url,
                params=search_params,
                headers=headers,
                timeout=15
            )
            response.raise_for_status()
            data = response.json()

            results = []
            for item in data.get("query", {}).get("search", []):
                page_id = item["pageid"]
                title = item["title"]
                snippet = item.get("snippet", "")

                # Получаем URL страницы
                url = f"https://{lang}.wikipedia.org/wiki/{title.replace(' ', '_')}"

                results.append({
                    "title": title,
                    "url": url,
                    "source": "Wikipedia",
                    "snippet": BeautifulSoup(snippet, "html.parser").get_text(),
                    "page_id": page_id
                })

            # Увеличенная задержка для соблюдения rate limits
            time.sleep(self.rate_limit_delay + 1)
            return results

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                print(f"[WARN] Wikipedia API blocked request (403). Try again later or check User-Agent.")
            else:
                print(f"[WARN] Wikipedia search failed with HTTP {e.response.status_code}: {e}")
            return []
        except Exception as e:
            print(f"[WARN] Wikipedia search failed: {e}")
            return []

    def search_habr(
            self,
            query: str
    ) -> List[Dict]:
        """
        Поиск на Habr.com (для технических тем)
        """
        if not self.enable_scraping:
            return []

        print(f"[INFO] Searching Habr: {query}")

        search_url = f"https://habr.com/ru/search/?q={query}&target_type=posts"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        try:
            response = requests.get(search_url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")
            articles = soup.find_all("article", class_="tm-articles-list__item", limit=5)

            results = []
            for article in articles:
                title_elem = article.find("a", class_="tm-title__link")
                if not title_elem:
                    continue

                title = title_elem.get_text(strip=True)
                url = "https://habr.com" + title_elem["href"]

                snippet_elem = article.find("div", class_="article-formatted-body")
                snippet = snippet_elem.get_text(strip=True)[:200] if snippet_elem else ""

                results.append({
                    "title": title,
                    "url": url,
                    "source": "Habr",
                    "snippet": snippet
                })

            time.sleep(self.rate_limit_delay)
            return results

        except Exception as e:
            print(f"[WARN] Habr search failed: {e}")
            return []

    def rank_articles_by_relevance(
            self,
            articles: List[Dict],
            query_embedding: np.ndarray,
            top_k: int = None
    ) -> List[Dict]:
        """
        Ранжирование статей по семантической релевантности
        """
        if not articles:
            return []

        if top_k is None:
            top_k = len(articles)

        print(f"[INFO] Ranking {len(articles)} articles by relevance")

        # Получаем эмбеддинги для статей
        article_texts = [
            f"{art['title']} {art.get('snippet', '')}"
            for art in articles
        ]

        article_embeddings = self.model.encode(
            article_texts,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        # Вычисляем косинусное сходство
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(
            query_embedding.reshape(1, -1),
            article_embeddings
        )[0]

        # Добавляем скор релевантности
        for i, art in enumerate(articles):
            art["relevance_score"] = float(similarities[i])

        # Сортируем по релевантности
        ranked = sorted(articles, key=lambda x: x["relevance_score"], reverse=True)

        return ranked[:top_k]

    def search_for_topics(
            self,
            topics: List[str],
            max_articles_per_topic: int = 3
    ) -> Dict[str, List[Dict]]:
        """
        Поиск статей для списка тем
        """
        print(f"\n{'=' * 60}")
        print(f"[INFO] Searching articles for {len(topics)} topics")
        print(f"{'=' * 60}\n")

        all_results = {}

        for topic in topics:
            print(f"[INFO] Topic: {topic}")

            # Получаем эмбеддинг запроса
            query_embedding = self.model.encode(topic, convert_to_numpy=True)

            # Поиск в разных источниках
            results = []

            # Wikipedia
            wiki_results = self.search_wikipedia(topic)
            results.extend(wiki_results)

            # Habr (если включен scraping)
            if self.enable_scraping:
                habr_results = self.search_habr(topic)
                results.extend(habr_results)

            # Ранжирование
            ranked = self.rank_articles_by_relevance(
                results,
                query_embedding,
                top_k=max_articles_per_topic
            )

            all_results[topic] = ranked
            print(f"[✓] Found {len(ranked)} articles for '{topic}'")
            print()

        return all_results

    def process_terms_file(
            self,
            terms_path: Path,
            output_dir: Path = None
    ) -> Dict:
        """
        Полный процесс поиска статей
        """
        print(f"\n{'=' * 60}")
        print("[INFO] Starting article search")
        print(f"{'=' * 60}\n")

        # Загрузка терминов
        with open(terms_path, 'r', encoding='utf-8') as f:
            terms_data = json.load(f)

        # Извлекаем топ-термины как темы для поиска
        technical_terms = terms_data["glossary"]["technical_terms"]
        topics = [term["term"] for term in technical_terms[:5]]  # Топ-5 терминов

        print(f"[INFO] Selected topics: {topics}")

        # Поиск статей
        articles_by_topic = self.search_for_topics(
            topics,
            max_articles_per_topic=self.max_articles // len(topics) if topics else 1
        )

        # Объединяем все статьи
        all_articles = []
        for topic, articles in articles_by_topic.items():
            for art in articles:
                art["topic"] = topic
                all_articles.append(art)

        # Удаляем дубликаты по URL
        seen_urls = set()
        unique_articles = []
        for art in all_articles:
            if art["url"] not in seen_urls:
                seen_urls.add(art["url"])
                unique_articles.append(art)

        # Ограничиваем количество
        unique_articles = unique_articles[:self.max_articles]

        result = {
            "total_articles": len(unique_articles),
            "articles": unique_articles,
            "topics_searched": topics,
            "sources": list(set(art["source"] for art in unique_articles))
        }

        print(f"\n[✓] Article search complete!")
        print(f"[INFO] Found {len(unique_articles)} unique articles")
        print(f"[INFO] Sources: {result['sources']}")

        # Сохранение
        if output_dir is None:
            output_dir = terms_path.parent

        self.save_articles(result, output_dir)

        return result

    def save_articles(self, articles_data: Dict, output_dir: Path):
        """Сохранение результатов поиска"""

        # JSON формат
        json_path = output_dir / "related_articles.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(articles_data, f, ensure_ascii=False, indent=2)
        print(f"[✓] Saved: {json_path}")

        # TXT формат (читаемый)
        txt_path = output_dir / "articles_list.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("РЕКОМЕНДУЕМЫЕ СТАТЬИ И МАТЕРИАЛЫ\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Всего статей: {articles_data['total_articles']}\n")
            f.write(f"Источники: {', '.join(articles_data['sources'])}\n")
            f.write(f"Темы поиска: {', '.join(articles_data['topics_searched'])}\n\n")

            f.write("=" * 70 + "\n\n")

            # Группировка по темам
            by_topic = {}
            for art in articles_data["articles"]:
                topic = art.get("topic", "Other")
                if topic not in by_topic:
                    by_topic[topic] = []
                by_topic[topic].append(art)

            for topic, articles in by_topic.items():
                f.write(f"ТЕМА: {topic}\n")
                f.write("-" * 70 + "\n\n")

                for i, art in enumerate(articles, 1):
                    f.write(f"{i}. {art['title']}\n")
                    f.write(f"   Источник: {art['source']}\n")
                    f.write(f"   URL: {art['url']}\n")

                    if "relevance_score" in art:
                        f.write(f"   Релевантность: {art['relevance_score']:.2f}\n")

                    if art.get("snippet"):
                        f.write(f"   Описание: {art['snippet'][:150]}...\n")

                    f.write("\n")

                f.write("=" * 70 + "\n\n")

        print(f"[✓] Saved: {txt_path}")

def main():
    """Пример использования"""
    import argparse

    parser = argparse.ArgumentParser(description="Search for related articles")
    parser.add_argument("terms", help="Path to terms_and_entities.json")
    parser.add_argument("--enable-scraping", action="store_true",
                        help="Enable web scraping (be careful with rate limits)")
    parser.add_argument("--max-articles", type=int, default=10,
                        help="Maximum number of articles")
    parser.add_argument("--rate-limit", type=int, default=2,
                        help="Delay between requests (seconds)")

    args = parser.parse_args()

    terms_path = Path(args.terms)
    output_dir = terms_path.parent

    # Создание поисковика
    searcher = ArticleSearcher(
        enable_scraping=args.enable_scraping,
        rate_limit_delay=args.rate_limit,
        max_articles=args.max_articles
    )

    # Поиск статей
    articles = searcher.process_terms_file(terms_path, output_dir)

    # Обновление checkpoint
    checkpoint_path = output_dir / "checkpoint.json"
    if checkpoint_path.exists():
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            checkpoint = json.load(f)

        checkpoint["stage"] = "article_search_complete"
        checkpoint["files"]["articles_json"] = str(output_dir / "related_articles.json")
        checkpoint["files"]["articles_txt"] = str(output_dir / "articles_list.txt")

        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)

    print(f"\n[SUCCESS] Article search complete!")

    if __name__ == "__main__":
        main()