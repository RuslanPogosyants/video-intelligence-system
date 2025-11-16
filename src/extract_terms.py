# src/extract_terms.py
"""
Извлечение терминов и именованных сущностей
"""
import json
import spacy
from pathlib import Path
from typing import List, Dict, Set, Tuple
from collections import Counter
import re


class TermExtractor:
    """Извлечение терминов и NER"""

    def __init__(self, model_name: str = "ru_core_news_lg"):
        """
        Args:
            model_name: название SpaCy модели
        """
        print(f"[INFO] Loading SpaCy model: {model_name}")

        try:
            self.nlp = spacy.load(model_name)
            print(f"[✓] Model loaded successfully")
        except OSError:
            print(f"[✗] Model not found. Downloading...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", model_name])
            self.nlp = spacy.load(model_name)
            print(f"[✓] Model loaded successfully")

        # Стоп-слова для фильтрации
        self.stop_words = self.nlp.Defaults.stop_words

    def extract_entities(self, text: str) -> Dict[str, List[Dict]]:
        """
        Извлечение именованных сущностей

        Returns:
            Dict с категориями сущностей
        """
        doc = self.nlp(text)

        entities = {
            "PERSON": [],  # Персоны
            "ORG": [],  # Организации
            "LOC": [],  # Локации
            "GPE": [],  # Географические/политические сущности
            "DATE": [],  # Даты
            "MONEY": [],  # Деньги
            "PERCENT": [],  # Проценты
            "MISC": []  # Прочее
        }

        for ent in doc.ents:
            label = ent.label_
            if label in entities:
                entities[label].append({
                    "text": ent.text,
                    "label": label,
                    "start": ent.start_char,
                    "end": ent.end_char
                })
            else:
                entities["MISC"].append({
                    "text": ent.text,
                    "label": label,
                    "start": ent.start_char,
                    "end": ent.end_char
                })

        return entities

    def extract_noun_phrases(self, text: str, min_length: int = 2) -> List[str]:
        """
        Извлечение именных групп (потенциальные термины)
        """
        doc = self.nlp(text)

        noun_phrases = []
        for chunk in doc.noun_chunks:
            # Фильтруем короткие и стоп-слова
            text = chunk.text.strip()
            if len(text.split()) >= min_length and text.lower() not in self.stop_words:
                noun_phrases.append(text)

        return noun_phrases

    def extract_technical_terms(
            self,
            text: str,
            min_frequency: int = 2,
            min_word_length: int = 5
    ) -> List[Tuple[str, int]]:
        """
        Извлечение технических терминов через частотный анализ
        """
        doc = self.nlp(text)

        # Собираем существительные и прилагательные
        candidates = []
        for token in doc:
            if token.pos_ in ["NOUN", "ADJ", "PROPN"]:
                # Лемматизация
                lemma = token.lemma_.lower()
                # Фильтрация
                if (len(lemma) >= min_word_length and
                        lemma not in self.stop_words and
                        lemma.isalpha()):
                    candidates.append(lemma)

        # Подсчёт частоты
        term_freq = Counter(candidates)

        # Фильтруем по минимальной частоте
        terms = [
            (term, freq)
            for term, freq in term_freq.most_common()
            if freq >= min_frequency
        ]

        return terms

    def extract_multi_word_terms(
            self,
            text: str,
            min_frequency: int = 2
    ) -> List[Tuple[str, int]]:
        """
        Извлечение многословных терминов (биграммы, триграммы)
        """
        doc = self.nlp(text)

        # Извлечение биграмм
        bigrams = []
        tokens = [token for token in doc if token.pos_ in ["NOUN", "ADJ", "PROPN"]]

        for i in range(len(tokens) - 1):
            bigram = f"{tokens[i].lemma_} {tokens[i + 1].lemma_}"
            bigrams.append(bigram.lower())

        # Извлечение триграмм
        trigrams = []
        for i in range(len(tokens) - 2):
            trigram = f"{tokens[i].lemma_} {tokens[i + 1].lemma_} {tokens[i + 2].lemma_}"
            trigrams.append(trigram.lower())

        # Подсчёт частоты
        all_terms = bigrams + trigrams
        term_freq = Counter(all_terms)

        # Фильтруем
        terms = [
            (term, freq)
            for term, freq in term_freq.most_common()
            if freq >= min_frequency
        ]

        return terms

    def create_glossary(
            self,
            terms: List[Tuple[str, int]],
            entities: Dict[str, List[Dict]],
            max_terms: int = 50
    ) -> Dict:
        """
        Создание глоссария терминов
        """
        print(f"[INFO] Creating glossary with up to {max_terms} terms")

        # Сортируем термины по частоте
        sorted_terms = sorted(terms, key=lambda x: x[1], reverse=True)[:max_terms]

        glossary = {
            "technical_terms": [
                {"term": term, "frequency": freq, "definition": ""}
                for term, freq in sorted_terms
            ],
            "named_entities": {
                "persons": list(set(e["text"] for e in entities.get("PERSON", []))),
                "organizations": list(set(e["text"] for e in entities.get("ORG", []))),
                "locations": list(set(e["text"] for e in entities.get("LOC", []) + entities.get("GPE", []))),
                "dates": list(set(e["text"] for e in entities.get("DATE", []))),
            }
        }

        return glossary

    def process_transcript(
            self,
            transcript_path: Path,
            output_dir: Path = None
    ) -> Dict:
        """
        Полный процесс извлечения терминов
        """
        print(f"\n{'=' * 60}")
        print("[INFO] Starting term extraction")
        print(f"{'=' * 60}\n")

        # Загрузка транскрипции
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript_data = json.load(f)

        full_text = transcript_data["full_text"]

        # Извлечение сущностей
        print("[INFO] Extracting named entities...")
        entities = self.extract_entities(full_text)

        # Извлечение терминов
        print("[INFO] Extracting technical terms...")
        single_terms = self.extract_technical_terms(full_text)
        multi_terms = self.extract_multi_word_terms(full_text)

        all_terms = single_terms + multi_terms

        # Создание глоссария
        glossary = self.create_glossary(all_terms, entities)

        # Статистика
        stats = {
            "total_entities": sum(len(entities[k]) for k in entities),
            "total_terms": len(all_terms),
            "unique_persons": len(glossary["named_entities"]["persons"]),
            "unique_orgs": len(glossary["named_entities"]["organizations"]),
            "unique_locations": len(glossary["named_entities"]["locations"])
        }

        result = {
            "glossary": glossary,
            "entities_detailed": entities,
            "all_terms": [{"term": t, "frequency": f} for t, f in all_terms[:100]],
            "statistics": stats
        }

        print(f"\n[✓] Term extraction complete!")
        print(f"[INFO] Entities found: {stats['total_entities']}")
        print(f"[INFO] Terms extracted: {stats['total_terms']}")

        # Сохранение
        if output_dir is None:
            output_dir = transcript_path.parent

        self.save_results(result, output_dir)

        return result

    def save_results(self, results: Dict, output_dir: Path):
        """Сохранение результатов"""

        # JSON формат
        json_path = output_dir / "terms_and_entities.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"[✓] Saved: {json_path}")

        # TXT формат (глоссарий)
        txt_path = output_dir / "glossary.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("ГЛОССАРИЙ ТЕРМИНОВ И СУЩНОСТЕЙ\n")
            f.write("=" * 70 + "\n\n")

            # Статистика
            stats = results["statistics"]
            f.write("СТАТИСТИКА:\n")
            f.write("-" * 70 + "\n")
            f.write(f"Всего сущностей: {stats['total_entities']}\n")
            f.write(f"Всего терминов: {stats['total_terms']}\n")
            f.write(f"Уникальных персон: {stats['unique_persons']}\n")
            f.write(f"Уникальных организаций: {stats['unique_orgs']}\n")
            f.write(f"Уникальных локаций: {stats['unique_locations']}\n\n")

            # Технические термины
            f.write("ТЕХНИЧЕСКИЕ ТЕРМИНЫ:\n")
            f.write("-" * 70 + "\n")
            for i, term_data in enumerate(results["glossary"]["technical_terms"], 1):
                f.write(f"{i}. {term_data['term']} (встречается {term_data['frequency']} раз)\n")
            f.write("\n")

            # Персоны
            persons = results["glossary"]["named_entities"]["persons"]
            if persons:
                f.write("ПЕРСОНЫ:\n")
                f.write("-" * 70 + "\n")
                for person in persons:
                    f.write(f"• {person}\n")
                f.write("\n")

            # Организации
            orgs = results["glossary"]["named_entities"]["organizations"]
            if orgs:
                f.write("ОРГАНИЗАЦИИ:\n")
                f.write("-" * 70 + "\n")
                for org in orgs:
                    f.write(f"• {org}\n")
                f.write("\n")

            # Локации
            locs = results["glossary"]["named_entities"]["locations"]
            if locs:
                f.write("ЛОКАЦИИ:\n")
                f.write("-" * 70 + "\n")
                for loc in locs:
                    f.write(f"• {loc}\n")
                f.write("\n")

        print(f"[✓] Saved: {txt_path}")


def main():
    """Пример использования"""
    import argparse

    parser = argparse.ArgumentParser(description="Extract terms and named entities")
    parser.add_argument("transcript", help="Path to transcript_raw.json")
    parser.add_argument("--model", default="ru_core_news_lg", help="SpaCy model")

    args = parser.parse_args()

    transcript_path = Path(args.transcript)
    output_dir = transcript_path.parent

    # Создание экстрактора
    extractor = TermExtractor(model_name=args.model)

    # Извлечение терминов
    results = extractor.process_transcript(transcript_path, output_dir)

    # Обновление checkpoint
    checkpoint_path = output_dir / "checkpoint.json"
    if checkpoint_path.exists():
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            checkpoint = json.load(f)

        checkpoint["stage"] = "term_extraction_complete"
        checkpoint["files"]["terms_json"] = str(output_dir / "terms_and_entities.json")
        checkpoint["files"]["glossary"] = str(output_dir / "glossary.txt")

        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)

    print(f"\n[SUCCESS] Term extraction complete!")


if __name__ == "__main__":
    main()