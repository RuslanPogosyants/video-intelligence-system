# src/transcribe.py
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
import torch
from faster_whisper import WhisperModel
import yt_dlp
from pydub import AudioSegment
from tqdm import tqdm


class VideoTranscriber:
    """Транскрибация видео с использованием Whisper"""

    def __init__(
            self,
            model_size: str = "base",
            device: str = "auto",
            compute_type: str = "float16",
            output_dir: str = "artifacts"
    ):
        """
        Args:
            model_size: tiny, base, small, medium, large-v2, large-v3
            device: cuda, cpu, auto
            compute_type: float16, int8, int8_float16
            output_dir: директория для сохранения результатов
        """
        self.model_size = model_size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Определение устройства
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Для CPU используем int8
        if self.device == "cpu":
            compute_type = "int8"

        print(f"[INFO] Initializing Whisper model: {model_size}")
        print(f"[INFO] Device: {self.device}, Compute type: {compute_type}")

        # Загрузка модели
        try:
            self.model = WhisperModel(
                model_size,
                device=self.device,
                compute_type=compute_type,
                download_root="models/whisper"
            )
            print("[✓] Model loaded successfully")
        except Exception as e:
            print(f"[✗] Failed to load model: {e}")
            raise

    def download_video(self, url: str, output_path: Path) -> Path:
        """Загрузка видео с YouTube или другого источника"""
        print(f"[INFO] Downloading video from: {url}")

        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': str(output_path / 'input_video.%(ext)s'),
            'quiet': False,
            'no_warnings': False,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                filename = ydl.prepare_filename(info)
                print(f"[✓] Downloaded: {filename}")
                return Path(filename)
        except Exception as e:
            print(f"[✗] Download failed: {e}")
            raise

    def extract_audio(self, video_path: Path, output_path: Path) -> Path:
        """Извлечение аудио из видео"""
        print(f"[INFO] Extracting audio from: {video_path.name}")

        audio_path = output_path / "audio.wav"

        try:
            # Используем pydub для извлечения
            audio = AudioSegment.from_file(str(video_path))

            # Конвертация в моно, 16kHz (оптимально для Whisper)
            audio = audio.set_channels(1)
            audio = audio.set_frame_rate(16000)

            audio.export(str(audio_path), format="wav")
            print(f"[✓] Audio extracted: {audio_path}")
            print(f"[INFO] Duration: {len(audio) / 1000:.2f} seconds")

            return audio_path
        except Exception as e:
            print(f"[✗] Audio extraction failed: {e}")
            raise

    def transcribe(
            self,
            audio_path: Path,
            language: str = "ru",
            task: str = "transcribe"
    ) -> Dict:
        """
        Транскрибация аудио

        Args:
            audio_path: путь к аудиофайлу
            language: код языка (ru, en, etc.)
            task: transcribe или translate

        Returns:
            Dict с сегментами транскрипции
        """
        print(f"[INFO] Transcribing: {audio_path.name}")
        print(f"[INFO] Language: {language}, Task: {task}")

        start_time = time.time()

        try:
            # Транскрибация
            segments, info = self.model.transcribe(
                str(audio_path),
                language=language,
                task=task,
                beam_size=5,
                best_of=5,
                temperature=0.0,
                vad_filter=True,  # Voice Activity Detection
                vad_parameters=dict(
                    min_silence_duration_ms=500
                )
            )

            print(f"[INFO] Detected language: {info.language} (probability: {info.language_probability:.2f})")

            # Сборка результатов
            result_segments = []
            full_text = []

            for segment in tqdm(segments, desc="Processing segments"):
                result_segments.append({
                    "id": segment.id,
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                    "words": [
                        {
                            "word": word.word,
                            "start": word.start,
                            "end": word.end,
                            "probability": word.probability
                        }
                        for word in (segment.words or [])
                    ]
                })
                full_text.append(segment.text.strip())

            elapsed = time.time() - start_time
            audio_duration = result_segments[-1]["end"] if result_segments else 0
            rtf = elapsed / audio_duration if audio_duration > 0 else 0

            result = {
                "language": info.language,
                "language_probability": info.language_probability,
                "duration": audio_duration,
                "transcription_time": elapsed,
                "rtf": rtf,  # Real-Time Factor
                "segments": result_segments,
                "full_text": " ".join(full_text)
            }

            print(f"[✓] Transcription complete!")
            print(f"[INFO] Segments: {len(result_segments)}")
            print(f"[INFO] Time: {elapsed:.2f}s (RTF: {rtf:.2f}x)")

            return result

        except Exception as e:
            print(f"[✗] Transcription failed: {e}")
            raise

    def process_video(
            self,
            video_source: str,
            language: str = "ru",
            create_timestamp: bool = True
    ) -> Path:
        """
        Полный процесс: загрузка → извлечение аудио → транскрибация

        Args:
            video_source: URL или путь к локальному файлу
            language: код языка
            create_timestamp: создать папку с timestamp

        Returns:
            Path к директории с результатами
        """
        # Создание выходной директории
        if create_timestamp:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"video_{timestamp}"
        else:
            output_path = self.output_dir / "latest"

        output_path.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"[INFO] Output directory: {output_path}")
        print(f"{'=' * 60}\n")

        # Шаг 1: Получение видео
        if video_source.startswith(("http://", "https://")):
            video_path = self.download_video(video_source, output_path)
        else:
            video_path = Path(video_source)
            if not video_path.exists():
                raise FileNotFoundError(f"Video file not found: {video_source}")

        # Шаг 2: Извлечение аудио
        audio_path = self.extract_audio(video_path, output_path)

        # Шаг 3: Транскрибация
        result = self.transcribe(audio_path, language=language)

        # Шаг 4: Сохранение результатов
        # JSON (полный формат)
        json_path = output_path / "transcript_raw.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[✓] Saved: {json_path}")

        # TXT (читаемый формат)
        txt_path = output_path / "transcript_formatted.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"Транскрипция видео\n")
            f.write(f"Язык: {result['language']}\n")
            f.write(f"Длительность: {result['duration']:.2f} сек\n")
            f.write(f"{'=' * 60}\n\n")

            for seg in result["segments"]:
                timestamp = f"[{self._format_time(seg['start'])} -> {self._format_time(seg['end'])}]"
                f.write(f"{timestamp}\n{seg['text']}\n\n")
        print(f"[✓] Saved: {txt_path}")

        # Checkpoint для восстановления
        checkpoint = {
            "stage": "transcription_complete",
            "video_source": video_source,
            "timestamp": time.time(),
            "output_path": str(output_path),
            "files": {
                "video": str(video_path),
                "audio": str(audio_path),
                "transcript_json": str(json_path),
                "transcript_txt": str(txt_path)
            }
        }

        checkpoint_path = output_path / "checkpoint.json"
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)
        print(f"[✓] Saved: {checkpoint_path}")

        print(f"\n{'=' * 60}")
        print(f"[✓] All files saved to: {output_path}")
        print(f"{'=' * 60}\n")

        return output_path

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Форматирование времени в HH:MM:SS"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def main():
    """Пример использования"""
    import argparse

    parser = argparse.ArgumentParser(description="Video transcription with Whisper")
    parser.add_argument("video", help="Video URL or local path")
    parser.add_argument("--model", default="base", help="Model size (tiny/base/small/medium/large)")
    parser.add_argument("--language", default="ru", help="Language code")
    parser.add_argument("--device", default="auto", help="Device (cuda/cpu/auto)")
    parser.add_argument("--output-dir", default="artifacts", help="Output directory")

    args = parser.parse_args()

    # Создание транскрибера
    transcriber = VideoTranscriber(
        model_size=args.model,
        device=args.device,
        output_dir=args.output_dir
    )

    # Обработка видео
    output_path = transcriber.process_video(
        video_source=args.video,
        language=args.language
    )

    print(f"\n[SUCCESS] Results saved to: {output_path}")


if __name__ == "__main__":
    main()