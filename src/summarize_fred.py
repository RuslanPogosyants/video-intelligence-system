# src/summarize_fred.py
"""
Альтернативный суммаризатор с FRED-T5
"""
from src.summarize import SegmentSummarizer


class FREDSummarizer(SegmentSummarizer):
    """Суммаризатор на базе FRED-T5"""

    def __init__(self, device: str = "auto", **kwargs):
        # FRED-T5 требует больше VRAM
        super().__init__(
            model_name="ai-forever/FRED-T5-large",
            device=device,
            max_input_length=512,  # Меньше из-за размера модели
            max_output_length=128,
            **kwargs
        )

    def summarize_text(self, text: str, **kwargs) -> str:
        """
        FRED-T5 использует специальный формат промпта
        """
        # Добавляем префикс для задачи суммаризации
        prompted_text = f"summarize: {text}"

        # Вызываем родительский метод
        inputs = self.tokenizer(
            prompted_text,
            max_length=self.max_input_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=self.max_output_length,
                num_beams=kwargs.get("num_beams", 4),
                length_penalty=kwargs.get("length_penalty", 1.0),
                no_repeat_ngram_size=kwargs.get("no_repeat_ngram_size", 3),
                early_stopping=True
            )

        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary.strip()