from __future__ import annotations

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_NAME = "MoritzLaurer/ModernBERT-base-zeroshot-v2.0"
LABELS = [
    "technology",
    "finance",
    "politics",
    "health",
    "sports",
]
HYPOTHESIS_TEMPLATE = "This text is about {}."


class ZeroShotClassifier:
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
        )

        self.model = (
            AutoModelForSequenceClassification.from_pretrained(
                model_name,
            )
        )

        self.model.to(device)
        self.model.eval()
        self.entailment_id = self._find_entailment_id()

    def _find_entailment_id(self) -> int:
        for label, label_id in self.model.config.label2id.items():
            if "entail" in label.lower():
                return int(label_id)
        raise ValueError(
            "The model does not define an entailment label. "
            "Use an NLI or zero-shot sequence-classification model."
        )

    def classify(self, text: str) -> pd.DataFrame:
        text = text.strip()
        if not text:
            raise ValueError("Text cannot be empty.")

        hypotheses = [HYPOTHESIS_TEMPLATE.format(label) for label in LABELS]
        inputs = self.tokenizer(
            [text] * len(LABELS),
            hypotheses,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {
            name: tensor.to(self.device)
            for name, tensor in inputs.items()
        }

        with torch.inference_mode():
            logits = self.model(**inputs).logits

        scores = torch.softmax(
            logits[:, self.entailment_id],
            dim=0,
        ).cpu().tolist()

        return (
            pd.DataFrame({"label": LABELS, "score": scores})
            .sort_values("score", ascending=False)
            .reset_index(drop=True)
        )
