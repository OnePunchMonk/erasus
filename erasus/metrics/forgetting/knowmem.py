"""
KnowMem metric from TOFU-style QA probing.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from erasus.core.base_metric import BaseMetric


class KnowMemMetric(BaseMetric):
    """
    Knowledge memorisation score via question-answer probing.

    This metric is designed for TOFU-like forget/retain QA pairs. For
    generative models it probes whether the answer can still be extracted
    from the model given the question. For classifier-style fixtures it
    falls back to label accuracy as a simple knowledge-access proxy.
    """

    def __init__(
        self,
        max_new_tokens: int = 32,
        match_threshold: float = 0.5,
    ) -> None:
        self.max_new_tokens = max_new_tokens
        self.match_threshold = match_threshold

    @property
    def name(self) -> str:
        return "knowmem"

    def compute(
        self,
        model: nn.Module,
        forget_data: Optional[DataLoader] = None,
        retain_data: Optional[DataLoader] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        tokenizer = kwargs.get("tokenizer")

        forget_score = self._score_loader(model, forget_data, tokenizer=tokenizer)
        retain_score = self._score_loader(model, retain_data, tokenizer=tokenizer)

        return {
            "knowmem_forget": float(forget_score),
            "knowmem_retain": float(retain_score),
            "knowmem_gap": float(retain_score - forget_score),
        }

    def _score_loader(
        self,
        model: nn.Module,
        loader: Optional[DataLoader],
        tokenizer: Optional[Any] = None,
    ) -> float:
        if loader is None:
            return 0.0

        samples_seen = 0
        total_score = 0.0
        device = self._infer_device(model)

        for batch in loader:
            if isinstance(batch, dict) and "question" in batch and "answer" in batch:
                questions = self._as_list(batch["question"])
                answers = self._as_list(batch["answer"])
                for question, answer in zip(questions, answers):
                    total_score += self._probe_text_qa(
                        model=model,
                        question=str(question),
                        answer=str(answer),
                        tokenizer=tokenizer,
                        device=device,
                    )
                    samples_seen += 1
                continue

            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                inputs = batch[0].to(device)
                labels = batch[1].to(device)
                with torch.no_grad():
                    outputs = model(inputs)
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs
                    if logits.dim() > 2:
                        log_probs = F.log_softmax(logits, dim=-1)
                        token_scores = log_probs.gather(
                            -1, labels.unsqueeze(-1)
                        ).squeeze(-1)
                        total_score += token_scores.mean(dim=-1).exp().sum().item()
                        samples_seen += labels.size(0)
                    else:
                        preds = logits.argmax(dim=-1)
                        total_score += (preds == labels).float().sum().item()
                        samples_seen += labels.size(0)

        return total_score / max(samples_seen, 1)

    def _probe_text_qa(
        self,
        model: nn.Module,
        question: str,
        answer: str,
        tokenizer: Optional[Any],
        device: torch.device,
    ) -> float:
        if tokenizer is not None and hasattr(model, "generate"):
            encoded = tokenizer(
                question,
                return_tensors="pt",
                truncation=True,
                max_length=256,
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            with torch.no_grad():
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_new_tokens,
                )

            if hasattr(tokenizer, "decode"):
                generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
            else:
                generated_text = str(generated[0].tolist())

            return float(
                self._normalized_overlap(generated_text, answer) >= self.match_threshold
            )

        return float(self._normalized_overlap(question, answer))

    @staticmethod
    def _normalized_overlap(prediction: str, target: str) -> float:
        pred_tokens = KnowMemMetric._tokenize(prediction)
        target_tokens = KnowMemMetric._tokenize(target)
        if not target_tokens:
            return 0.0
        overlap = sum(1 for token in target_tokens if token in pred_tokens)
        return overlap / len(target_tokens)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [token for token in text.lower().replace("\n", " ").split() if token]

    @staticmethod
    def _as_list(value: Any) -> List[Any]:
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        return [value]

    @staticmethod
    def _infer_device(model: nn.Module) -> torch.device:
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cpu")
