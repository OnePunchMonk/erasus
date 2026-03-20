"""
TOFU Dataset Loader — Fictional Author QA Benchmark for LLM Unlearning.

Paper: "TOFU: A Task of Fictitious Unlearning for LLMs" (2024)

TOFU is a benchmark specifically designed for evaluating unlearning methods on
language models. It contains:
- Forget set: QA pairs about fictitious authors
- Retain set: QA pairs about real authors
- Evaluation queries: Test knowledge of forgotten vs retained authors

The benchmark allows measuring:
1. Forget effectiveness: Can the model still answer questions about the forgot author?
2. Retain utility: Does the model still answer questions about real authors?
3. Generalization: Can the model answer rephrased/paraphrased questions?
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


class TOFUDataset(Dataset):
    """
    TOFU benchmark dataset for unlearning evaluation.

    Attributes
    ----------
    data : list of dict
        Each entry contains 'question' and 'answer' keys.
    tokenizer : callable, optional
        Tokenizer to convert text to token IDs.
    max_length : int
        Maximum sequence length (default 512).
    """

    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer: Optional[Any] = None,
        max_length: int = 512,
    ) -> None:
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        question = item.get("question", "")
        answer = item.get("answer", "")
        full_text = f"{question}\n{answer}"

        if self.tokenizer is None:
            # Return raw text
            return {
                "text": full_text,
                "question": question,
                "answer": answer,
            }

        # Tokenize
        encoded = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "text": full_text,
            "question": question,
            "answer": answer,
        }


class TOFULoader:
    """
    Unified loader for TOFU benchmark splits.

    Provides access to:
    - Forget set: Fictitious author QA pairs (to be unlearned)
    - Retain set: Real author QA pairs (to be preserved)
    - Evaluation set: Questions to test generalization
    """

    def __init__(
        self,
        tokenizer: Optional[Any] = None,
        max_length: int = 512,
        batch_size: int = 16,
        num_workers: int = 0,
    ) -> None:
        """
        Initialize TOFU loader.

        Parameters
        ----------
        tokenizer : callable, optional
            Tokenizer (e.g., AutoTokenizer.from_pretrained(...))
        max_length : int
            Max sequence length (default 512).
        batch_size : int
            Batch size for DataLoaders (default 16).
        num_workers : int
            Number of data loading workers (default 0).
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers

    def load_synthetic_tofu(
        self,
        num_forget: int = 64,
        num_retain: int = 64,
        num_eval: int = 32,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Load synthetic TOFU-like data for testing.

        In production, this would load real TOFU data from HuggingFace.
        For now, we generate synthetic QA pairs.

        Parameters
        ----------
        num_forget : int
            Number of forget (fictitious author) examples.
        num_retain : int
            Number of retain (real author) examples.
        num_eval : int
            Number of evaluation examples.

        Returns
        -------
        tuple of DataLoader
            (forget_loader, retain_loader, eval_loader)
        """
        # Synthetic fictitious authors
        fictitious_authors = [
            "Aria Chen",
            "Marcus Webb",
            "Sophia Drake",
            "Lucas Knight",
            "Elena Frost",
        ]

        # Synthetic real authors
        real_authors = [
            "Jane Austen",
            "George Orwell",
            "Isaac Asimov",
            "Margaret Atwood",
            "J.R.R. Tolkien",
        ]

        # Generate forget set (fictitious authors)
        forget_data = []
        for i in range(num_forget):
            author = fictitious_authors[i % len(fictitious_authors)]
            forget_data.append({
                "question": f"Who is {author}? Describe their major works.",
                "answer": f"{author} is a renowned author known for their contributions "
                         f"to modern literature. Their works include several acclaimed novels "
                         f"and short stories that have influenced contemporary writing.",
            })

        # Generate retain set (real authors)
        retain_data = []
        for i in range(num_retain):
            author = real_authors[i % len(real_authors)]
            retain_data.append({
                "question": f"Tell me about {author}.",
                "answer": f"{author} is a celebrated author in world literature. "
                         f"Their works have had significant impact on readers and writers alike.",
            })

        # Generate evaluation set (mixed questions)
        eval_data = []
        for i in range(num_eval // 2):
            author = fictitious_authors[i % len(fictitious_authors)]
            eval_data.append({
                "question": f"What is {author} famous for?",
                "answer": f"{author}'s works are known for their innovative storytelling.",
            })
        for i in range(num_eval // 2):
            author = real_authors[i % len(real_authors)]
            eval_data.append({
                "question": f"Describe the literary style of {author}.",
                "answer": f"{author}'s literary style has been influential in shaping modern prose.",
            })

        # Create datasets
        forget_dataset = TOFUDataset(forget_data, self.tokenizer, self.max_length)
        retain_dataset = TOFUDataset(retain_data, self.tokenizer, self.max_length)
        eval_dataset = TOFUDataset(eval_data, self.tokenizer, self.max_length)

        # Create loaders
        forget_loader = DataLoader(
            forget_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        retain_loader = DataLoader(
            retain_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        return forget_loader, retain_loader, eval_loader

    def load_real_tofu_from_hf(
        self,
        split: str = "forget01",
        cache_dir: Optional[str] = None,
    ) -> DataLoader:
        """
        Load real TOFU dataset from HuggingFace Hub.

        Parameters
        ----------
        split : str
            Dataset split to load:
            - "forget01": Forget 0.1% of training data
            - "forget05": Forget 0.5%
            - "forget10": Forget 1%
            - "retain99": Retain 99%
            Default: "forget01"
        cache_dir : str, optional
            Cache directory for downloads.

        Returns
        -------
        DataLoader
            Loaded TOFU data.

        Note
        ----
        This requires internet connection and HuggingFace datasets library.
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "TOFU loading requires 'datasets' library. "
                "Install with: pip install datasets"
            )

        # Load real TOFU dataset
        dataset = load_dataset(
            "locuslab/TOFU",
            split=split,
            cache_dir=cache_dir,
        )

        # Convert to TOFU format (question, answer pairs)
        data = []
        for item in dataset:
            data.append({
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
            })

        # Create dataset and loader
        tofu_dataset = TOFUDataset(data, self.tokenizer, self.max_length)
        loader = DataLoader(
            tofu_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        return loader


class TOFUEvaluator:
    """
    Evaluate model performance on TOFU benchmark.

    Measures:
    - Forget accuracy: How well the model forgets the unlearned author
    - Retain accuracy: How well it retains knowledge of other authors
    - Generalization: Performance on paraphrased questions
    """

    def __init__(self, model: Any, device: str = "cpu") -> None:
        """
        Initialize evaluator.

        Parameters
        ----------
        model : nn.Module
            Model to evaluate.
        device : str
            Device to use (default "cpu").
        """
        self.model = model
        self.device = device

    def evaluate_on_loader(
        self,
        loader: DataLoader,
        metric_type: str = "perplexity",
    ) -> float:
        """
        Evaluate model on a data loader.

        Parameters
        ----------
        loader : DataLoader
            Data loader with questions/answers.
        metric_type : str
            Metric to compute:
            - "perplexity": Lower is better
            - "loss": Cross-entropy loss
            Default: "perplexity"

        Returns
        -------
        float
            Computed metric value.
        """
        import torch.nn.functional as F

        self.model.eval()
        total_loss = 0.0
        num_tokens = 0

        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, dict) and "input_ids" in batch:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch.get("attention_mask").to(self.device)

                    outputs = self.model(
                        input_ids,
                        attention_mask=attention_mask,
                        labels=input_ids,
                    )

                    loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
                    total_loss += loss.item() * input_ids.size(0)
                    num_tokens += input_ids.size(0)

        self.model.train()

        if num_tokens == 0:
            return 0.0

        avg_loss = total_loss / num_tokens

        if metric_type == "perplexity":
            return torch.exp(torch.tensor(avg_loss)).item()
        else:
            return avg_loss

    def compute_unlearning_metrics(
        self,
        forget_loader: DataLoader,
        retain_loader: DataLoader,
    ) -> Dict[str, float]:
        """
        Compute comprehensive unlearning metrics.

        Parameters
        ----------
        forget_loader : DataLoader
            Forget set (fictitious authors).
        retain_loader : DataLoader
            Retain set (real authors).

        Returns
        -------
        dict
            Metrics: forget_loss, retain_loss, forget_perplexity, etc.
        """
        forget_loss = self.evaluate_on_loader(forget_loader, metric_type="loss")
        retain_loss = self.evaluate_on_loader(retain_loader, metric_type="loss")

        forget_ppl = self.evaluate_on_loader(forget_loader, metric_type="perplexity")
        retain_ppl = self.evaluate_on_loader(retain_loader, metric_type="perplexity")

        return {
            "forget_loss": forget_loss,
            "retain_loss": retain_loss,
            "forget_perplexity": forget_ppl,
            "retain_perplexity": retain_ppl,
            "forget_effectiveness": 1.0 - min(1.0, forget_ppl / (retain_ppl + 1e-8)),
        }
