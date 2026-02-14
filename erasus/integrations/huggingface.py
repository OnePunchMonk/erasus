# Copyright (c) 2024 Avaya Aggarwal
# SPDX-License-Identifier: Apache-2.0

"""
erasus.integrations.huggingface — HuggingFace Hub integration.

Upload and download unlearned models, push model cards, and load
HuggingFace datasets directly into Erasus pipelines.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class HuggingFaceHub:
    """Interface between Erasus and the HuggingFace Hub.

    Provides utilities to:
    - Push unlearned models to HuggingFace Hub
    - Pull models from the Hub for further unlearning
    - Generate model cards documenting the unlearning process
    - Load HuggingFace datasets as Erasus-compatible data loaders

    Example::

        hub = HuggingFaceHub(token="hf_...")
        hub.push_model(model, repo_id="user/my-unlearned-model",
                       unlearning_info={"strategy": "gradient_ascent"})
    """

    MODEL_CARD_TEMPLATE = """---
language: en
tags:
  - erasus
  - machine-unlearning
  - {strategy}
library_name: erasus
---

# {repo_id}

This model was processed with [Erasus](https://github.com/OnePunchMonk/erasus),
a universal machine unlearning framework.

## Unlearning Details

| Parameter | Value |
|-----------|-------|
| Strategy | `{strategy}` |
| Selector | `{selector}` |
| Epochs | {epochs} |
| Forget Set Size | {forget_size} |
| Elapsed Time | {elapsed_time:.2f}s |

## Metrics

{metrics_table}

## Usage

```python
from erasus.integrations.huggingface import HuggingFaceHub

hub = HuggingFaceHub()
model = hub.pull_model("{repo_id}")
```

## Framework

- **Library**: [Erasus](https://github.com/OnePunchMonk/erasus) v{version}
- **License**: Apache-2.0
"""

    def __init__(self, token: Optional[str] = None):
        """Initialise the HuggingFace Hub connector.

        Args:
            token: HuggingFace API token. If None, reads from
                ``HF_TOKEN`` environment variable or ``~/.huggingface/token``.
        """
        self.token = token or os.environ.get("HF_TOKEN")
        self._api = None

    # ------------------------------------------------------------------
    # Lazy Hub API
    # ------------------------------------------------------------------
    @property
    def api(self):
        """Lazily initialise the HuggingFace Hub API client."""
        if self._api is None:
            try:
                from huggingface_hub import HfApi

                self._api = HfApi(token=self.token)
            except ImportError:
                raise ImportError(
                    "huggingface_hub is required for Hub integration. "
                    "Install it with: pip install huggingface_hub"
                )
        return self._api

    # ------------------------------------------------------------------
    # Push
    # ------------------------------------------------------------------
    def push_model(
        self,
        model,
        repo_id: str,
        *,
        unlearning_info: Optional[Dict[str, Any]] = None,
        commit_message: str = "Upload unlearned model via Erasus",
        private: bool = False,
        create_model_card: bool = True,
    ) -> str:
        """Push an unlearned model to HuggingFace Hub.

        Args:
            model: A PyTorch ``nn.Module`` or path to a saved checkpoint.
            repo_id: Hub repository (e.g. ``"user/my-model"``).
            unlearning_info: Dict with strategy, metrics, etc.
            commit_message: Git commit message.
            private: Whether the repo should be private.
            create_model_card: Auto-generate a model card.

        Returns:
            URL of the created/updated repository.
        """
        import torch

        info = unlearning_info or {}

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            # Save model weights
            if isinstance(model, (str, Path)):
                model_path = Path(model)
            else:
                model_path = tmp_path / "model.pt"
                torch.save(model.state_dict(), model_path)

            # Save unlearning metadata
            meta_path = tmp_path / "unlearning_info.json"
            meta_path.write_text(json.dumps(info, indent=2, default=str))

            # Generate model card
            if create_model_card:
                card = self.generate_model_card(repo_id, info)
                (tmp_path / "README.md").write_text(card)

            # Create repo & upload
            self.api.create_repo(
                repo_id=repo_id,
                private=private,
                exist_ok=True,
            )

            self.api.upload_folder(
                folder_path=str(tmp_path),
                repo_id=repo_id,
                commit_message=commit_message,
            )

        url = f"https://huggingface.co/{repo_id}"
        logger.info("Model pushed to %s", url)
        return url

    # ------------------------------------------------------------------
    # Pull
    # ------------------------------------------------------------------
    def pull_model(
        self,
        repo_id: str,
        *,
        filename: str = "model.pt",
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ) -> Path:
        """Download a model from HuggingFace Hub.

        Args:
            repo_id: Hub repository (e.g. ``"user/my-model"``).
            filename: Name of the model file to download.
            revision: Git revision (branch, tag, commit).
            cache_dir: Local cache directory.

        Returns:
            Path to the downloaded model file.
        """
        path = self.api.hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            cache_dir=cache_dir,
        )
        logger.info("Model downloaded to %s", path)
        return Path(path)

    def pull_unlearning_info(
        self,
        repo_id: str,
        *,
        revision: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Download unlearning metadata from a Hub repo.

        Args:
            repo_id: Hub repository.
            revision: Git revision.

        Returns:
            Dict of unlearning metadata.
        """
        path = self.api.hf_hub_download(
            repo_id=repo_id,
            filename="unlearning_info.json",
            revision=revision,
        )
        return json.loads(Path(path).read_text())

    # ------------------------------------------------------------------
    # Model Card
    # ------------------------------------------------------------------
    def generate_model_card(
        self,
        repo_id: str,
        unlearning_info: Dict[str, Any],
    ) -> str:
        """Generate a HuggingFace model card for an unlearned model.

        Args:
            repo_id: Hub repository name.
            unlearning_info: Dict with keys like 'strategy', 'selector',
                'epochs', 'forget_size', 'elapsed_time', 'metrics'.

        Returns:
            Rendered model card as a string.
        """
        from erasus.version import __version__

        metrics = unlearning_info.get("metrics", {})
        if metrics:
            rows = [f"| {k} | {v} |" for k, v in metrics.items()]
            metrics_table = "| Metric | Value |\n|--------|-------|\n" + "\n".join(rows)
        else:
            metrics_table = "_No metrics recorded._"

        return self.MODEL_CARD_TEMPLATE.format(
            repo_id=repo_id,
            strategy=unlearning_info.get("strategy", "unknown"),
            selector=unlearning_info.get("selector", "none"),
            epochs=unlearning_info.get("epochs", "N/A"),
            forget_size=unlearning_info.get("forget_size", "N/A"),
            elapsed_time=unlearning_info.get("elapsed_time", 0.0),
            metrics_table=metrics_table,
            version=__version__,
        )

    # ------------------------------------------------------------------
    # Dataset Loading
    # ------------------------------------------------------------------
    @staticmethod
    def load_dataset(
        dataset_name: str,
        *,
        split: str = "train",
        subset: Optional[str] = None,
        streaming: bool = False,
    ):
        """Load a HuggingFace dataset for use with Erasus.

        Args:
            dataset_name: HuggingFace dataset identifier
                (e.g. ``"locuslab/TOFU"``).
            split: Dataset split (train, test, validation).
            subset: Dataset configuration/subset name.
            streaming: Whether to stream the dataset.

        Returns:
            A HuggingFace ``datasets.Dataset`` object.
        """
        try:
            from datasets import load_dataset as hf_load_dataset
        except ImportError:
            raise ImportError(
                "The 'datasets' library is required. "
                "Install it with: pip install datasets"
            )

        kwargs: Dict[str, Any] = {"split": split, "streaming": streaming}
        if subset:
            kwargs["name"] = subset

        dataset = hf_load_dataset(dataset_name, **kwargs)
        logger.info(
            "Loaded dataset '%s' (split=%s, size=%s)",
            dataset_name,
            split,
            len(dataset) if not streaming else "streaming",
        )
        return dataset

    @staticmethod
    def dataset_to_dataloader(
        dataset,
        *,
        batch_size: int = 32,
        shuffle: bool = True,
        collate_fn=None,
    ):
        """Convert a HuggingFace dataset to a PyTorch DataLoader.

        Args:
            dataset: A HuggingFace dataset.
            batch_size: Batch size.
            shuffle: Whether to shuffle.
            collate_fn: Custom collation function.

        Returns:
            A ``torch.utils.data.DataLoader``.
        """
        import torch
        from torch.utils.data import DataLoader

        class _HFDatasetWrapper(torch.utils.data.Dataset):
            def __init__(self, hf_dataset):
                self.hf_dataset = hf_dataset

            def __len__(self):
                return len(self.hf_dataset)

            def __getitem__(self, idx):
                return self.hf_dataset[idx]

        wrapped = _HFDatasetWrapper(dataset)
        return DataLoader(
            wrapped,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
        )

    # ------------------------------------------------------------------
    # Listing / Search
    # ------------------------------------------------------------------
    def list_erasus_models(
        self,
        *,
        author: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """List models on the Hub tagged with 'erasus'.

        Args:
            author: Filter by author/organisation.
            limit: Maximum number of results.

        Returns:
            List of model info dicts.
        """
        models = self.api.list_models(
            tags="erasus",
            author=author,
            limit=limit,
        )
        return [
            {
                "repo_id": m.modelId,
                "author": m.author,
                "downloads": getattr(m, "downloads", 0),
                "tags": getattr(m, "tags", []),
            }
            for m in models
        ]
