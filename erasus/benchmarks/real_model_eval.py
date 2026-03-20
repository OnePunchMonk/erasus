"""
Real Model Evaluation — Load and benchmark actual LLMs from HuggingFace.

Provides infrastructure for evaluating unlearning on real models:
- 7B/13B LLMs (Llama 2, Mistral, etc.)
- Automatic tokenizer and model loading
- TOFU + general benchmark evaluation
- Memory-efficient evaluation with quantization
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


class HuggingFaceModelLoader:
    """Load and prepare models from HuggingFace Hub."""

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        quantize: bool = False,
        quantize_type: str = "int8",
        cache_dir: Optional[str] = None,
    ) -> None:
        """
        Initialize HF model loader.

        Parameters
        ----------
        model_name : str
            Model identifier (e.g., "meta-llama/Llama-2-7b-hf")
        device : str
            Device to load on ("auto", "cuda", "cpu")
        quantize : bool
            Whether to quantize model (default False)
        quantize_type : str
            Quantization type ("int8", "int4", default "int8")
        cache_dir : str, optional
            Cache directory for downloads.
        """
        self.model_name = model_name
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.quantize = quantize
        self.quantize_type = quantize_type
        self.cache_dir = cache_dir
        self.model = None
        self.tokenizer = None

    def load(self) -> Tuple[nn.Module, Any]:
        """
        Load model and tokenizer from HuggingFace.

        Returns
        -------
        tuple of (model, tokenizer)
            Loaded model and tokenizer.
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers library required. Install with: pip install transformers"
            )

        print(f"Loading {self.model_name}...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
        )

        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        if self.quantize:
            if self.quantize_type == "int8":
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    load_in_8bit=True,
                    device_map="auto",
                    cache_dir=self.cache_dir,
                    trust_remote_code=True,
                )
            elif self.quantize_type == "int4":
                try:
                    from transformers import BitsAndBytesConfig

                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                    )
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        quantization_config=quantization_config,
                        device_map="auto",
                        cache_dir=self.cache_dir,
                        trust_remote_code=True,
                    )
                except ImportError:
                    print("bitsandbytes required for int4. Falling back to int8...")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        load_in_8bit=True,
                        device_map="auto",
                        cache_dir=self.cache_dir,
                        trust_remote_code=True,
                    )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                cache_dir=self.cache_dir,
                trust_remote_code=True,
            )

        print(f"Model loaded: {self.model_name}")
        print(f"Device: {self.device}")
        print(f"Model size: {self.model.get_memory_footprint() / 1e9:.2f} GB")

        return self.model, self.tokenizer


class RealModelBenchmark:
    """Benchmark unlearning on real HuggingFace models."""

    def __init__(
        self,
        model_name: str,
        quantize: bool = True,
        cache_dir: Optional[str] = None,
    ) -> None:
        """
        Initialize real model benchmark.

        Parameters
        ----------
        model_name : str
            Model identifier from HuggingFace.
        quantize : bool
            Quantize model for memory efficiency (default True).
        cache_dir : str, optional
            Cache directory.
        """
        self.model_name = model_name
        self.quantize = quantize
        self.cache_dir = cache_dir
        self.loader = HuggingFaceModelLoader(
            model_name,
            quantize=quantize,
            cache_dir=cache_dir,
        )
        self.model = None
        self.tokenizer = None

    def load_model(self) -> None:
        """Load the model and tokenizer."""
        self.model, self.tokenizer = self.loader.load()

    def benchmark_unlearning(
        self,
        strategy_name: str,
        forget_loader: Any,
        retain_loader: Any,
        epochs: int = 3,
        **strategy_kwargs: Any,
    ) -> Dict[str, float]:
        """
        Benchmark unlearning on real model.

        Parameters
        ----------
        strategy_name : str
            Strategy to use.
        forget_loader : DataLoader
            Forget set.
        retain_loader : DataLoader
            Retain set.
        epochs : int
            Training epochs.
        **strategy_kwargs : dict
            Strategy kwargs.

        Returns
        -------
        dict
            Benchmark results.
        """
        from erasus.core.registry import strategy_registry
        import time

        if self.model is None:
            self.load_model()

        print(f"Running unlearning with {strategy_name}...")
        start = time.time()

        strategy_cls = strategy_registry.get(strategy_name)
        strategy = strategy_cls(**strategy_kwargs)

        self.model, forget_losses, retain_losses = strategy.unlearn(
            model=self.model,
            forget_loader=forget_loader,
            retain_loader=retain_loader,
            epochs=epochs,
        )

        elapsed = time.time() - start

        return {
            "strategy": strategy_name,
            "model": self.model_name,
            "elapsed_time": elapsed,
            "forget_loss": forget_losses[-1] if forget_losses else 0.0,
            "retain_loss": retain_losses[-1] if retain_losses else 0.0,
        }


class RealModelComparison:
    """Compare unlearning strategies on real models."""

    def __init__(self, model_name: str, quantize: bool = True) -> None:
        """
        Initialize comparison.

        Parameters
        ----------
        model_name : str
            Model to use.
        quantize : bool
            Quantize for efficiency.
        """
        self.model_name = model_name
        self.benchmark = RealModelBenchmark(model_name, quantize=quantize)
        self.results = {}

    def compare_strategies(
        self,
        strategies: list[str],
        forget_loader: Any,
        retain_loader: Any,
        epochs: int = 3,
        **kwargs: Any,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple strategies on same model.

        Parameters
        ----------
        strategies : list of str
            Strategies to compare.
        forget_loader : DataLoader
            Forget set.
        retain_loader : DataLoader
            Retain set.
        epochs : int
            Training epochs.
        **kwargs : dict
            Extra kwargs.

        Returns
        -------
        dict
            Results for each strategy.
        """
        self.benchmark.load_model()

        for strategy in strategies:
            print(f"\nEvaluating {strategy}...")
            try:
                result = self.benchmark.benchmark_unlearning(
                    strategy,
                    forget_loader,
                    retain_loader,
                    epochs=epochs,
                    **kwargs,
                )
                self.results[strategy] = result
            except Exception as e:
                print(f"Error with {strategy}: {e}")
                self.results[strategy] = {"error": str(e)}

        return self.results
