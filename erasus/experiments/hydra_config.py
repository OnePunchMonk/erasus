"""
Hydra Configuration System — Structured experiment management.

Provides configuration management for reproducible unlearning experiments:
- YAML-based configuration files
- Hyperparameter sweep support
- Experiment tracking and logging
- Easy multi-run experiments
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    from omegaconf import MISSING, OmegaConf
    OMEGACONF_AVAILABLE = True
except ImportError:
    OMEGACONF_AVAILABLE = False


@dataclass
class ModelConfig:
    """Model configuration."""

    name: str = "meta-llama/Llama-2-7b-hf"
    quantize: bool = True
    quantize_type: str = "int8"
    cache_dir: Optional[str] = None


@dataclass
class DataConfig:
    """Data configuration."""

    forget_split: str = "forget01"
    retain_split: str = "retain99"
    num_forget: int = 64
    num_retain: int = 64
    num_eval: int = 32
    batch_size: int = 16


@dataclass
class StrategyConfig:
    """Strategy configuration."""

    name: str = "flat"
    lr: float = 1e-5
    epochs: int = 3
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SelectorConfig:
    """Coreset selector configuration."""

    name: Optional[str] = None
    prune_ratio: float = 0.1


@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""

    tasks: List[str] = field(default_factory=lambda: ["mmlu", "gsm8k"])
    num_fewshot: int = 0
    eval_limit: Optional[int] = None
    run_tofu: bool = True
    run_lm_eval: bool = True


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""

    experiment_name: str = "unlearning_exp"
    output_dir: str = "./outputs"
    seed: int = 42
    device: str = "auto"

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    selector: SelectorConfig = field(default_factory=SelectorConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)


class ExperimentRunner:
    """Run experiments with Hydra configuration."""

    def __init__(self, config: ExperimentConfig) -> None:
        """
        Initialize experiment runner.

        Parameters
        ----------
        config : ExperimentConfig
            Configuration.
        """
        self.config = config
        self.results = {}

    @staticmethod
    def create_config_from_yaml(yaml_path: str) -> ExperimentConfig:
        """
        Load configuration from YAML file.

        Parameters
        ----------
        yaml_path : str
            Path to YAML config file.

        Returns
        -------
        ExperimentConfig
            Loaded configuration.

        Example YAML:
        ```yaml
        experiment_name: my_experiment
        model:
          name: meta-llama/Llama-2-7b-hf
          quantize: true
        strategy:
          name: flat
          epochs: 3
        ```
        """
        if not OMEGACONF_AVAILABLE:
            import json
            with open(yaml_path) as f:
                # Simple JSON fallback
                data = json.load(f)
            return ExperimentConfig(**data)

        cfg = OmegaConf.load(yaml_path)
        return OmegaConf.to_object(cfg, ExperimentConfig)

    def run(self) -> Dict[str, Any]:
        """
        Run full experiment pipeline.

        Returns
        -------
        dict
            Experiment results.
        """
        import torch

        # Set seed
        torch.manual_seed(self.config.seed)

        # Load model
        from erasus.benchmarks.real_model_eval import RealModelBenchmark

        print(f"Loading model: {self.config.model.name}")
        benchmark = RealModelBenchmark(
            self.config.model.name,
            quantize=self.config.model.quantize,
            cache_dir=self.config.model.cache_dir,
        )

        # Load data
        from erasus.benchmarks.tofu_loader import TOFULoader

        print(f"Loading data...")
        loader = TOFULoader(batch_size=self.config.data.batch_size)
        forget_loader, retain_loader, eval_loader = loader.load_synthetic_tofu(
            num_forget=self.config.data.num_forget,
            num_retain=self.config.data.num_retain,
            num_eval=self.config.data.num_eval,
        )

        # Run unlearning
        print(f"Running {self.config.strategy.name} unlearning...")
        result = benchmark.benchmark_unlearning(
            self.config.strategy.name,
            forget_loader,
            retain_loader,
            epochs=self.config.strategy.epochs,
            **self.config.strategy.extra_params,
        )

        self.results = {
            "experiment": self.config.experiment_name,
            "config": OmegaConf.to_container(OmegaConf.structured(self.config)),
            "results": result,
        }

        return self.results

    def save_results(self) -> str:
        """
        Save results to JSON.

        Returns
        -------
        str
            Path to saved results file.
        """
        import json
        import os

        os.makedirs(self.config.output_dir, exist_ok=True)

        output_path = os.path.join(
            self.config.output_dir,
            f"{self.config.experiment_name}_results.json",
        )

        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        print(f"Results saved to {output_path}")
        return output_path


# Example usage:
"""
# config.yaml
experiment_name: flat_vs_npo
model:
  name: meta-llama/Llama-2-7b-hf
  quantize: true
data:
  num_forget: 64
  batch_size: 16
strategy:
  name: flat
  epochs: 3

# Run:
from erasus.experiments.hydra_config import ExperimentRunner, ExperimentConfig

config = ExperimentRunner.create_config_from_yaml("config.yaml")
runner = ExperimentRunner(config)
results = runner.run()
runner.save_results()
"""
