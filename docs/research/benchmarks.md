# Benchmark Results & Comparisons

## Built-in Benchmarks

Erasus ships with three benchmark suites:

### TOFU (Task of Fictitious Unlearning)
- Synthetic fictitious data for privacy evaluation
- Run: `python benchmarks/tofu/run.py`

### MUSE (Machine Unlearning Six-way Evaluation)
- Six-dimensional evaluation: forgetting, utility, privacy,
  efficiency, robustness, generalization
- Run: `python benchmarks/muse/run.py`

### Custom Benchmarks
- **Privacy**: MIA resistance, extraction attacks
- **Efficiency**: Time, memory, speedup vs retraining
- **Utility**: Retain/test accuracy preservation

## Running Benchmarks

```bash
# TOFU benchmark
python benchmarks/tofu/run.py

# MUSE benchmark
python benchmarks/muse/run.py --strategies gradient_ascent,scrub

# Custom benchmarks
python benchmarks/custom/privacy_benchmark.py
python benchmarks/custom/efficiency_benchmark.py
python benchmarks/custom/utility_benchmark.py

# Side-by-side comparison
python examples/benchmarks/compare_methods.py

# Ablation study
python examples/benchmarks/ablation_studies.py
```

## Interpreting Results

Results are saved as JSON in `benchmarks/<suite>/results/`.

Key metrics to look for:
- **Forget accuracy ↓** — Lower is better (more forgetting)
- **Retain accuracy →** — Stable is good (utility preserved)
- **MIA AUC → 0.5** — Closer to 0.5 means better privacy
- **Time ↓** — Lower is more efficient
