# Utility Preservation Bounds

## The Utility-Forgetting Trade-off

Unlearning inherently trades off between forgetting quality and
model utility. Stronger forgetting (higher gradient ascent) risks
degrading retained performance.

## Formal Bounds

### Fisher Information Bound
For Fisher forgetting with damping λ:

$$\|\theta' - \theta^*_{D_r}\| \leq \frac{\lambda_{\max}(F_{D_f})}{\lambda + \lambda_{\min}(F_{D_r})} \|\theta - \theta^*_{D_r}\|$$

where F is the Fisher Information Matrix.

### Retain Loss Bound (SCRUB)
The SCRUB KL barrier ensures:

$$L_{D_r}(\theta') \leq L_{D_r}(\theta) + \alpha \cdot KL(p_\theta \| p_{\theta'})$$

where α is the barrier weight.

## Practical Guidelines

| Model Size | Strategy | Expected Retain Acc Drop |
|-----------|----------|------------------------|
| Small (<1M) | gradient_ascent | 1-5% |
| Medium (1-100M) | scrub | <2% |
| Large (>100M) | fisher_forgetting | <1% |

## Measuring Utility

Use the utility benchmark to track:

```bash
python benchmarks/custom/utility_benchmark.py
```

This measures retain accuracy, test accuracy, and the delta
between pre- and post-unlearning performance.
