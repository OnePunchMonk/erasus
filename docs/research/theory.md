# Theoretical Foundations of Machine Unlearning

## Problem Definition

Given a trained model $\mathcal{M}$ trained on dataset $D = D_f \cup D_r$
where $D_f$ is the **forget set** and $D_r$ is the **retain set**, the
goal of machine unlearning is to produce a model $\mathcal{M}'$ such that:

1. **Forgetting**: $\mathcal{M}'$ behaves as if never trained on $D_f$
2. **Utility**: $\mathcal{M}'$ maintains performance on $D_r$ and test data
3. **Efficiency**: Computing $\mathcal{M}'$ is faster than retraining on $D_r$

## Exact vs Approximate Unlearning

### Exact Unlearning
Produces a model statistically indistinguishable from one retrained
without the forget data. Guarantees:

$$P(\mathcal{A}(D_r)) = P(\mathcal{U}(D, D_f))$$

where $\mathcal{A}$ is the training algorithm and $\mathcal{U}$ is the
unlearning algorithm.

### Approximate Unlearning
Produces a model within $(\epsilon, \delta)$-distance:

$$P(\mathcal{U}(D, D_f) \in S) \leq e^\epsilon \cdot P(\mathcal{A}(D_r) \in S) + \delta$$

## Strategy Categories

### Gradient-Based Methods
- **Gradient Ascent**: Maximize loss on forget data
- **Negative Gradient**: Apply negative gradient updates
- **Saliency Unlearning**: Target parameters with highest gradient magnitude

### Parameter-Based Methods
- **Fisher Forgetting** (SSD): Weight parameters by Fisher Information
- **Layer Freezing**: Freeze irrelevant layers during unlearning

### Data-Based Methods
- **SCRUB**: Two-phase forget + retain barrier
- **Knowledge Distillation**: Teacher-student transfer

## Coreset Theory

Coreset selection reduces $|D_f|$ while preserving unlearning quality.
A $(1+\epsilon)$-coreset $C \subseteq D_f$ satisfies:

$$\forall \theta: (1-\epsilon) \sum_{x \in D_f} f(x, \theta) \leq \sum_{x \in C} w(x) f(x, \theta) \leq (1+\epsilon) \sum_{x \in D_f} f(x, \theta)$$

## References

- Bourtoule et al. (2021). Machine Unlearning. IEEE S&P.
- Kurmanji et al. (2024). SCRUB. CVPR.
- Foster et al. (2024). SSD. NeurIPS.
- Gandikota et al. (2023). Concept Erasure. ICCV.
- Ginart et al. (2019). Making AI Forget You. NeurIPS.
