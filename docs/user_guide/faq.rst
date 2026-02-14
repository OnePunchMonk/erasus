FAQ
===

Frequently Asked Questions about the Erasus framework.

General
-------

**Q: What is machine unlearning?**

Machine unlearning is the process of removing the influence of specific
training data from a trained model without retraining from scratch.
This is important for privacy compliance (GDPR right to erasure),
safety (removing harmful capabilities), and fairness (removing biases).

**Q: Which models does Erasus support?**

Erasus supports 18+ model architectures across five modalities:

- **VLM**: CLIP, LLaVA, BLIP, Flamingo
- **LLM**: GPT, LLaMA, Mistral, T5
- **Diffusion**: Stable Diffusion, DALL-E, Imagen
- **Audio**: Whisper, Wav2Vec, CLAP
- **Video**: VideoMAE, VideoCLIP

**Q: Can I use Erasus with my own custom model?**

Yes! Use ``ErasusUnlearner`` with any ``nn.Module``. For specialised
support, wrap your model following the patterns in ``erasus/models/``.

Strategies
----------

**Q: Which strategy should I use?**

See the :doc:`/guide/strategies` for a decision matrix. As a starting
point, try ``gradient_ascent`` for speed or ``scrub`` for balanced
forgetting and utility.

**Q: Can I combine multiple strategies?**

Yes, use ``EnsembleStrategy`` to combine strategies with configurable
weights.

**Q: How many epochs should I run?**

Start with 3-5 epochs and monitor the forget loss curve. Use early
stopping (``erasus.utils.early_stopping``) to avoid over-unlearning.

Data
----

**Q: How do I split data into forget and retain sets?**

Use ``torch.utils.data.Subset`` with index lists, or use the built-in
dataset classes (``TOFUDataset``, ``MUSEDataset``) that provide
pre-defined splits.

**Q: Do I need the full retain set?**

No. Use a coreset selector (e.g., ``herding``) to create a small
representative retain set for faster training.

Privacy
-------

**Q: Does unlearning guarantee privacy?**

Approximate unlearning does not provide formal privacy guarantees.
For certified guarantees, use the ``certification`` module and pair
unlearning with differential privacy (``privacy`` module).

**Q: How do I verify that unlearning worked?**

Use ``MetricSuite`` with MIA and extraction attack metrics. For formal
verification, use ``UnlearningVerifier`` from the certification module.

Performance
-----------

**Q: How fast is unlearning compared to retraining?**

Typically 10-100× faster than retraining from scratch, depending on
the strategy and model size. Use the efficiency benchmark
(``benchmarks/custom/efficiency_benchmark.py``) to measure this.

**Q: Does Erasus support distributed training?**

Yes, via ``erasus.utils.distributed`` which wraps PyTorch DDP.
