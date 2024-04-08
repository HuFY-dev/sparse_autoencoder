# New hyperparameters

## `AutoencoderHyperparameters`

```python
sae_type: Parameter[str] = field(default=Parameter("sae"))
"""Type of the autoencoder

Choices: ["sae", "normalized_sae"]
Default is "sae" which is the original implementation of SAE
"""
```

```python
noise_scale: Parameter[float] = field(default=Parameter(1.0))
"""Noise scale.

Scale of the Gaussian noise to add to the output before activation.
This is only applicable for normalized_sae that uses the TanhEncoder.
"""
```

## `LossHyperparameters`

```python
l1_normalization_power: Parameter[float] = field(default=Parameter(0.0))
"""The power of ||x||_2 in the L1 normalization step.

This normalization is used to match the scale of L1 with L2. This is useful when the L1 and L2
losses are not on the same scale, which can cause one loss to dominate the other.

NOTE: This will only change the scale of the l1 loss, so it's recommended to use different l1 coefficients for different normalization methods.
"""

l2_normalization_power: Parameter[float] = field(default=Parameter(0.0))
"""The power of ||x||_2 in the L2 normalization step.

The normalization is done by multiplying the MSE by the norm of the input activations raised to
this power. This is useful when there exist high-norm outliers in the input activations causing
the model to overfit. For regular L2, this should be 0.

This can be useful because the input vectors can vary in magnitude and normalizing them can help
to ensure that the loss is not dominated by activations of high magnitudes (often
uninterpretable activations from the <|endoftext|> token).
"""
```
