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
normalization_method: Parameter[str] = field(default=Parameter(value="none"))
"""Normalize by input norm.

Choices: ["none", "input_norm", "input_norm_squared"]

The normalization method to use for the mse (L2) loss. Default is "none", which returns the raw
L2 loss. If set to "input_norm", the loss is normalized by the input norm. If set to "input_norm
squared", the loss is normalized by the input norm squared. 

NOTE: This will only change the scale of the l2 loss but not the l1, so it's recommended to use
different l1 coefficients for different normalization methods.
Intuitions of different normalization methods:
- "none": This is equivalent to not normalizing the loss, so inputs with high norms (>100) will
have significantly higher losses than inputs with lower norms (~50). This term can be
approximated by 2 * norm(x)^2 * (1 - cosine_similarity(x', x)) which grows quadratically with
the norm of the input assuming the cosine similarity is constant.
- "input_norm": This divides the loss by the norm of the input vector. This term can be
approximated by 2 * norm(x) * (1 - cosine_similarity(x', x)) which grows linearly with the norm
assuming the cosine similarity is constant.
- "input_norm_squared": This divides the loss by the squared norm of the input vector. This term
can be approximated by 2 * (1 - cosine_similarity(x', x)) which is constant with respect to the
norm of the input assuming the cosine similarity is constant. This allows the model to better
pick up information from low norm inputs.
This can be useful because the input vectors can vary in magnitude and normalizing them can help
to ensure that the loss is not dominated by activations of high magnitudes (often
uninterpretable activations from the <|endoftext|> token).
"""
```