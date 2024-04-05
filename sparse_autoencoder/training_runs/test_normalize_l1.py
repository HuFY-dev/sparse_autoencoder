"""Run an sweep on all layers of GPT2 Small.

Command:

```bash
git clone https://github.com/ai-safety-foundation/sparse_autoencoder.git && cd sparse_autoencoder &&
poetry env use python3.11 && poetry install &&
poetry run python sparse_autoencoder/training_runs/gpt2.py
```
"""

import os

from sparse_autoencoder import (
    ActivationResamplerHyperparameters,
    AutoencoderHyperparameters,
    Hyperparameters,
    LossHyperparameters,
    Method,
    OptimizerHyperparameters,
    Parameter,
    PipelineHyperparameters,
    SourceDataHyperparameters,
    SourceModelHyperparameters,
    SweepConfig,
    sweep,
)


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def train_gpt_small_mlp_layers(
    expansion_factor: int = 4,
    n_layers: int = 12,
) -> None:
    """Run a new sweep experiment on GPT 2 Small's MLP layers.

    Args:
        expansion_factor: Expansion factor for the autoencoder.
        n_layers: Number of layers to train on. Max is 12.

    """
    sweep_config = SweepConfig(
        parameters=Hyperparameters(
            loss=LossHyperparameters(
                l1_coefficient=Parameter(max=1e-4, min=1e-5),
                match_l1_l2_scale=Parameter(value=True),
            ),
            optimizer=OptimizerHyperparameters(
                lr=Parameter(max=0.001, min=0.00001),
            ),
            source_model=SourceModelHyperparameters(
                name=Parameter("gpt2"),
                cache_names=Parameter(
                    [f"blocks.{layer}.hook_mlp_out" for layer in range(n_layers)]
                ),
                hook_dimension=Parameter(768),
            ),
            source_data=SourceDataHyperparameters(
                dataset_path=Parameter("alancooney/sae-monology-pile-uncopyrighted-tokenizer-gpt2"),
                context_size=Parameter(256),
                pre_tokenized=Parameter(value=True),
                pre_download=Parameter(value=False),  # Default to streaming the dataset
            ),
            autoencoder=AutoencoderHyperparameters(
                expansion_factor=Parameter(value=expansion_factor),
            ),
            pipeline=PipelineHyperparameters(
                max_activations=Parameter(20_000_000),
                checkpoint_frequency=Parameter(10_000_000),
                validation_frequency=Parameter(2_000_000),
                max_store_size=Parameter(100_000),
                source_data_batch_size=Parameter(16),
                train_batch_size=Parameter(4096),
                log_frequency=Parameter(1),
            ),
            activation_resampler=ActivationResamplerHyperparameters(
                resample_interval=Parameter(200_000_000),
                n_activations_activity_collate=Parameter(100_000_000),
                threshold_is_dead_portion_fires=Parameter(1e-6),
                max_n_resamples=Parameter(4),
                resample_dataset_size=Parameter(100_000),
            ),
        ),
        method=Method.RANDOM,
    )

    sweep(sweep_config=sweep_config)


if __name__ == "__main__":
    train_gpt_small_mlp_layers(expansion_factor=16)