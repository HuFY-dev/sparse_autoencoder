"""L2 Reconstruction loss."""

from typing import Any

from jaxtyping import Float, Int64
from pydantic import PositiveInt, validate_call
import torch
from torch import Tensor
from torchmetrics import Metric

from sparse_autoencoder.tensor_types import Axis


class L2ReconstructionLoss(Metric):
    """L2 Reconstruction loss (MSE).

    L2 reconstruction loss is calculated as the sum squared error between each each input vector
    and it's corresponding decoded vector. The original paper found that models trained with some
    loss functions such as cross-entropy loss generally prefer to represent features
    polysemantically, whereas models trained with L2 may achieve the same loss for both
    polysemantic and monosemantic representations of true features.

    You have the option to set L2 reconstruction loss to normalize the input activations before
    calculating the loss. This can be useful because the input vectors can vary in magnitude and
    normalizing them can help to ensure that the loss is not dominated by the magnitude of the
    activations.

    Example:
        >>> import torch
        >>> loss = L2ReconstructionLoss(num_components=1)
        >>> source_activations = torch.tensor([
        ...     [ # Batch 1
        ...         [4., 2.] # Component 1
        ...     ],
        ...     [ # Batch 2
        ...         [2., 0.] # Component 1
        ...     ]
        ... ])
        >>> decoded_activations = torch.tensor([
        ...     [ # Batch 1
        ...         [2., 0.] # Component 1 (MSE of 4)
        ...     ],
        ...     [ # Batch 2
        ...         [0., 0.] # Component 1 (MSE of 2)
        ...     ]
        ... ])
        >>> loss.forward(
        ...     decoded_activations=decoded_activations, source_activations=source_activations
        ... )
        tensor(3.)
    """

    # Torchmetrics settings
    is_differentiable: bool | None = True
    higher_is_better = False
    full_state_update: bool | None = False
    plot_lower_bound: float | None = 0.0

    # Settings
    _num_components: int
    _keep_batch_dim: bool
    _normalization_power: int

    @property
    def keep_batch_dim(self) -> bool:
        """Whether to keep the batch dimension in the loss output."""
        return self._keep_batch_dim

    @keep_batch_dim.setter
    def keep_batch_dim(self, keep_batch_dim: bool) -> None:
        """Set whether to keep the batch dimension in the loss output.

        When setting this we need to change the state to either a list if keeping the batch
        dimension (so we can accumulate all the losses and concatenate them at the end along this
        dimension). Alternatively it should be a tensor if not keeping the batch dimension (so we
        can sum the losses over the batch dimension during update and then take the mean).

        By doing this in a setter we allow changing of this setting after the metric is initialised.
        """
        self._keep_batch_dim = keep_batch_dim
        self.reset()  # Reset the metric to update the state
        if keep_batch_dim and not isinstance(self.mse, list):
            self.add_state(
                "mse",
                default=[],
                dist_reduce_fx="sum",
            )
        elif not isinstance(self.mse, Tensor):
            self.add_state(
                "mse",
                default=torch.zeros(self._num_components),
                dist_reduce_fx="sum",
            )

    @property
    def normalization_power(self) -> int:
        """The power of ||x||_2 in the normalization step.

        The normalization is done by dividing the MSE by the norm of the input activations raised to
        this power. Normally it should be one of {0, 1, 2}. This is useful for e.g. normalizing the
        loss by the input norm. For regular L2,this should be 0.
        """
        return self._normalization_power

    @normalization_power.setter
    def normalization_power(self, normalization_power: int) -> None:
        """Set the normalization power."""
        self._normalization_power = normalization_power
        self.reset()  # Reset the metric to update the state

    # State
    mse: (
        Float[Tensor, Axis.COMPONENT_OPTIONAL]
        | list[Float[Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL)]]
        | None
    ) = None
    num_activation_vectors: Int64[Tensor, Axis.SINGLE_ITEM]

    @validate_call
    def __init__(
        self,
        num_components: PositiveInt = 1,
        *,
        keep_batch_dim: bool = False,
        normalization_method: str = "none",
    ) -> None:
        """Initialise the L2 reconstruction loss."""
        super().__init__()
        self._num_components = num_components
        self.keep_batch_dim = keep_batch_dim
        if normalization_method == "none":
            self.normalization_power = 0
        elif normalization_method == "input_norm":
            self.normalization_power = 1
        elif normalization_method == "input_norm_squared":
            self.normalization_power = 2
        else:
            error_message = f"Normalization method {normalization_method} not recognised."
            raise ValueError(error_message)
        self.add_state(
            "num_activation_vectors",
            default=torch.tensor(0, dtype=torch.int64),
            dist_reduce_fx="sum",
        )

    @staticmethod
    def calculate_mse(
        decoded_activations: Float[
            Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.INPUT_OUTPUT_FEATURE)
        ],
        source_activations: Float[
            Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.INPUT_OUTPUT_FEATURE)
        ],
    ) -> Float[Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL)]:
        """Calculate the MSE."""
        return (decoded_activations - source_activations).pow(2).mean(dim=-1)

    @staticmethod
    def normalize_mse(
        source_activations: Float[
            Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.INPUT_OUTPUT_FEATURE)
        ],
        mse: Float[Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL)],
        normalization_power: int = 0,
    ) -> Float[Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL)]:
        """Normalize the mse by input norm.

        When `normalization_power` is set to 0, this is equivalent to not normalizing the loss.
        When set to 1, this is equivalent to normalizing the loss by the input norm, making the
        loss similar to (1 - cosine similarity) * input norm. When set to 2, this is equivalent
        to normalizing the loss by the input norm squared, making the loss similar to (1 - cosine
        similarity).
        """
        return mse / source_activations.norm(dim=-1, p=2).pow(normalization_power)

    def update(
        self,
        decoded_activations: Float[
            Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.INPUT_OUTPUT_FEATURE)
        ],
        source_activations: Float[
            Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.INPUT_OUTPUT_FEATURE)
        ],
        **kwargs: Any,  # type: ignore # noqa: ARG002, ANN401 (allows combining with other metrics)
    ) -> None:
        """Update the metric state.

        If we're keeping the batch dimension, we simply take the mse of the activations
        (over the features dimension) and then append this tensor to a list. Then during compute we
        just concatenate and return this list. This is useful for e.g. getting L1 loss by batch item
        when resampling neurons (see the neuron resampler for details).

        By contrast if we're averaging over the batch dimension, we sum the activations over the
        batch dimension during update (on each process), and then divide by the number of activation
        vectors on compute to get the mean.

        Args:
            decoded_activations: The decoded activations from the autoencoder.
            source_activations: The source activations from the autoencoder.
            **kwargs: Ignored keyword arguments (to allow use with other metrics in a collection).
        """
        mse = self.calculate_mse(decoded_activations, source_activations)
        mse = self.normalize_mse(
            source_activations, mse, normalization_power=self.normalization_power
        )

        if self.keep_batch_dim:
            self.mse.append(mse)  # type: ignore
        else:
            self.mse += mse.sum(dim=0)
            self.num_activation_vectors += source_activations.shape[0]

    def compute(self) -> Float[Tensor, Axis.COMPONENT_OPTIONAL]:
        """Compute the metric."""
        return (
            torch.cat(self.mse)  # type: ignore
            if self.keep_batch_dim
            else self.mse / self.num_activation_vectors
        )
