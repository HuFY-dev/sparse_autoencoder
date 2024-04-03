"""Sparse Autoencoder loss."""

from typing import Any

from jaxtyping import Float, Int64
from pydantic import PositiveFloat, PositiveInt, validate_call
import torch
from torch import Tensor
from torchmetrics import Metric

from sparse_autoencoder.metrics.loss.l1_absolute_loss import L1AbsoluteLoss
from sparse_autoencoder.metrics.loss.l2_reconstruction_loss import L2ReconstructionLoss
from sparse_autoencoder.tensor_types import Axis


class SparseAutoencoderLoss(Metric):
    """Sparse Autoencoder loss.

    This is the same as composing `L1AbsoluteLoss() * l1_coefficient + L2ReconstructionLoss()`. It
    is separated out so that you can use all three metrics (l1, l2, total loss) in the same
    `MetricCollection` and they will then share state (to avoid calculating the same thing twice).
    """

    # Torchmetrics settings
    is_differentiable: bool | None = True
    higher_is_better = False
    full_state_update: bool | None = False
    plot_lower_bound: float | None = 0.0

    # Settings
    _num_components: int
    _keep_batch_dim: bool
    _l2_normalization_power: int
    _l1_coefficient: float
    _l1_normalization_power: int

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
            self.add_state(
                "absolute_loss",
                default=[],
                dist_reduce_fx="sum",
            )
        elif not isinstance(self.mse, Tensor):
            self.add_state(
                "mse",
                default=torch.zeros(self._num_components),
                dist_reduce_fx="sum",
            )
            self.add_state(
                "absolute_loss",
                default=torch.zeros(self._num_components),
                dist_reduce_fx="sum",
            )

    @property
    def l2_normalization_power(self) -> int:
        """The power of ||x||_2 in the normalization step.

        The normalization is done by dividing the MSE by the norm of the input activations raised to
        this power. Normally it should be one of {0, 1, 2}. This is useful for e.g. normalizing the
        loss by the input norm. For regular L2,this should be 0.
        """
        return self._l2_normalization_power

    @l2_normalization_power.setter
    def l2_normalization_power(self, l2_normalization_power: int) -> None:
        """Set the normalization power."""
        self._l2_normalization_power = l2_normalization_power
        self.reset()  # Reset the metric to update the state

    # State
    absolute_loss: (
        Float[Tensor, Axis.names(Axis.COMPONENT_OPTIONAL)]
        | list[Float[Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL)]]
        | None
    ) = None
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
        l1_coefficient: PositiveFloat = 0.001,
        *,
        keep_batch_dim: bool = False,
        l2_normalization_method: str = "none",
        match_l1_l2_scale: bool = False,
    ):
        """Initialise the metric."""
        super().__init__()
        self._num_components = num_components
        self.keep_batch_dim = keep_batch_dim
        if l2_normalization_method == "none":
            self.l2_normalization_power = 0
        elif l2_normalization_method == "input_norm":
            self.l2_normalization_power = 1
        elif l2_normalization_method == "input_norm_squared":
            self.l2_normalization_power = 2
        else:
            error_message = f"Normalization method {l2_normalization_method} not recognised."
            raise ValueError(error_message)
        self._l1_coefficient = l1_coefficient
        self._l1_normalization_power = (
            2 - self.l2_normalization_power
            if match_l1_l2_scale
            else 0 # Default to not match scale
        )

        # Add the state
        self.add_state(
            "num_activation_vectors",
            default=torch.tensor(0, dtype=torch.int64),
            dist_reduce_fx="sum",
        )

    @staticmethod
    def normalize_abs_sum(
        source_activations: Float[
            Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.INPUT_OUTPUT_FEATURE)
        ],
        absolute_loss: Float[Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL)],
        l1_normalization_power: int,
    ) -> Float[Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL)]:
        """Normalize the absolute sum of the learned activations.
        
        This is designed to work with the normalized MSE loss, where the MSE is normalized by the
        input norm. This normalization matches the L1 loss to the same scale as the MSE loss. More
        details see ANALYSIS.md.

        Args:
            source_activations: Source activations.
            absolute_loss: The original L1 loss.
            l1_normalization_power: L1 normalization power.

        Returns:
            Normalized absolute sum of the learned activations.
        """
        return absolute_loss * source_activations.norm(dim=-1, p=2).pow(l1_normalization_power)

    def update(
        self,
        source_activations: Float[
            Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.INPUT_OUTPUT_FEATURE)
        ],
        learned_activations: Float[
            Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.LEARNT_FEATURE)
        ],
        decoded_activations: Float[
            Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.INPUT_OUTPUT_FEATURE)
        ],
        **kwargs: Any,  # type: ignore # noqa: ARG002, ANN401 (allows combining with other metrics))
    ) -> None:
        """Update the metric."""
        absolute_loss = L1AbsoluteLoss.calculate_abs_sum(learned_activations)
        absolute_loss = self.normalize_abs_sum(
            source_activations, absolute_loss, self._l1_normalization_power
        )
        mse = L2ReconstructionLoss.calculate_mse(decoded_activations, source_activations)
        mse = L2ReconstructionLoss.normalize_mse(source_activations, mse, self.l2_normalization_power)

        if self.keep_batch_dim:
            self.absolute_loss.append(absolute_loss)  # type: ignore
            self.mse.append(mse)  # type: ignore
        else:
            self.absolute_loss += absolute_loss.sum(dim=0)
            self.mse += mse.sum(dim=0)
            self.num_activation_vectors += learned_activations.shape[0]

    def compute(self) -> Tensor:
        """Compute the metric."""
        l1 = (
            torch.cat(self.absolute_loss)  # type: ignore
            if self.keep_batch_dim
            else self.absolute_loss / self.num_activation_vectors
        )

        l2 = (
            torch.cat(self.mse)  # type: ignore
            if self.keep_batch_dim
            else self.mse / self.num_activation_vectors
        )

        return l1 * self._l1_coefficient + l2

    def forward(  # type: ignore[override] (narrowing)
        self,
        source_activations: Float[
            Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.INPUT_OUTPUT_FEATURE)
        ],
        learned_activations: Float[
            Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.LEARNT_FEATURE)
        ],
        decoded_activations: Float[
            Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.INPUT_OUTPUT_FEATURE)
        ],
    ) -> Tensor:
        """Forward pass."""
        return super().forward(
            source_activations=source_activations,
            learned_activations=learned_activations,
            decoded_activations=decoded_activations,
        )
