"""
Lightning module for training a gene expression decoder on frozen raw UNI2 embeddings.

Unlike GeneExpressionDecoderLightning (which operates on CellWhisperer-projected
embeddings), this module takes raw 1536-dim UNI2 features and decodes them directly
to gene expression. This is used as the "two-stage baseline" for the reviewer:
  H&E image → UNI2 → decoder → predicted expression → downstream cell type classifier

During training, the module includes a frozen UNI2 model to extract embeddings from
image patches provided by JointEmbedDataModule. During inference, pre-extracted
1536-dim embeddings can be fed directly via forward().
"""

import pyarrow  # needed

import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Optional
import logging
from pathlib import Path

from spatialwhisperer.jointemb.loss.discriminator import MILinearBlock
from spatialwhisperer.expression_decoder.gene_expression_decoder import (
    GeneExpressionDecoderConfig,
)

logger = logging.getLogger(__name__)


class RawUNI2DecoderLightning(pl.LightningModule):
    """
    Lightning module for training a gene expression decoder on frozen raw UNI2 embeddings.

    Architecture: UNI2 features (1536-dim) → MILinearBlock → Linear → gene expression

    During training, a frozen UNI2 model converts patches → embeddings.
    During inference (forward()), pre-extracted embeddings are accepted directly.
    """

    def __init__(
        self,
        gene_list_path: str,
        uni2_weights_dir: Optional[str] = None,
        uni2_embed_dim: int = 1536,
        projection_dim: int = 2048,
        learning_rate: float = 1e-3,
        max_epochs: int = 4,
        loss_type: str = "mse",
    ):
        super().__init__()

        # Read gene list to determine output dimension
        decoder_config = GeneExpressionDecoderConfig(
            gene_list_path=gene_list_path,
            embedding_dim=projection_dim,
        )
        self.num_genes = decoder_config.num_genes

        # MILinearBlock: 1536 → projection_dim (matches the architecture used
        # in SpotWhisperer's image projection head)
        self.projection = MILinearBlock(uni2_embed_dim, units=projection_dim, bln=True)

        # Linear decoder: projection_dim → num_genes
        self.decoder = nn.Linear(projection_dim, self.num_genes)

        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.loss_type = loss_type

        self._uni2_weights_dir = uni2_weights_dir

        # _uni2_model is intentionally NOT registered as an nn.Module attribute
        # so it is excluded from the checkpoint state_dict. It is loaded lazily
        # at runtime and stored in __dict__ directly (bypassing Module.__setattr__).
        object.__setattr__(self, "_uni2_model", None)

        self.save_hyperparameters()

    def _get_uni2(self):
        """Lazily load and cache the frozen UNI2 model."""
        if self._uni2_model is None:
            from spatialwhisperer.jointemb.uni_model import UNIModel, UNIConfig
            from spatialwhisperer.config import get_path

            weights_dir = self._uni2_weights_dir
            if weights_dir is None:
                weights_dir = str(get_path(["paths", "uni2_weights"]))
                if not Path(weights_dir).exists():
                    # Fallback: look in resources/uni2
                    from spatialwhisperer.config import config as cw_config

                    weights_dir = str(cw_config["PROJECT_ROOT"] / "resources" / "uni2")

            uni_config = UNIConfig(cell_level_model=False, context_model=True)
            self._uni2_model = UNIModel.from_pretrained(
                str(Path(weights_dir) / "pytorch_model.bin"),
                config=uni_config,
            )
            # Freeze UNI2
            for param in self._uni2_model.parameters():
                param.requires_grad = False
            self._uni2_model.eval()
            self._uni2_model = self._uni2_model.to(self.device)
        return self._uni2_model

    def _extract_uni2_embeds(self, batch: Dict) -> torch.Tensor:
        """Extract raw UNI2 embeddings from image patches in the batch."""
        uni2 = self._get_uni2()
        with torch.no_grad():
            _, embeds = uni2(
                patches_ctx=batch["patches_ctx"],
                patches_cell=batch["patches_cell"],
            )
        return embeds  # [B, 1536]

    def forward(self, image_embeds: torch.Tensor) -> torch.Tensor:
        """
        Decode gene expression from pre-extracted UNI2 embeddings.

        Args:
            image_embeds: [batch_size, 1536] raw UNI2 features
        Returns:
            predicted_expression: [batch_size, num_genes] predicted log(expression+1)
        """
        projected = self.projection(image_embeds)
        return self.decoder(projected)

    def _compute_loss(
        self, predicted: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        if self.loss_type == "mse":
            return F.mse_loss(predicted, target)
        elif self.loss_type == "mae":
            return F.l1_loss(predicted, target)
        elif self.loss_type == "huber":
            return F.huber_loss(predicted, target)
        raise ValueError(f"Unknown loss type: {self.loss_type}")

    def _step(self, batch, log_prefix):
        """Shared logic for training and validation steps."""
        image_embeds = self._extract_uni2_embeds(batch)
        predicted_expr = self(image_embeds)
        target_expr = batch["expression_expr"]
        loss = self._compute_loss(predicted_expr, target_expr)

        is_train = log_prefix == "train"
        self.log(
            f"{log_prefix}/loss", loss, on_step=is_train, on_epoch=True, prog_bar=True
        )

        with torch.no_grad():
            correlations = []
            for pred, tgt in zip(predicted_expr, target_expr):
                corr = torch.corrcoef(torch.stack([pred, tgt]))[0, 1]
                if not torch.isnan(corr):
                    correlations.append(corr)
            if correlations:
                self.log(
                    f"{log_prefix}/correlation",
                    torch.stack(correlations).mean(),
                    on_step=is_train,
                    on_epoch=True,
                    prog_bar=not is_train,
                )
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        optimizer = AdamW(
            # Only optimize projection + decoder, not UNI2
            list(self.projection.parameters()) + list(self.decoder.parameters()),
            lr=self.learning_rate,
            betas=(0.9, 0.98),
            weight_decay=0.01,
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.estimated_stepping_batches,
            eta_min=self.learning_rate / 20,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
