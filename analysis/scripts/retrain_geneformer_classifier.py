"""
Retrain Geneformer classifier (frozen backbone + linear head) on CellXGene Census.
Standalone script version that avoids jupyter-nbconvert and inline python -c quoting issues.
"""

import pyarrow

import sys
from pathlib import Path

# finetuning_eval is a local module under src/figures/notebooks/
_project_dir = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_project_dir / "src" / "figures" / "notebooks"))

import torch
import torch.nn as nn
import anndata
import lightning
import logging
import os

from lightning.pytorch import Trainer
from cellwhisperer.jointemb.dataset.jointemb import JointEmbedDataModule
from finetuning_eval.models.geneformer import GeneformerCelltypeModel, GeneformerConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Data module
dm = JointEmbedDataModule(
    tokenizer="bert",
    transcriptome_processor=snakemake.wildcards.model,
    dataset_names="cellxgene_census",
    batch_size=snakemake.params.batch_size,
    train_fraction=0.9,
    include_labels=snakemake.params.label_col,
    use_replicates=False,
)

# Count classes
adata = anndata.read_h5ad(snakemake.input.training_data, backed="r")
num_classes = adata.obs[snakemake.params.label_col].drop_duplicates().shape[0]
logger.info(f"Number of classes: {num_classes}")
adata.file.close()

# Model
model = GeneformerCelltypeModel(
    GeneformerConfig(), num_classes=num_classes, freeze=snakemake.params.freeze_fm
)


class FinetuningModule(lightning.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def step(self, batch, batch_idx):
        label = batch.pop("labels")
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, device=self.device)
        pred = self.model(**batch)
        return nn.functional.cross_entropy(pred, label)

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=snakemake.params.learning_rate)


finetune_module = FinetuningModule(model)

dm.prepare_data()
dm.setup()

try:
    from lightning.pytorch.loggers import WandbLogger

    wandb_logger = WandbLogger(
        project="cellwhisperer",
        entity="single-cellm",
        name=f"retrain_{snakemake.wildcards.model}_{snakemake.wildcards.training_options}",
        group=os.environ.get("WANDB_RUN_GROUP", "icml_revisions"),
        log_model=False,
    )
except Exception as e:
    logger.warning(f"wandb init failed: {e}, using default logger")
    wandb_logger = True

trainer = Trainer(
    max_epochs=snakemake.params.num_epochs,
    logger=wandb_logger,
    precision="bf16",
)
trainer.fit(finetune_module, datamodule=dm)

os.makedirs(os.path.dirname(snakemake.output.model_weights), exist_ok=True)
torch.save(model.state_dict(), snakemake.output.model_weights)
logger.info(f"Saved model to {snakemake.output.model_weights}")
