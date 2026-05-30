import importlib
import logging
import os
import pkgutil
import sys
from pathlib import Path
from typing import Tuple, Optional, Union
from spatialwhisperer.jointemb.processing import TranscriptomeTextDualEncoderProcessor
from spatialwhisperer.jointemb.cellwhisperer_lightning import (
    TranscriptomeTextDualEncoderLightning,
)
from spatialwhisperer.config import config, model_path_from_name
from transformers import AutoTokenizer
from transformers.configuration_utils import PretrainedConfig
import torch


def _alias_cellwhisperer_to_spatialwhisperer():
    """Register `cellwhisperer.*` as aliases for `spatialwhisperer.*` in sys.modules.

    Published checkpoints were pickled before the package rename and reference
    classes under `cellwhisperer.jointemb.cellwhisperer_lightning.*`. Without
    this alias, `torch.load(<ckpt>)` raises `ModuleNotFoundError: No module
    named 'cellwhisperer'` on a fresh install of this repo.
    """
    if "cellwhisperer" in sys.modules:
        return
    import spatialwhisperer  # noqa: F401 — ensure parent package is initialised
    sys.modules["cellwhisperer"] = sys.modules["spatialwhisperer"]
    for _, name, _ in pkgutil.walk_packages(
        sys.modules["spatialwhisperer"].__path__, prefix="spatialwhisperer."
    ):
        try:
            importlib.import_module(name)
        except Exception:
            continue
        sys.modules.setdefault(
            name.replace("spatialwhisperer.", "cellwhisperer.", 1),
            sys.modules[name],
        )


_GENEFORMER_HF_REPO = "ctheodoris/Geneformer"
_GENEFORMER_HF_REVISION = "9d41e7053af8a702003d99305cee01cd34b62ab7"
_GENEFORMER_FILES = ("config.json", "pytorch_model.bin", "training_args.bin")
_UNI2_HF_REPO = "MahmoodLab/UNI2-h"


def ensure_geneformer_weights(target_dir: Optional[Path] = None) -> Path:
    """Download the Geneformer-12L-30M files into `target_dir` if missing.

    Defaults to `<PROJECT_ROOT>/resources/geneformer-12L-30M/` (matches
    `config.yaml`'s `model_name_path_map.geneformer`). Returns the target dir.
    """
    from huggingface_hub import hf_hub_download

    if target_dir is None:
        target_dir = config["PROJECT_ROOT"] / "resources" / "geneformer-12L-30M"
    target_dir = Path(target_dir)
    if all((target_dir / f).exists() for f in _GENEFORMER_FILES):
        return target_dir

    target_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Downloading Geneformer to {target_dir}")
    for filename in _GENEFORMER_FILES:
        dst = target_dir / filename
        # exists() follows symlinks; is_symlink() catches dangling ones too
        if dst.exists() or dst.is_symlink():
            continue
        downloaded = hf_hub_download(
            repo_id=_GENEFORMER_HF_REPO,
            filename=f"geneformer-12L-30M/{filename}",
            revision=_GENEFORMER_HF_REVISION,
            token=os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN"),
        )
        dst.symlink_to(downloaded)
    return target_dir


def ensure_uni2_weights(target_dir: Optional[Path] = None) -> Path:
    """Download UNI2-h weights into `target_dir` if missing.

    MahmoodLab/UNI2-h is a gated model. Set `HUGGINGFACE_TOKEN`/`HF_TOKEN`
    after accepting the terms at https://huggingface.co/MahmoodLab/UNI2-h.
    Defaults to `<PROJECT_ROOT>/resources/uni2/` (matches `config.yaml`'s
    `model_name_path_map.uni2`). Returns the target dir.
    """
    from huggingface_hub import snapshot_download

    if target_dir is None:
        target_dir = config["PROJECT_ROOT"] / "resources" / "uni2"
    target_dir = Path(target_dir)
    if (target_dir / "pytorch_model.bin").exists() or (target_dir / "model.safetensors").exists():
        return target_dir

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    logging.info(f"Downloading UNI2 to {target_dir} (gated; requires HF auth)")
    snapshot_download(
        repo_id=_UNI2_HF_REPO,
        local_dir=str(target_dir),
        token=os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN"),
    )
    return target_dir


def load_spatialwhisperer_model(
    hf_repo: str = "Good-Lab/spatialwhisperer",
    auto_download_components: bool = True,
    **kwargs,
):
    """Convenience wrapper: download the SpatialWhisperer checkpoint + the UNI2
    and Geneformer foundation-model weights it depends on, then load.

    Args:
        hf_repo: HuggingFace repo id holding the SpatialWhisperer checkpoint
            (defaults to the public release `Good-Lab/spatialwhisperer`).
        auto_download_components: When True (default), fetch UNI2 + Geneformer
            into `<PROJECT_ROOT>/resources/` if not already present. Skip with
            False if the components are managed externally.
        **kwargs: Passed through to `load_cellwhisperer_model` (e.g. `eval`,
            `cache`, `device`).

    Returns: (pl_model, tokenizer, transcriptome_processor, image_processor).
    """
    if auto_download_components:
        ensure_geneformer_weights()
        ensure_uni2_weights()
    return load_cellwhisperer_model(model_path=f"hf://{hf_repo}", **kwargs)


def _patch_pretrained_config_unpickle():
    """Ensure old checkpoints (transformers<4.45) can be unpickled with transformers>=4.57.

    Newer transformers made _attn_implementation a property backed by
    _attn_implementation_internal and added dtype. Old pickled configs lack
    these attributes, causing AttributeError on access after unpickling."""
    if getattr(PretrainedConfig, "_compat_patched", False):
        return

    def _compat_setstate(self, state):
        if "_attn_implementation_internal" not in state:
            state["_attn_implementation_internal"] = state.pop("_attn_implementation", None)
        if "dtype" not in state:
            state["dtype"] = state.get("torch_dtype", None)
        self.__dict__.update(state)

    PretrainedConfig.__setstate__ = _compat_setstate
    PretrainedConfig._compat_patched = True


def load_cellwhisperer_model(
    model_path: str = None,
    eval: bool = True,
    cache: bool = False,
    transcriptome_model_type: str = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[
    TranscriptomeTextDualEncoderLightning,
    AutoTokenizer,
    TranscriptomeTextDualEncoderProcessor,
]:
    """
    Load a CellWhisperer model from a given path or HuggingFace repo.
    Args:
        model_path: Path to a Lightning .ckpt, OR a HuggingFace repo id
            prefixed with `hf://` (e.g. `hf://Good-Lab/spatialwhisperer`)
            -- the ckpt named `<repo-basename>.ckpt` is fetched via
            `hf_hub_download` (HUGGINGFACE_TOKEN / HF_TOKEN env for private
            repos). Can be None if `transcriptome_model_type` is specified.
        eval: Whether to set the model to eval mode.
        cache: Convert both models into frozencached models to enable caching
        transcriptome_model_type: Type of the transcriptome model. Must be one of "geneformer", "scgpt", "uce" or None. If None, model_path must be specified.
    Returns:
        pl_model: The loaded TranscriptomeTextDualEncoderLightning model.
        tokenizer: The tokenizer used for the model.
        transcriptome_processor: The transcriptome processor used for the model.
    """

    assert not (model_path is None and transcriptome_model_type is None), (
        "Either model_path or transcriptome_model_type must be specified."
    )

    # Old checkpoints reference `cellwhisperer.*` classes; alias them.
    _alias_cellwhisperer_to_spatialwhisperer()

    # The published HF checkpoint ships with UNI2 and Geneformer weights stripped
    # (license + FrozenCachedModel load path), so make sure both are on disk
    # under PROJECT_ROOT/resources/{uni2,geneformer-12L-30M}/ before instantiating
    # the Lightning module. Idempotent: skips work if already cached.
    if model_path is not None:
        try:
            ensure_geneformer_weights()
            ensure_uni2_weights()
        except Exception as e:
            logging.warning(f"Foundation-model weight download skipped: {e}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_path is not None:
        if str(model_path).startswith("hf://"):
            import os
            from huggingface_hub import hf_hub_download
            repo_id = str(model_path).removeprefix("hf://")
            model_path = hf_hub_download(
                repo_id=repo_id,
                filename=repo_id.split("/", 1)[1] + ".ckpt",
                token=os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN"),
            )
        model_path = Path(model_path).expanduser()
        # Backward compat: old checkpoints (transformers<4.45) lack attributes
        # that newer transformers expects on PretrainedConfig subclasses.
        _patch_pretrained_config_unpickle()
        # torch 2.6+ defaults weights_only=True which breaks Lightning checkpoint loading
        _orig_load = torch.load
        torch.load = lambda *args, **kwargs: _orig_load(
            *args,
            **{k: v for k, v in kwargs.items() if k != "weights_only"},
            weights_only=False,
        )
        try:
            pl_model = TranscriptomeTextDualEncoderLightning.load_from_checkpoint(
                model_path
            )
        finally:
            torch.load = _orig_load
    else:
        pl_model = TranscriptomeTextDualEncoderLightning(
            model_config={"transcriptome_model_type": transcriptome_model_type},
            loss_config={},
        )
        pl_model.load_pretrained_models(
            transcriptome_model_directory=model_path_from_name(
                transcriptome_model_type
            ),
            text_model_name_or_path=model_path_from_name(
                pl_model.model.text_model.config.model_type
            ),
            image_model_name_or_path=model_path_from_name(
                pl_model.model.image_model.config.model_type
            ),
        )

    # this is for freezing.
    pl_model.freeze()

    if cache:
        # # This is just for allow caching based on `FrozenCachedModels`, you can omit it
        pl_model.model.freeze_models(force_freeze=True)

    if eval:
        pl_model.eval().to(device)
    else:
        pl_model.to(device)

    processor = TranscriptomeTextDualEncoderProcessor(
        pl_model.model.transcriptome_model.config.model_type,
        model_path_from_name(pl_model.model.text_model.config.model_type),
        pl_model.model.image_model.config.model_type,
    )

    tokenizer = processor.tokenizer
    transcriptome_processor = processor.transcriptome_processor
    image_processor = processor.image_processor

    return pl_model, tokenizer, transcriptome_processor, image_processor
