from dataclasses import dataclass
from typing import List

import yaml

from ...util.hparams import HyperParams


@dataclass
class MOSEHyperParams(HyperParams):
    # Method
    layers: List[int]
    num_steps: int
    lr: float
    weight_decay: float
    kl_factor: float
    norm_constraint: float

    # Module templates
    rewrite_module_tmp: str
    layer_module_tmp: str
    mlp_module_tmp: str
    attn_module_tmp: str
    ln_f_module: str
    lm_head_module: str
    device: int
    alg_name: str
    model_name: str

    # Defaults
    batch_size: int = 128
    max_length: int = 30
    block: int = 4

    # Closed-form target construction defaults reused from MEMIT-style editors.
    fact_token: str = "subject_last"
    v_num_grad_steps: int = 20
    v_lr: float = 5e-1
    v_loss_layer: int = -1
    v_weight_decay: float = 5e-1
    clamp_norm_factor: float = 4.0

    # Closed-form MOSE-specific controls.
    preservation_weight: float = 1.0
    max_history: int = 256

    model_parallel: bool = False

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):
        if ".yaml" not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + ".yaml"

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)

        assert (
            config and config["alg_name"] in {"OFT", "MOSE"}
        ) or print(
            f"MOSEHyperParams can not load from {hparams_name_or_path}, "
            f'alg_name is {config["alg_name"]} '
        )
        return cls(**config)


# Compatibility alias for existing OFT-named configs and imports.
OFTHyperParams = MOSEHyperParams
