from copy import deepcopy
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...util import nethook
from ...util.generate import generate_fast
from ..memit.compute_ks import compute_ks
from ..memit.compute_z import compute_z, get_module_input_output_at_words
from .mose_hparams import MOSEHyperParams


CONTEXT_TEMPLATES_CACHE = None
EDIT_HISTORY: Dict[str, Dict[str, torch.Tensor]] = {}


def _rewrite_module_template(rewrite_module_tmp: str) -> str:
    for suffix in (".weight", ".R"):
        if rewrite_module_tmp.endswith(suffix):
            return rewrite_module_tmp[: -len(suffix)]
    return rewrite_module_tmp


def _format_module_name(hparams: MOSEHyperParams, layer: int) -> str:
    return _rewrite_module_template(hparams.rewrite_module_tmp).format(layer)


def _resolve_module(model: AutoModelForCausalLM, module_name: str) -> nn.Module:
    return nethook.get_module(model, module_name)


def _weight_param_name(module_name: str) -> str:
    return f"{module_name}.weight"


def _weight_matrix(module: nn.Module) -> torch.Tensor:
    weight = module.weight
    if hasattr(module, "in_features") and hasattr(module, "out_features"):
        return weight
    return weight.T


def _assign_weight_matrix(module: nn.Module, matrix: torch.Tensor) -> None:
    with torch.no_grad():
        if hasattr(module, "in_features") and hasattr(module, "out_features"):
            module.weight[...] = matrix.to(module.weight.dtype)
        else:
            module.weight[...] = matrix.T.to(module.weight.dtype)


def _prepare_requests(requests: List[Dict]) -> List[Dict]:
    prepared = deepcopy(requests)
    for request in prepared:
        if request["target_new"] and request["target_new"][0] != " ":
            request["target_new"] = " " + request["target_new"]

        if "{}" not in request["prompt"]:
            subject = request.get("subject")
            if subject is None:
                raise ValueError(
                    "MOSE closed-form editing needs `subject` when prompt has no `{}`."
                )
            if subject not in request["prompt"]:
                raise ValueError(
                    f"Subject `{subject}` does not appear in prompt `{request['prompt']}`."
                )
            request["prompt"] = request["prompt"].replace(subject, "{}")
    return prepared


def _get_context_templates(model, tok):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = [["{}"]] + [
            [
                f.replace("{", " ").replace("}", " ") + ". {}"
                for f in generate_fast(
                    model,
                    tok,
                    ["The", "Therefore", "Because", "I", "You"],
                    n_gen_per_prompt=n_gen // 5,
                    max_out_len=length,
                )
            ]
            for length, n_gen in [(10, 5)]
        ]
        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE


def _orthogonal_procrustes(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    cross_cov = target @ source.T
    u, _, vh = torch.linalg.svd(cross_cov, full_matrices=False)
    rotation = u @ vh

    if torch.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vh

    return rotation


def _pinv_targets(weight: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.linalg.lstsq(weight, targets).solution


def _gather_history(
    weight_name: str, current_param: torch.Tensor, device: torch.device
):
    hist = EDIT_HISTORY.get(weight_name)
    if hist is None or hist["keys"].numel() == 0:
        return None, None

    if not torch.allclose(
        current_param.detach().cpu(),
        hist["expected_weight"],
        atol=1e-5,
        rtol=1e-4,
    ):
        EDIT_HISTORY.pop(weight_name, None)
        return None, None

    return hist["keys"].to(device), hist["values"].to(device)


def _update_history(
    weight_name: str,
    new_keys: torch.Tensor,
    new_values: torch.Tensor,
    expected_weight: torch.Tensor,
    max_history: int,
) -> None:
    hist = EDIT_HISTORY.get(weight_name)
    keys_cpu = new_keys.detach().cpu()
    values_cpu = new_values.detach().cpu()

    if hist is None:
        keys = keys_cpu
        values = values_cpu
    else:
        keys = torch.cat([hist["keys"], keys_cpu], dim=1)
        values = torch.cat([hist["values"], values_cpu], dim=1)

    if keys.size(1) > max_history:
        keys = keys[:, -max_history:]
        values = values[:, -max_history:]

    EDIT_HISTORY[weight_name] = {
        "keys": keys,
        "values": values,
        "expected_weight": expected_weight.detach().cpu(),
    }


def apply_mose_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: MOSEHyperParams,
    copy=False,
    return_orig_weights=False,
    keep_original_weight=False,
    **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    weights_copy = {}
    if copy:
        model = deepcopy(model)

    deltas = execute_mose(model, tok, requests, hparams)

    with torch.no_grad():
        for weight_name, upd_matrix in deltas.items():
            param = nethook.get_parameter(model, weight_name)
            if return_orig_weights and weight_name not in weights_copy:
                weights_copy[weight_name] = param.detach().clone()
            param[...] += upd_matrix.to(param.dtype)

    print(f"New weights successfully inserted into {list(deltas.keys())}")

    if not keep_original_weight:
        weights_copy = {}

    return model, weights_copy


def execute_mose(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: MOSEHyperParams,
    **kwargs: Any,
) -> Dict[str, torch.Tensor]:
    """
    Closed-form MOSE update via orthogonal Procrustes.
    """

    device = torch.device(f"cuda:{hparams.device}")
    model = model.to(device)
    requests = _prepare_requests(requests)

    for request in requests:
        print(
            "Executing MOSE closed-form edit for: "
            f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']}]"
        )

    module_hparams = deepcopy(hparams)
    module_hparams.rewrite_module_tmp = _rewrite_module_template(
        hparams.rewrite_module_tmp
    )

    module_info = {}
    weights_copy = {}
    for layer in hparams.layers:
        module_name = _format_module_name(hparams, layer)
        module = _resolve_module(model, module_name)
        weight_name = _weight_param_name(module_name)
        param = nethook.get_parameter(model, weight_name)
        module_info[layer] = {
            "module_name": module_name,
            "weight_name": weight_name,
            "module": module,
            "param": param,
        }
        weights_copy[weight_name] = param.detach().clone()

    context_templates = _get_context_templates(model, tok)
    z_layer = hparams.layers[-1]
    zs = torch.stack(
        [
            compute_z(model, tok, request, module_hparams, z_layer, context_templates)
            for request in requests
        ],
        dim=1,
    )

    deltas: Dict[str, torch.Tensor] = {}
    history_updates: Dict[str, Dict[str, torch.Tensor]] = {}

    for i, layer in enumerate(hparams.layers):
        info = module_info[layer]
        module_name = info["module_name"]
        weight_name = info["weight_name"]
        module = info["module"]

        print(f"\n\nLAYER {layer}\n")

        layer_keys = compute_ks(
            model, tok, requests, module_hparams, layer, context_templates
        ).T.to(device=device, dtype=torch.float64)
        print(f"Writing {layer_keys.size(1)} key/value pair(s) into layer {layer}")

        final_outputs = get_module_input_output_at_words(
            model,
            tok,
            z_layer,
            context_templates=[request["prompt"] for request in requests],
            words=[request["subject"] for request in requests],
            module_template=module_hparams.layer_module_tmp,
            fact_token_strategy=module_hparams.fact_token,
            track="out",
        ).T.to(device=device, dtype=torch.float64)

        targets = zs.to(device=device, dtype=torch.float64) - final_outputs
        resid = targets / (len(hparams.layers) - i)

        current_outputs = get_module_input_output_at_words(
            model,
            tok,
            layer,
            context_templates=[request["prompt"] for request in requests],
            words=[request["subject"] for request in requests],
            module_template=module_name,
            fact_token_strategy=module_hparams.fact_token,
            track="out",
        ).T.to(device=device, dtype=torch.float64)
        desired_outputs = current_outputs + resid

        weight_matrix = _weight_matrix(module).detach().to(
            device=device, dtype=torch.float64
        )
        pseudo_targets = _pinv_targets(weight_matrix, desired_outputs)

        hist_keys, hist_values = _gather_history(weight_name, info["param"], device)
        if hist_keys is not None:
            hist_keys = hist_keys.to(dtype=torch.float64)
            hist_values = hist_values.to(dtype=torch.float64)
            hist_pseudo_targets = _pinv_targets(weight_matrix, hist_values)
            hist_scale = torch.sqrt(
                torch.tensor(
                    module_hparams.preservation_weight,
                    device=device,
                    dtype=torch.float64,
                )
            )
            source = torch.cat([hist_scale * hist_keys, layer_keys], dim=1)
            target = torch.cat([hist_scale * hist_pseudo_targets, pseudo_targets], dim=1)
        else:
            source = layer_keys
            target = pseudo_targets

        rotation = _orthogonal_procrustes(source, target)
        new_weight = weight_matrix @ rotation
        upd_matrix = (new_weight - weight_matrix).to(dtype=_weight_matrix(module).dtype)

        print("orig norm", torch.linalg.norm(weight_matrix.float()))
        print("upd norm", torch.linalg.norm(upd_matrix.float()))

        _assign_weight_matrix(module, new_weight.to(dtype=_weight_matrix(module).dtype))
        deltas[weight_name] = (
            info["param"].detach().cpu() - weights_copy[weight_name].detach().cpu()
        )
        history_updates[weight_name] = {
            "keys": layer_keys.detach(),
            "values": desired_outputs.detach(),
            "expected_weight": info["param"].detach(),
        }

    for weight_name, update in history_updates.items():
        _update_history(
            weight_name,
            update["keys"],
            update["values"],
            update["expected_weight"],
            module_hparams.max_history,
        )

    with torch.no_grad():
        for layer in hparams.layers:
            info = module_info[layer]
            weight_name = info["weight_name"]
            info["param"][...] = weights_copy[weight_name]

    print(f"Deltas successfully computed for {list(deltas.keys())}")
    return deltas


# Compatibility aliases so existing OFT-based configs and scripts keep working.
OFTHyperParams = MOSEHyperParams
apply_oft_to_model = apply_mose_to_model
execute_oft = execute_mose
