from __future__ import annotations

import contextlib
import json
from pathlib import Path

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        if rank <= 0 or alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        self.base = base
        self.rank = int(rank)
        self.scale = float(alpha) / self.rank
        self.lora_a = nn.Parameter(torch.empty(self.rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_a, a=5 ** 0.5)
        self.enabled = True
        self.token_mask: torch.Tensor | None = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        result = self.base(inputs)
        if not self.enabled:
            return result
        hidden = torch.nn.functional.linear(inputs, self.lora_a)
        update = torch.nn.functional.linear(hidden, self.lora_b)
        if self.token_mask is not None:
            if self.token_mask.shape != inputs.shape[:-1]:
                raise ValueError("LoRA token mask must match batch and sequence dimensions")
            update = update * self.token_mask.to(update).unsqueeze(-1)
        return result + update * self.scale


class LoRAController:
    def __init__(self, modules: dict[str, LoRALinear], target: str = "q_proj") -> None:
        self.modules = modules
        self.target = target

    @contextlib.contextmanager
    def disabled(self):
        previous = {index: module.enabled for index, module in self.modules.items()}
        try:
            for module in self.modules.values():
                module.enabled = False
            yield
        finally:
            for index, module in self.modules.items():
                module.enabled = previous[index]

    def parameters(self):
        for module in self.modules.values():
            yield module.lora_a
            yield module.lora_b

    @contextlib.contextmanager
    def masked_tokens(self, mask: torch.Tensor):
        previous = {key: module.token_mask for key, module in self.modules.items()}
        try:
            for module in self.modules.values():
                module.token_mask = mask
            yield
        finally:
            for key, module in self.modules.items():
                module.token_mask = previous[key]


def _layers(model):
    backbone = getattr(model, "model", None)
    layers = getattr(backbone, "layers", None)
    if layers is None:
        raise RuntimeError("model does not expose model.layers")
    return layers


def install_lora(
    model,
    *,
    target: str,
    layer_start: int,
    layer_end: int,
    rank: int,
    alpha: float,
) -> LoRAController:
    targets = tuple(value.strip() for value in target.replace("+", ",").split(",") if value.strip())
    supported = {"q_proj", "k_proj", "v_proj", "o_proj", "down_proj"}
    if not targets or set(targets) - supported:
        raise ValueError(f"unsupported LoRA target: {target}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    layers = _layers(model)
    if not 0 <= layer_start <= layer_end < len(layers):
        raise ValueError(f"LoRA layer range must fit [0, {len(layers) - 1}]")
    modules: dict[str, LoRALinear] = {}
    for layer_index in range(layer_start, layer_end + 1):
        for projection_name in targets:
            parent = (
                layers[layer_index].mlp
                if projection_name == "down_proj"
                else layers[layer_index].self_attn
            )
            projection = getattr(parent, projection_name)
            if isinstance(projection, LoRALinear):
                raise RuntimeError(
                    f"layer {layer_index} {projection_name} already has LoRA"
                )
            wrapper = LoRALinear(projection, rank, alpha).to(
                device=projection.weight.device,
                dtype=projection.weight.dtype,
            )
            setattr(parent, projection_name, wrapper)
            modules[f"{layer_index}.{projection_name}"] = wrapper
    canonical_target = "+".join(targets)
    controller = LoRAController(modules, target=canonical_target)
    controller.model_identifier = getattr(model, "_attuner_model_identifier", None)
    model._kv_order_lora_controller = controller
    return controller


def install_q_lora(
    model,
    *,
    layer_start: int,
    layer_end: int,
    rank: int,
    alpha: float,
) -> LoRAController:
    return install_lora(
        model,
        target="q_proj",
        layer_start=layer_start,
        layer_end=layer_end,
        rank=rank,
        alpha=alpha,
    )


def save_adapter(
    path: Path,
    controller: LoRAController,
    *,
    layer_start: int,
    layer_end: int,
    rank: int,
    alpha: float,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    state = {}
    for key, module in controller.modules.items():
        state[f"{key}.lora_a"] = module.lora_a.detach().cpu()
        state[f"{key}.lora_b"] = module.lora_b.detach().cpu()
    torch.save(state, path / "adapter.pt")
    (path / "adapter_config.json").write_text(json.dumps({
        "target": controller.target,
        "layer_start": layer_start,
        "layer_end": layer_end,
        "rank": rank,
        "alpha": alpha,
        "model_identifier": getattr(controller, "model_identifier", None),
    }, indent=2) + "\n")


def load_adapter(model, path: Path) -> LoRAController:
    config = json.loads((path / "adapter_config.json").read_text())
    controller = install_lora(
        model,
        target=str(config.get("target", "q_proj")),
        layer_start=int(config["layer_start"]),
        layer_end=int(config["layer_end"]),
        rank=int(config["rank"]),
        alpha=float(config["alpha"]),
    )
    load_adapter_weights(controller, path)
    return controller


def load_adapter_weights(controller: LoRAController, path: Path) -> None:
    config = json.loads((path / "adapter_config.json").read_text())
    expected_model = getattr(controller, "model_identifier", None)
    if expected_model and config.get("model_identifier") != expected_model:
        raise ValueError("Adapter model identity differs; cross-model reuse is forbidden")
    expected = {
        "target": controller.target,
        "layer_start": min(int(key.split(".", 1)[0]) for key in controller.modules),
        "layer_end": max(int(key.split(".", 1)[0]) for key in controller.modules),
        "rank": next(iter(controller.modules.values())).rank,
        "alpha": next(iter(controller.modules.values())).scale * next(iter(controller.modules.values())).rank,
    }
    actual = {
        "target": str(config.get("target", "q_proj")),
        "layer_start": int(config["layer_start"]),
        "layer_end": int(config["layer_end"]),
        "rank": int(config["rank"]),
        "alpha": float(config["alpha"]),
    }
    if actual != expected:
        raise ValueError(
            f"adapter architecture mismatch: expected {expected}, got {actual}"
        )
    state = torch.load(path / "adapter.pt", map_location="cpu", weights_only=True)
    for key, module in controller.modules.items():
        a_key = f"{key}.lora_a"
        b_key = f"{key}.lora_b"
        if a_key not in state and controller.target in {"q_proj", "k_proj", "v_proj", "o_proj"}:
            layer_index = key.split(".", 1)[0]
            a_key = f"{layer_index}.lora_a"
            b_key = f"{layer_index}.lora_b"
        module.lora_a.data.copy_(state[a_key].to(module.lora_a))
        module.lora_b.data.copy_(state[b_key].to(module.lora_b))
