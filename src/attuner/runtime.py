"""Unified causal, independent-artifact and gated Attuner runtime."""
from __future__ import annotations

import contextlib
import os
import re
import time
from collections import OrderedDict
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as transformers_version

from .cache import forward_tokens, concat_caches, cache_length, move_cache
from .plan import build_plan
from .lora import install_q_lora, load_adapter


def option(config, name, default=None):
    return config.get(name, default) if isinstance(config, dict) else getattr(config, name, default)


def model_files_info(model_path):
    """Describe local model files using readable filesystem metadata."""
    root = Path(model_path).resolve()
    files = {}
    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        name = path.name
        if (name.endswith((".safetensors", ".bin", ".json", ".model", ".tiktoken")) or
                name in {"merges.txt", "vocab.txt"}):
            stat = path.stat()
            files[name] = dict(bytes=stat.st_size, modified_ns=stat.st_mtime_ns,
                               changed_ns=stat.st_ctime_ns, inode=stat.st_ino, device=stat.st_dev)
    if not any(name.endswith((".safetensors", ".bin")) for name in files):
        raise ValueError("Model path has no local weight files")
    return files


class Runtime:
    def __init__(self, config):
        self.config = config
        model_path = str(option(config, "model"))
        dtype_name = option(config, "dtype", "bfloat16")
        dtype = getattr(torch, dtype_name) if isinstance(dtype_name, str) else dtype_name
        self.device = torch.device(option(config, "device", "cuda"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, local_files_only=True,
            attn_implementation=option(config, "attention_backend", "sdpa"),
            low_cpu_mem_usage=True).to(self.device).eval()
        cfg = self.model.config
        required = ("num_hidden_layers", "num_attention_heads", "hidden_size")
        if any(not hasattr(cfg, key) for key in required):
            raise ValueError("Model config lacks decoder attention dimensions")
        if getattr(cfg, "sliding_window", None) and getattr(cfg, "use_sliding_window", True):
            raise ValueError("Sliding-window caches need a model-specific validated backend")
        self.model_name = Path(model_path.rstrip("/")).name
        self.architecture = dict(model_name=self.model_name, model_type=cfg.model_type,
            num_hidden_layers=cfg.num_hidden_layers, num_attention_heads=cfg.num_attention_heads,
            num_key_value_heads=getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
            head_dim=getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads),
            rope_theta=getattr(cfg, "rope_theta", None), rope_scaling=getattr(cfg, "rope_scaling", None),
            rope_parameters=getattr(cfg, "rope_parameters", None))
        self.model_identifier = (
            f"{Path(model_path).resolve()}|{cfg.model_type}|L{cfg.num_hidden_layers}|"
            f"H{cfg.num_attention_heads}|T{len(self.tokenizer)}|{dtype}"
        )
        self.model_files = model_files_info(model_path)
        self.runtime_versions = dict(torch=str(torch.__version__), transformers=transformers_version)
        self.cache_identity = dict(model=self.model_identifier, model_files=self.model_files,
            runtime_versions=self.runtime_versions,
            architecture=self.architecture, attention_backend=option(config, 'attention_backend', 'sdpa'))
        self.model._attuner_model_identifier = self.model_identifier
        self.controller = None
        self._resident_cache = OrderedDict()
        self._resident_bytes = 0
        self._resident_limit = int(float(option(config, 'cache_memory_gib', 0)) * 1024**3)
        adapter = option(config, "adapter", None)
        if adapter:
            self.controller = load_adapter(self.model, Path(adapter))
        self.cache_dir = option(config, "cache_dir", None)
        if self.cache_dir:
            self.cache_dir = Path(self.cache_dir) / self.model_name
            # Multiple distributed ranks can create the same NFS parent at once.
            # Some NFS clients still surface FileExistsError with exist_ok=True.
            for attempt in range(20):
                try:
                    self.cache_dir.mkdir(parents=True, exist_ok=True)
                    break
                except FileExistsError:
                    if self.cache_dir.is_dir():
                        break
                    if attempt == 19:
                        raise
                    time.sleep(0.1)
            from .cache import bind_cache_directory
            bind_cache_directory(self.cache_dir, self.cache_identity)
        eos = self.model.generation_config.eos_token_id
        if eos is None:
            eos = self.tokenizer.eos_token_id
        self.eos_token_ids = set(eos if isinstance(eos, list) else [eos]) - {None}

    def install_adapter(self, *, layer_start=None, layer_end=None, rank=8, alpha=16):
        depth = self.architecture["num_hidden_layers"]
        self.controller = install_q_lora(self.model, layer_start=depth // 2 if layer_start is None else layer_start,
            layer_end=depth - 1 if layer_end is None else layer_end, rank=rank, alpha=alpha)
        return self.controller

    def plan(self, row):
        return build_plan(self.tokenizer, row, enable_thinking=option(self.config, "enable_thinking", False))

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _controller(self):
        return self.controller or getattr(self.model, "_kv_order_lora_controller", None)

    def _forward(self, tokens, *, cache=(), position_start=None, mask=None, training=False):
        controller = self._controller()
        gating = contextlib.nullcontext()
        if controller:
            gating = controller.disabled() if mask is False else (
                controller.masked_tokens(torch.tensor([mask], dtype=torch.bool, device=self.device))
                if mask is not None else contextlib.nullcontext())
        with torch.set_grad_enabled(training), gating:
            return forward_tokens(self.model, tokens, device=self.device, cache=cache,
                                  position_start=position_start, last_logits_only=True)

    def _warm(self, plan, method="pic"):
        caches, elapsed = [], 0.0
        self._cache_read_seconds = 0.0
        for segment in plan.segments:
            position = segment.start if method == "position_oracle" else 0
            raw_keys = method == "raw_keys"
            key = ("raw" if raw_keys else "rotated", segment.token_ids, position)
            safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", segment.document_id)[:120]
            path = self.cache_dir / f"artifact-{safe_id}-p{position}-{'raw' if raw_keys else 'rotated'}.pt" if self.cache_dir else None
            if getattr(self, "require_warm_cache", False) and key not in self._resident_cache:
                raise RuntimeError("Measured request missed CPU RAM cache; increase cache_memory_gib")
            if key in self._resident_cache:
                read_started = time.perf_counter()
                cache = self._resident_cache.pop(key)
                self._resident_cache[key] = cache
                self._cache_read_seconds += time.perf_counter() - read_started
            elif path and path.exists():
                read_started = time.perf_counter()
                payload = torch.load(path, map_location="cpu", weights_only=True)
                if (payload.get("cache_identity") != self.cache_identity or
                        tuple(payload["token_ids"]) != segment.token_ids):
                    raise ValueError(f"Cache artifact content differs: {path}")
                cache = payload["cache"]
                self._cache_read_seconds += time.perf_counter() - read_started
            else:
                self._sync()
                start = time.perf_counter()
                if raw_keys:
                    from .attention_ops import capture_raw_keys
                    with capture_raw_keys(self.model) as keys:
                        _, cache = self._forward(segment.token_ids, position_start=position, mask=False)
                    if len(keys) != len(cache):
                        raise RuntimeError("Incomplete pre-RoPE K capture")
                    cache = tuple((keys[i], value) for i, (_, value) in enumerate(cache))
                else:
                    _, cache = self._forward(segment.token_ids, position_start=position, mask=False)
                self._sync()
                cache = move_cache(cache, "cpu", detach=True)
                if path:
                    import os
                    temporary = path.with_suffix(f".{os.getpid()}.tmp")
                    torch.save({"token_ids": list(segment.token_ids), "cache": cache,
                                "cache_identity": self.cache_identity}, temporary)
                    temporary.replace(path)
                elapsed += time.perf_counter() - start
            self.remember_cache(key, cache)
            caches.append(cache)
        return caches, elapsed

    def remember_cache(self, key, cache):
        if not self._resident_limit or key in self._resident_cache:
            return
        size = sum(t.numel()*t.element_size() for layer in cache for t in layer)
        while self._resident_cache and self._resident_bytes + size > self._resident_limit:
            _, evicted = self._resident_cache.popitem(last=False)
            self._resident_bytes -= sum(t.numel()*t.element_size() for layer in evicted for t in layer)
        if size <= self._resident_limit:
            self._resident_cache[key] = cache
            self._resident_bytes += size

    def clear_cache(self):
        self._resident_cache.clear()
        self._resident_bytes = 0

    def prepare_request(self, plan, method):
        """Build only reusable CPU cache pieces outside the measured interval."""
        if method in {"pic", "pic_attuner", "position_oracle"}:
            self._warm(plan, method)

    def _assemble(self, plan, caches, *, training=False, adapter=False, raw_keys=False, require_logits=True):
        if len(caches) != len(plan.segments):
            raise ValueError("Artifact cache count mismatch")
        cache, cursor, logits, load_time = (), 0, None, 0.0
        index = 0
        while index < len(plan.segments):
            segment = plan.segments[index]
            if segment.start > cursor:
                mask = plan.adapter_mask[cursor:segment.start] if adapter else False
                logits, cache = self._forward(plan.token_ids[cursor:segment.start], cache=cache,
                    position_start=cursor, mask=mask, training=training)
            self._sync()
            start = time.perf_counter()
            parts = [cache]
            group_end = segment.end
            while index < len(plan.segments):
                current = plan.segments[index]
                piece = move_cache(caches[index], self.device)
                if raw_keys:
                    from .attention_ops import cache_rope
                    piece = cache_rope(self.model, piece, current.start)
                parts.append(piece)
                group_end = current.end
                index += 1
                if index == len(plan.segments) or plan.segments[index].start != group_end:
                    break
            cache = concat_caches(parts)
            del parts
            self._sync()
            load_time += time.perf_counter() - start
            if cache_length(cache) != group_end:
                raise RuntimeError("Assembled cache position mismatch")
            cursor = group_end
        if cursor < len(plan.token_ids):
            mask = plan.adapter_mask[cursor:] if adapter else False
            logits, cache = self._forward(plan.token_ids[cursor:], cache=cache,
                position_start=cursor, mask=mask, training=training)
        if logits is None and require_logits:
            raise RuntimeError("Prompt requires a normally prefetched question/suffix")
        return logits, cache, load_time

    def prefill(self, plan, method, training=False):
        method = {"noheal": "pic", "no_heal": "pic", "full+attuner": "full_attuner",
                  "pic-pos": "position_oracle", "pic_pos": "position_oracle",
                  "pic+attuner": "pic_attuner"}.get(method.lower(), method.lower())
        if method not in {"full", "pic", "position_oracle", "full_attuner", "pic_attuner"}:
            raise ValueError(f"Unknown method: {method}")
        adapter = method.endswith("attuner")
        if adapter and self._controller() is None:
            raise ValueError("Attuner method requires a model-specific adapter")
        reuse = method in {"pic", "position_oracle", "pic_attuner"}
        caches, build_seconds = self._warm(plan, method) if reuse else ([], 0.0)
        self._sync()
        started = time.perf_counter()
        read_seconds = getattr(self, "_cache_read_seconds", 0.0) if reuse else 0.0
        if reuse:
            logits, cache, load_seconds = self._assemble(plan, caches, training=training, adapter=adapter)
        else:
            logits, cache = self._forward(plan.token_ids, position_start=0,
                mask=plan.adapter_mask if adapter else False, training=training)
            load_seconds = 0.0
        self._sync()
        return logits, cache, dict(cache_build_seconds=build_seconds, kv_load_seconds=load_seconds + read_seconds,
            reuse_tokens=sum(s.token_count for s in plan.segments) if reuse else 0,
            recompute_tokens=0, online_prefill_seconds=time.perf_counter() - started + read_seconds,
            online_started=started - read_seconds)

    def generate(self, row, method, max_new_tokens=None, temperature=None, stop=None,
                 prepared_plan=None, **kwargs):
        maximum = int(max_new_tokens if max_new_tokens is not None else option(self.config, "max_new_tokens", 512))
        temperature = float(temperature if temperature is not None else option(self.config, "temperature", 0.0))
        if maximum <= 0 or temperature < 0:
            raise ValueError("max_new_tokens must be positive; temperature nonnegative")
        if kwargs:
            raise ValueError(f"Unsupported generation parameters: {sorted(kwargs)}")
        plan = prepared_plan if prepared_plan is not None else self.plan(row)
        capacity = int(getattr(self.model.config, 'max_position_embeddings', 0))
        if capacity and len(plan.token_ids) + maximum > capacity:
            raise ValueError(f'Prompt plus generation budget exceeds configured model context: '
                             f'{len(plan.token_ids)} + {maximum} > {capacity}')
        # Strict RAM-only measurement includes all lookups/layout work, not an
        # estimate obtained by subtracting cache-build time from a cold call.
        self._sync()
        measured_start = time.perf_counter() if getattr(self, "require_warm_cache", False) else None
        logits, cache, timing = self.prefill(plan, method)
        if measured_start is not None:
            if timing["cache_build_seconds"] != 0:
                raise RuntimeError("Cache construction occurred in measured TTFT")
            timing["online_started"] = measured_start
            timing["online_prefill_seconds"] = time.perf_counter() - measured_start
        online_started = timing.pop("online_started")
        next_position = timing.pop("logical_next_position", len(plan.token_ids))
        attention_context = contextlib.nullcontext()
        stop_values = option(self.config, "stop", []) if stop is None else stop
        if isinstance(stop_values, str):
            stop_values = [stop_values]
        if any(not isinstance(s, str) or not s for s in (stop_values or [])):
            raise ValueError("Stop strings must be nonempty strings")
        generated, first_at, finish_reason = [], None, "length"
        terminal_eos, stopped_text, terminal_stop = None, None, None
        with torch.no_grad(), attention_context:
            for index in range(maximum):
                scores = logits[:, -1, :]
                token = int(scores.argmax(-1).item()) if temperature == 0 else int(
                    torch.multinomial(torch.softmax(scores.float() / temperature, -1), 1).item())
                if first_at is None:
                    self._sync()
                    first_at = time.perf_counter()
                generated.append(token)
                if token in self.eos_token_ids:
                    terminal_eos = token
                    finish_reason = "stop"
                    break
                if stop_values:
                    decoded = self.tokenizer.decode(generated, skip_special_tokens=True,
                                                    clean_up_tokenization_spaces=False)
                    matches = [(decoded.find(s), s) for s in stop_values if s in decoded]
                    if matches:
                        start, terminal_stop = min(matches, key=lambda pair: pair[0])
                        stopped_text = decoded[:start]
                        # Preserve the actual greedy trajectory: never retokenize a
                        # stop string or cut token IDs at an ambiguous text boundary.
                        finish_reason = "stop"
                        break
                if index + 1 < maximum:
                    logits, cache = self._forward([token], cache=cache,
                        position_start=next_position + index,
                        mask=None if "attuner" in method.lower() else False)
        self._sync()
        finished = time.perf_counter()
        text = self.tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if stopped_text is not None:
            text = stopped_text
        return dict(raw_output=text, response_token_ids=generated, output_tokens=len(generated),
            terminal_eos_token_id=terminal_eos, terminal_stop_string=terminal_stop,
            generation_temperature=temperature,
            ttft_seconds=first_at - online_started, e2e_seconds=finished - online_started,
            prompt_tokens=len(plan.token_ids), prompt_token_ids=list(plan.token_ids),
            message_count=len(row["messages"]), finish_reason=finish_reason, **timing)
