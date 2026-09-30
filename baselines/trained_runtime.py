"""Single-request inference for saved Qwen3 trained baseline ports.

    runtime = TrainedBaselineRuntime(dict(method="kvlink", checkpoint=checkpoint,
                                         device="cuda", dtype="bfloat16"))
    plan = runtime.plan(row)
    runtime.prepare_request(plan, "kvlink")  # optional: populate CPU RAM entries
    result = runtime.generate(row, "kvlink", 128, prepared_plan=plan)

Encoder-PIC uses config ``model`` (otherwise the saved
``base_model`` is used). KVLink always loads its full model and tokenizer from
the checkpoint, without reinitializing/resizing link embeddings. Loading is
local-only by default. Warm-cache timing is opt-in via require_warm_cache;
ordinary quality calls may build cache misses and do not report TTFT.

The per-runtime RAM cache is keyed by exact artifact token tuples, independently
of document IDs, query, order and position. An LRU tensor-byte budget is set by
``cache_memory_gib`` (default 24); current-request entries cannot be evicted.
Do not mutate model weights or share a runtime between concurrent requests.
"""
from __future__ import annotations

import json
import math
import time
from collections import OrderedDict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from attuner.attention_ops import cache_rope, capture_raw_keys, decoder
from attuner.cache import concat_cache, concat_caches, forward_tokens, move_cache
from attuner.plan import PromptPlan, build_plan
from baselines.encoder_pic.qwen3 import Qwen3EncoderPIC
from baselines.kvlink.qwen3 import add_links
from baselines.progress import phase


class TrainedBaselineRuntime:
    def __init__(self, config):
        self.config = dict(config)
        budget = float(config.get("cache_memory_gib", 24))
        if not math.isfinite(budget) or budget < 0:
            raise ValueError("cache_memory_gib must be finite and nonnegative")
        self._cache_limit_bytes = int(budget * 1024**3)
        self._cache_bytes = 0
        method, checkpoint = config["method"], config["checkpoint"]
        device = config.get("device", "cpu")
        dtype = config.get("dtype", "float32")
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        local_files_only = config.get("local_files_only", True)
        base_model = config.get("model")
        if method not in {"kvlink", "encoder_pic"}:
            raise ValueError("method must be kvlink or encoder_pic")
        self.method = method
        self.checkpoint = str(Path(checkpoint))
        config = json.loads((Path(checkpoint) / "baseline_config.json").read_text())
        if config.get("method") != method:
            raise ValueError("Checkpoint baseline method does not match requested method")
        source = checkpoint if method == "kvlink" else base_model or config.get("base_model")
        if not source:
            raise ValueError("Encoder-PIC requires base_model or a saved base_model path")
        kwargs = dict(local_files_only=local_files_only, attn_implementation="sdpa")
        if dtype is not None:
            kwargs["torch_dtype"] = dtype
        phase("load_tokenizer", source=str(source))
        self.tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=local_files_only)
        phase("load_model_weights", source=str(source))
        self.model = AutoModelForCausalLM.from_pretrained(source, **kwargs)
        phase("move_model_to_device", device=str(device))
        self.model = self.model.to(device).eval()
        if self.model.config.model_type != "qwen3":
            raise ValueError("Trained baseline runtime supports dense Qwen3 only")
        decoder(self.model)  # Reject sliding-window/nonstandard attention caches.
        self.device = self.model.device
        self.encoder_pic = None
        self.link_ids = None
        if method == "kvlink":
            cache_rope(self.model, (), 0)  # Fail early on unsupported RoPE scaling.
            self.link_ids = config["link_ids"]
            count, width = config["max_artifacts"], config["links_per_artifact"]
            if (not isinstance(count, int) or count < 0 or not isinstance(width, int)
                    or width < 1 or len(self.link_ids) != count
                    or any(len(group) != width for group in self.link_ids)):
                raise ValueError("Invalid saved link-token layout")
            flat = [token for group in self.link_ids for token in group]
            if (len(set(flat)) != len(flat) or any(
                    not isinstance(token, int) or not 0 <= token < self.model.config.vocab_size
                    for token in flat)):
                raise ValueError("Invalid saved link-token IDs")
            for i, group in enumerate(self.link_ids):
                names = [f"<|kvlink_{i}_{j}|>" for j in range(width)]
                if self.tokenizer.convert_tokens_to_ids(names) != group:
                    raise ValueError("Checkpoint tokenizer disagrees with saved link-token IDs")
        else:
            layers = config["insertion_layers"]
            if (not layers or any(not isinstance(i, int) for i in layers)
                    or len(set(layers)) != len(layers)):
                raise ValueError("Invalid saved Encoder-PIC insertion layers")
            phase("load_encoder_pic_modules", checkpoint=self.checkpoint)
            self.encoder_pic = Qwen3EncoderPIC(self.model, insertion_layers=layers)
            weights = torch.load(Path(checkpoint) / "encoder_pic.pt", map_location="cpu",
                                 weights_only=True)
            expected = {name for name in self.encoder_pic.state_dict()
                        if not name.startswith("base.")}
            if set(weights) != expected:
                raise ValueError("Encoder-PIC checkpoint must contain exactly the saved new modules; "
                                 f"missing={sorted(expected - set(weights))}, "
                                 f"unexpected={sorted(set(weights) - expected)}")
            self.encoder_pic.load_state_dict(weights, strict=False)
            self.encoder_pic.eval()
        self.model.requires_grad_(False)
        if self.encoder_pic is not None:
            self.encoder_pic.requires_grad_(False)
        eos = self.model.generation_config.eos_token_id
        if eos is None:
            eos = self.tokenizer.eos_token_id
        self.eos_token_ids = set(eos if isinstance(eos, (list, tuple)) else [eos]) - {None}
        self._artifact_cache = OrderedDict()
        self.require_warm_cache = False
        phase("runtime_ready", method=self.method)

    def _sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def plan(self, row):
        return build_plan(self.tokenizer, row,
                          enable_thinking=self.config.get("enable_thinking", False))

    def _check_method(self, method):
        if method is not None and method != self.method:
            raise ValueError("Requested method differs from loaded checkpoint")

    def clear_cache(self):
        """Release all reusable CPU artifact entries."""
        self._artifact_cache.clear()
        self._cache_bytes = 0

    @staticmethod
    def _cache_nbytes(piece):
        return sum(tensor.numel() * tensor.element_size() for layer in piece for tensor in layer)

    def _validate_plan(self, plan):
        if not isinstance(plan, PromptPlan) or not plan.token_ids:
            raise ValueError("Expected a nonempty PromptPlan")
        cursor = 0
        for segment in plan.segments:
            if (not cursor <= segment.start < segment.end <= len(plan.token_ids)
                    or tuple(segment.token_ids) != tuple(plan.token_ids[segment.start:segment.end])):
                raise ValueError("Artifact spans must be ordered, nonempty and match prompt tokens")
            cursor = segment.end
        if self.method == "kvlink" and len(plan.segments) > len(self.link_ids):
            raise ValueError("More artifacts than saved link-token groups")

    @torch.inference_mode()
    def prepare_request(self, plan, method=None):
        """Encode cache misses independently; return per-call build/hit counts.

        KVLink entries hold normalized pre-RoPE keys and values for all decoder
        layers. Encoder-PIC entries hold projected cross-attention K/V for each
        insertion layer, with bidirectional encoding confined to one artifact.
        All persistent tensors are detached CPU copies. Every current-plan key
        is protected, including hits encountered after misses. A request whose
        unique artifacts exceed the tensor-byte budget raises ValueError;
        zero budget permits only requests without artifacts.
        """
        self._check_method(method)
        self._validate_plan(plan)
        protected = {tuple(segment.token_ids) for segment in plan.segments}
        request_bytes = sum(self._cache_nbytes(self._artifact_cache[key])
                            for key in protected if key in self._artifact_cache)
        built = hits = 0
        for segment in plan.segments:
            key = tuple(segment.token_ids)
            if key in self._artifact_cache:
                self._artifact_cache.move_to_end(key)
                hits += 1
                continue
            if getattr(self, 'require_warm_cache', False):
                raise RuntimeError('Measured request encountered a cold artifact cache')
            if self.encoder_pic is None:
                with capture_raw_keys(self.model) as keys:
                    _, piece = forward_tokens(self.model, key, device=self.device,
                                              last_logits_only=True)
                piece = tuple((keys[i], value) for i, (_, value) in enumerate(piece))
            else:
                piece = self.encoder_pic.encode([key])
            size = self._cache_nbytes(piece)
            request_bytes += size
            if request_bytes > self._cache_limit_bytes:
                raise ValueError("Request artifact cache exceeds cache_memory_gib budget: "
                                 f"requires at least {request_bytes} tensor bytes, "
                                 f"budget is {self._cache_limit_bytes}")
            while self._cache_bytes + size > self._cache_limit_bytes:
                victim = next(key for key in self._artifact_cache if key not in protected)
                self._cache_bytes -= self._cache_nbytes(self._artifact_cache.pop(victim))
            self._artifact_cache[key] = tuple(
                tuple(tensor.detach().to("cpu", copy=True) for tensor in layer)
                for layer in piece)
            self._cache_bytes += size
            built += 1
        return dict(artifact_cache_builds=built, artifact_cache_hits=hits,
                    artifact_cache_entries=len(self._artifact_cache),
                    artifact_cache_bytes=self._cache_bytes,
                    artifact_cache_limit_bytes=self._cache_limit_bytes)

    def _kvlink_prefill(self, linked):
        cache, cursor, logits = (), 0, None
        for segment in linked.segments:
            if cursor < segment.start:
                logits, cache = forward_tokens(self.model, linked.token_ids[cursor:segment.start],
                    device=self.device, cache=cache, last_logits_only=True)
            piece = move_cache(self._artifact_cache[tuple(segment.token_ids)], self.device)
            cache = concat_cache(cache, cache_rope(self.model, piece, segment.start))
            cursor = segment.end
        # Every nonempty saved link group is online, including the final one.
        if cursor < len(linked.token_ids):
            logits, cache = forward_tokens(self.model, linked.token_ids[cursor:],
                device=self.device, cache=cache, last_logits_only=True)
        return logits, cache

    @torch.inference_mode()
    def generate(self, row=None, method=None, max_new_tokens=None, *, prepared_plan=None,
                 temperature=0):
        """Return text/raw_output, response_token_ids, prompt_token_ids, metadata.

        Greedy argmax only; generated IDs include a terminal EOS if produced,
        while text skips special tokens. Metadata reports cache work, lengths,
        and termination. Warm-cache calls also report synchronized TTFT. An explicitly
        prepared plan takes precedence over row. Cache misses are built here
        if prepare_request was not called first.
        """
        self._check_method(method)
        if temperature != 0:
            raise ValueError('Trained baseline evaluation uses greedy decoding')
        if max_new_tokens is None:
            max_new_tokens = self.config.get("max_new_tokens", 128)
        if not isinstance(max_new_tokens, int) or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        if prepared_plan is None and row is None:
            raise ValueError("Provide row or prepared_plan")
        plan = prepared_plan if prepared_plan is not None else self.plan(row)
        self._sync()
        started = time.perf_counter() if getattr(self, 'require_warm_cache', False) else None
        self._validate_plan(plan)
        artifact_tokens = sum(segment.token_count for segment in plan.segments)
        if self.method == "kvlink":
            linked = add_links(plan, self.link_ids).plan
            decoder_tokens = len(linked.token_ids)
            online_tokens = decoder_tokens - artifact_tokens
        else:
            # Compact positions match Qwen3EncoderPIC.forward during training.
            online, cursor = [], 0
            for segment in plan.segments:
                online.extend(plan.token_ids[cursor:segment.start])
                cursor = segment.end
            online.extend(plan.token_ids[cursor:])
            decoder_tokens = online_tokens = len(online)
            if not online:
                raise ValueError("Encoder-PIC requires at least one online prompt token")
        capacity = self.model.config.max_position_embeddings
        if capacity and (decoder_tokens + max_new_tokens > capacity or any(
                segment.token_count > capacity for segment in plan.segments)):
            raise ValueError("Prompt/independent artifact plus generation exceeds model context")
        try:
            counts = self.prepare_request(plan)
            if self.encoder_pic is None:
                logits, cache = self._kvlink_prefill(linked)
            else:
                self.encoder_pic._states = concat_caches([
                    move_cache(self._artifact_cache[tuple(segment.token_ids)], self.device)
                    for segment in plan.segments]) or None
                logits, cache = forward_tokens(self.model, online, device=self.device,
                                               last_logits_only=True)
            generated, terminal_eos, finish_reason = [], None, "length"
            first_at = None
            for index in range(max_new_tokens):
                token = int(logits[:, -1, :].argmax(-1).item())
                if index == 0 and started is not None:
                    self._sync()
                    first_at = time.perf_counter()
                generated.append(token)
                if token in self.eos_token_ids:
                    terminal_eos, finish_reason = token, "stop"
                    break
                if index + 1 < max_new_tokens:
                    logits, cache = forward_tokens(self.model, [token], device=self.device,
                                                   cache=cache, last_logits_only=True)
            text = self.tokenizer.decode(generated, skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False)
            timing = {} if started is None else dict(
                ttft_seconds=first_at-started, cache_build_seconds=0.0)
            return dict(text=text, raw_output=text, response_token_ids=generated,
                        output_tokens=len(generated), **timing,
                        prompt_token_ids=list(plan.token_ids),
                        metadata=dict(method=self.method, checkpoint=self.checkpoint,
                            quality_only=started is None, cache_storage="cpu_ram",
                            prompt_tokens=len(plan.token_ids), decoder_prompt_tokens=decoder_tokens,
                            online_prompt_tokens=online_tokens, artifact_tokens=artifact_tokens,
                            output_tokens=len(generated), generation_temperature=0.0,
                            finish_reason=finish_reason, terminal_eos_token_id=terminal_eos,
                            **counts))
        finally:
            if self.encoder_pic is not None:
                self.encoder_pic.clear_context()


class TrainedComparisonRuntime:
    """One trained checkpoint plus a separate unmodified Full reference.

    Separate models keep Encoder-PIC hooks and KVLink's learned base weights
    from changing the Full reference. Calls are sequential within a GPU worker.
    """
    def __init__(self, config, method, checkpoint, include_full=True):
        from baselines.qwen3_runtime import Qwen3BaselineRuntime
        self.trained = TrainedBaselineRuntime(dict(config, method=method, checkpoint=checkpoint))
        self.full = Qwen3BaselineRuntime(config) if include_full else None
        self.model_name = Path(config['model']).name
        self.require_warm_cache = False

    def plan(self, row):
        plan = self.trained.plan(row)
        if self.full is not None and self.full.plan(row).token_ids != plan.token_ids:
            raise ValueError('Trained checkpoint and Full tokenize the request differently')
        return plan

    def clear_cache(self):
        self.trained.clear_cache()
        if self.full is not None:
            self.full.clear_cache()

    def _sync(self):
        self.trained._sync()

    def _runtime(self, method):
        if method == self.trained.method:
            return self.trained
        if method == 'full' and self.full is not None:
            return self.full
        raise ValueError('Unsupported method in trained comparison: '+method)

    def prepare_request(self, plan, method):
        return self._runtime(method).prepare_request(plan, method)

    def generate(self, row, method, **kwargs):
        runtime = self._runtime(method)
        runtime.require_warm_cache = self.require_warm_cache
        try:
            return runtime.generate(row, method, **kwargs)
        finally:
            runtime.require_warm_cache = False
