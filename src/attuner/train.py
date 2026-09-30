"""Pure Full-teacher/PIC-student KL on identical Full greedy response prefixes."""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist

from .cache import teacher_forced_logits
from .io import append_jsonl, comparable_identity, ensure_manifest, exact_coverage, read_jsonl, write_json
from .lora import save_adapter
from .runtime import Runtime


def batch_settings(settings, world):
    micro = int(settings.get("micro_batch_size", 1))
    accumulation = int(settings.get("grad_accumulation", 4))
    global_size = int(settings.get("global_batch_size", micro * accumulation * world))
    if min(micro, accumulation, global_size, world) <= 0:
        raise ValueError("Batch sizes, accumulation and world size must be positive")
    if "micro_batch_size" in settings and global_size != micro * accumulation * world:
        raise ValueError("global_batch_size must equal micro_batch_size * GPU count * grad_accumulation")
    return micro, global_size


def prepare_training_sample(runtime, config, row, trajectory, epoch, seed):
    settings = config["training"]
    if trajectory.get("model_identifier") != runtime.model_identifier:
        raise ValueError("Teacher response comes from a different base model")
    if trajectory.get("model") not in {runtime.model_name, str(config.get("model"))}:
        raise ValueError("Teacher rollout declares a different model")
    if trajectory.get("method") != "full" or trajectory.get("generation_temperature") != 0.0:
        raise ValueError("Training requires Full causal greedy trajectories")
    response = list(trajectory["response_token_ids"])
    if not response:
        raise ValueError("Full teacher response prefix must be nonempty")
    terminal_eos = trajectory.get("terminal_eos_token_id")
    if terminal_eos is not None and (response[-1] != terminal_eos or terminal_eos not in runtime.eos_token_ids):
        raise ValueError("Rollout terminal EOS differs from its actual token prefix")
    if trajectory.get("finish_reason") == "stop" and terminal_eos is None and not trajectory.get("terminal_stop_string"):
        raise ValueError("Stopped rollout does not identify its terminal condition")
    teacher_plan = runtime.plan(row)
    if trajectory.get("prompt_token_ids") != list(teacher_plan.token_ids):
        raise ValueError("Teacher rollout and training prompt differ")
    return teacher_plan, teacher_plan, response, list(range(len(teacher_plan.segments)))



def sample_losses(runtime, teacher_plans, student_plans, responses, micro_batch_size):
    if micro_batch_size > 1:
        from .microbatch import distillation_losses
        return distillation_losses(runtime, teacher_plans, student_plans, responses)
    teacher_plan, student_plan, response = teacher_plans[0], student_plans[0], responses[0]
    with torch.no_grad(), runtime.controller.disabled():
        tl, tc, _ = runtime.prefill(teacher_plan, "full")
        teacher = teacher_forced_logits(runtime.model, tl, tc, response,
            device=runtime.device, prompt_length=len(teacher_plan.token_ids)).float()
        log_teacher = teacher.log_softmax(-1).detach()
        del teacher, tc, tl
    sl, sc, _ = runtime.prefill(student_plan, "pic_attuner", training=True)
    student = teacher_forced_logits(runtime.model, sl, sc, response,
        device=runtime.device, prompt_length=len(student_plan.token_ids))
    log_student = student.float().log_softmax(-1)
    return (log_teacher.exp() * (log_teacher - log_student)).sum(-1).mean().reshape(1)


def _save_state(path, state):
    temp = path.with_suffix(f".{os.getpid()}.tmp")
    with temp.open("wb") as handle:
        torch.save(state, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def _rng_state(device):
    return dict(torch=torch.get_rng_state(), python=random.getstate(),
                cuda=torch.cuda.get_rng_state(device) if device.type == "cuda" else None)


def _restore_rng(state, device):
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"], device)


def _gpu_utilization(device):
    if device.type != "cuda":
        return None
    try:
        return torch.cuda.utilization(device)
    except (ImportError, ModuleNotFoundError, RuntimeError, AttributeError):
        return None


def _reconcile_log(path, committed_update, pending):
    """The atomic checkpoint commits an update; its journal repairs interrupted writes."""
    rows = read_jsonl(path, recover_tail=True)
    rows = [row for row in rows if row["update"] <= committed_update]
    def key(row):
        return (row["update"], row.get("epoch"), row.get("instance_id"))
    indexed = {key(row): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"Duplicate committed training log records: {path}")
    for row in pending:
        existing = indexed.get(key(row))
        if existing is not None and existing != row:
            raise ValueError(f"Training journal conflicts with committed log: {path}")
        if existing is None:
            rows.append(row)
            indexed[key(row)] = row
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    return rows


def train(config, dataset, examples, trajectories, output):
    settings = config["training"]
    artifact_order = settings.get("artifact_order", "official")
    if artifact_order != "official":
        raise ValueError(f"Unsupported training artifact_order: {artifact_order}")
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    micro_batch_size, batch_size = batch_settings(settings, world)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if world > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
    seed = int(config.get("seed", 20260907))
    torch.manual_seed(seed)
    random.seed(seed)
    cfg = {**config, "device": f"cuda:{local_rank}" if world > 1 else config.get("device", "cuda"),
           "adapter": None}
    runtime = Runtime(cfg)
    depth = runtime.architecture["num_hidden_layers"]
    layer_start = settings.get("layer_start")
    layer_end = settings.get("layer_end")
    layer_start = depth // 2 if layer_start is None else int(layer_start)
    layer_end = depth - 1 if layer_end is None else int(layer_end)
    lora_rank = int(settings.get("rank", 8))
    alpha = float(settings.get("alpha", 2 * lora_rank))
    controller = runtime.install_adapter(layer_start=layer_start, layer_end=layer_end,
                                         rank=lora_rank, alpha=alpha)
    params = list(controller.parameters())
    optimizer = torch.optim.AdamW(params, lr=float(settings["learning_rate"]),
                                 weight_decay=float(settings.get("weight_decay", 0.0)))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    ids = [r["instance_id"] for r in examples]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Training examples must be nonempty with unique IDs")
    rolled = exact_coverage(trajectories, ids)
    identity = dict(model=runtime.model_identifier, dataset=dataset,
        data_ids=ids, response_ids=[row["instance_id"] for row in trajectories],
        settings={k: v for k, v in settings.items() if k != "mixed_data_manifest"},
        seed=seed, world_size=world,
        objective="KL(p_full || p_attuner)", teacher="full",
        student="pic", response_prefix="same full greedy", artifact_order=artifact_order)
    validation_error = None
    if rank == 0:
        try:
            ensure_manifest(output, identity, {"architecture": runtime.architecture,
                                              "trainable_parameters": sum(p.numel() for p in params)})
        except Exception as exc:
            validation_error = f'{type(exc).__name__}: {exc}'
    if world > 1:
        status = [validation_error]
        dist.broadcast_object_list(status, src=0)
        validation_error = status[0]
    if validation_error:
        if world > 1:
            dist.destroy_process_group()
        raise ValueError(validation_error)
    checkpoint = output / "resume.pt"
    cursor, updates, elapsed_before, loss_sum, sample_count = 0, 0, 0.0, 0.0, 0
    state, peak_before = None, 0
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if comparable_identity(state["identity"]) != comparable_identity(identity):
            raise ValueError("Training checkpoint identity mismatch")
        for p, tensor in zip(params, state["parameters"], strict=True):
            p.data.copy_(tensor.to(p))
        optimizer.load_state_dict(state["optimizer"])
        cursor, updates, elapsed_before = state["cursor"], state["updates"], state["elapsed"]
        loss_sum, sample_count = state["loss_sum"], state["sample_count"]
        _restore_rng(state["rank_states"][rank]["rng"], runtime.device)
        peak_before = state["rank_states"][rank]["peak_memory_bytes"]
    if (output / "COMPLETE").exists():
        from .io import verified_complete
        verified_complete(output, identity)
        if world > 1:
            dist.destroy_process_group()
        return
    journal = state["rank_states"][rank]["records"] if state else []
    committed_samples = _reconcile_log(output / f"train_loss.rank{rank}.jsonl", updates, journal)
    if rank == 0:
        update_rows = _reconcile_log(output / "predictions.jsonl", updates,
                                    [state["update_record"]] if state else [])
        if [row["update"] for row in update_rows] != list(range(1, updates + 1)):
            raise ValueError("Committed optimizer update log has gaps or is reordered")
    epochs = int(settings.get("epochs", 2))
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    order = []
    batches = []
    for epoch in range(epochs):
        indices = list(range(len(examples)))
        random.Random(seed + epoch).shuffle(indices)
        epoch_order = [(epoch, i) for i in indices]
        order.extend(epoch_order)
        for start in range(0, len(epoch_order), batch_size):
            batches.append(epoch_order[start:start + batch_size])
    if not 0 <= cursor <= len(order) or cursor != sample_count:
        raise ValueError("Training checkpoint cursor/sample count is invalid")
    if not 0 <= updates <= len(batches):
        raise ValueError("Training checkpoint update count is invalid")
    consumed_batches = batches[:updates]
    if cursor != sum(len(batch) for batch in consumed_batches):
        raise ValueError("Checkpoint does not lie at an epoch-safe optimizer boundary")
    expected_committed = [item for batch in consumed_batches
                          for item in batch[rank::world]]
    expected_keys = [(epoch, examples[index]["instance_id"]) for epoch, index in expected_committed]
    if [(r["epoch"], r["instance_id"]) for r in committed_samples] != expected_keys:
        raise ValueError("Committed per-rank sample log differs from checkpoint cursor")
    started = time.perf_counter()
    elapsed = elapsed_before
    record = state["update_record"] if state else None
    while updates < len(batches):
        batch = batches[updates]
        optimizer.zero_grad(set_to_none=True)
        local_loss = 0.0
        local_records = []
        local_batch = batch[rank::world]
        for start in range(0, len(local_batch), micro_batch_size):
            items = local_batch[start:start + micro_batch_size]
            prepared = [prepare_training_sample(runtime, config, examples[index],
                rolled[examples[index]["instance_id"]], epoch, seed) for epoch, index in items]
            losses = sample_losses(runtime, [x[0] for x in prepared],
                [x[1] for x in prepared], [x[2] for x in prepared], micro_batch_size)
            if not torch.isfinite(losses).all():
                raise RuntimeError("Nonfinite micro-batch KL")
            # SUM across ranks below produces the example mean, including a
            # final incomplete global batch and ranks with no remaining rows.
            (losses.sum() / len(batch)).backward()
            values = losses.detach().cpu().tolist()
            local_loss += sum(values)
            for (epoch, index), info, value in zip(items, prepared, values):
                row = examples[index]
                teacher_plan, student_plan, response, permutation = info
                local_records.append(dict(
                    update=updates + 1, epoch=epoch, instance_id=row["instance_id"], loss=value,
                    source_dataset=row.get("source_dataset", row.get("dataset", dataset)),
                    response_tokens=len(response), elapsed_seconds=elapsed_before + time.perf_counter()-started,
                    artifact_order=artifact_order, permutation=permutation,
                    teacher_prompt_tokens=len(teacher_plan.token_ids),
                    student_prompt_tokens=len(student_plan.token_ids),
                    micro_batch_size=len(items),
                    gpu_utilization_percent=_gpu_utilization(runtime.device),
                    gpu_memory_allocated_bytes=torch.cuda.memory_allocated(runtime.device) if runtime.device.type == "cuda" else 0,
                    gpu_peak_memory_bytes=torch.cuda.max_memory_allocated(runtime.device) if runtime.device.type == "cuda" else 0))
            del losses, prepared
        if world > 1:
            for p in params:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            loss_tensor = torch.tensor(local_loss, dtype=torch.float64, device=runtime.device)
            dist.all_reduce(loss_tensor)
            local_loss = float(loss_tensor)
        grad_norm = torch.nn.utils.clip_grad_norm_(params, float(settings.get("max_grad_norm", 1.0)),
                                                  error_if_nonfinite=True)
        optimizer.step()
        cursor += len(batch)
        updates += 1
        loss_sum += local_loss
        sample_count += len(batch)
        runtime._sync()
        elapsed = elapsed_before + time.perf_counter() - started
        local_state = dict(rng=_rng_state(runtime.device), records=local_records,
            peak_memory_bytes=max(peak_before, torch.cuda.max_memory_allocated(runtime.device)
                                  if runtime.device.type == "cuda" else 0))
        rank_states = [None] * world
        if world > 1:
            dist.all_gather_object(rank_states, local_state)
        else:
            rank_states[0] = local_state
        record = dict(update=updates, consumed_samples=cursor, total_samples=len(order),
            mean_kl=local_loss / len(batch), elapsed_seconds=elapsed, gradient_norm=float(grad_norm),
            learning_rate=optimizer.param_groups[0]["lr"], samples_in_update=len(batch),
            global_batch_size=batch_size, micro_batch_size=micro_batch_size,
            response_tokens=sum(r["response_tokens"] for s in rank_states for r in s["records"]),
            peak_memory_bytes_by_rank=[s["peak_memory_bytes"] for s in rank_states])
        if rank == 0:
            _save_state(checkpoint, dict(identity=identity, cursor=cursor,
                updates=updates, elapsed=elapsed, parameters=[p.detach().cpu() for p in params],
                optimizer=optimizer.state_dict(), rank_states=rank_states, loss_sum=loss_sum,
                sample_count=sample_count, update_record=record))
        if world > 1:
            dist.barrier()
        for sample_record in local_records:
            append_jsonl(output / f"train_loss.rank{rank}.jsonl", sample_record)
        if rank == 0:
            append_jsonl(output / "predictions.jsonl", record)
            print(record, flush=True)
            is_epoch_end = (updates == len(batches) or
                            batches[updates][0][0] != batch[0][0])
            if is_epoch_end:
                completed_epoch = batch[0][0] + 1
                epoch_output = output / f"checkpoint-epoch{completed_epoch}"
                save_adapter(epoch_output, controller, layer_start=layer_start,
                             layer_end=layer_end, rank=lora_rank, alpha=alpha)
                write_json(epoch_output / "summary.json", dict(
                    dataset=dataset, model=runtime.model_name,
                    training_examples=len(examples), completed_epochs=completed_epoch,
                    optimizer_updates=updates, consumed_samples=sample_count,
                    mean_kl=loss_sum / sample_count, lora_rank=lora_rank,
                    layers=[layer_start, layer_end], target="q_proj",
                    objective="KL(p_full || p_attuner)",
                    artifact_order=artifact_order, order_seed=seed))
    if rank == 0:
        elapsed = elapsed_before + time.perf_counter() - started
        save_adapter(output / "adapter", controller, layer_start=layer_start,
                     layer_end=layer_end, rank=lora_rank, alpha=alpha)
        summary = dict(dataset=dataset, model=runtime.model_name, training_examples=len(examples),
            epochs=epochs, optimizer_updates=updates, mean_kl=loss_sum/sample_count,
            training_seconds=elapsed, gpu_count=world, gpu_seconds=elapsed*world,
            peak_memory_bytes_by_rank=record["peak_memory_bytes_by_rank"] if updates and cursor else [],
            peak_memory_bytes=max(record["peak_memory_bytes_by_rank"]),
            lora_rank=lora_rank, layers=[layer_start, layer_end], target="q_proj",
            objective="KL(p_full || p_attuner)", artifact_order=artifact_order,
            order_seed=seed, ce_weight=0, bridge_tokens=0)
        from collections import Counter
        summary["source_counts"] = dict(Counter(
            row.get("source_dataset", row.get("dataset", dataset)) for row in examples))
        summary["loss_reduction"] = "response-token mean per example, then example mean"
        summary["weight_decay"] = optimizer.param_groups[0]["weight_decay"]
        summary["micro_batch_size"] = micro_batch_size
        summary["global_batch_size"] = batch_size
        write_json(output / "summary.json", summary)
        write_json(output / "paired_analysis.json", {"applicable": False,
                                                     "reason": "training run; paired quality in evaluation"})
        from .io import complete
        complete(output)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


def main():
    import argparse
    from .config import load, SOURCES
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/qwen3_4b.yaml')
    p.add_argument('--model', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data', type=Path, default=Path('data/prepared/mixed12k/train.jsonl'))
    p.add_argument('--teacher-dir', type=Path, default=Path('data/teachers'))
    p.add_argument('--limit', type=int, help='Short training smoke test; use a separate output directory')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    cfg = load(args.config)
    cfg['model'] = args.model
    world = int(os.environ.get('WORLD_SIZE', 1))
    batch = cfg['training']['global_batch_size']
    micro = cfg['training']['micro_batch_size']
    if batch % (micro * world):
        p.error('Global batch must be divisible by micro-batch times GPU count')
    cfg['training']['grad_accumulation'] = batch // (micro * world)
    if args.dry_run:
        print(json.dumps(cfg['training'], indent=2))
        return
    rows = read_jsonl(args.data)
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        p.error('No training data: run python -m attuner.prepare --download first')
    required = {r['instance_id'] for r in rows}
    teachers = [r for source in SOURCES for r in read_jsonl(
        args.teacher_dir / Path(args.model).name / f'{source}.jsonl')
        if r['instance_id'] in required]
    exact_coverage(teachers, required)
    train(cfg, 'mixed12k', rows, teachers, args.output)


if __name__ == '__main__':
    main()
