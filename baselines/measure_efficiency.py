"""One-GPU, first-token-only measurements, separate from quality generation.

Speedups require matched Full measurements in the same output directory.
Saved timings belong to this execution protocol; concurrent quality runs
must not use them as latency data.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]


def emit(event, **values):
    print(json.dumps(dict(time=time.strftime('%F %T'), event=event, **values)), flush=True)


def fixed_prefix(plans):
    """Only a token-identical causal prefix before the first artifact is reusable."""
    common = None
    for plan in plans:
        prefix = plan.token_ids[:plan.segments[0].start] if plan.segments else ()
        if common is None:
            common = prefix
        else:
            end = min(len(common), len(prefix))
            index = 0
            while index < end and common[index] == prefix[index]:
                index += 1
            common = common[:index]
        if not common:
            return ()
    return tuple(common or ())


def gpu_snapshot(gpu):
    try:
        return subprocess.check_output(['nvidia-smi', '-i', str(gpu),
            '--query-gpu=uuid,memory.used,memory.free,utilization.gpu,utilization.memory',
            '--format=csv,noheader,nounits'], text=True, timeout=5).strip()
    except (OSError, subprocess.SubprocessError):
        return 'unavailable'


def summarize_speedups(output, jobs):
    """Derive ratios only from matched Full timings in this output directory."""
    from attuner.io import read_jsonl, write_json
    for dataset, method in jobs:
        full_path = output / dataset / 'full' / 'timings.jsonl'
        out = output / dataset / method
        if not full_path.exists() or not (out / 'summary.json').exists():
            continue
        full = read_jsonl(full_path)
        rows = read_jsonl(out / 'timings.jsonl')
        reference = {r['instance_id']: r for r in full}
        if len(reference) != len(full) or {r['instance_id'] for r in rows} != set(reference):
            raise ValueError('Cannot compare TTFT on different cohorts')
        if any(r['gpu_uuid'] != reference[r['instance_id']]['gpu_uuid'] for r in rows):
            raise ValueError('Full and method must use the same per-request GPU assignment')
        summary = json.loads((out / 'summary.json').read_text())
        summary['speedup'] = statistics.median(r['ttft_ms'] for r in full) / statistics.median(r['ttft_ms'] for r in rows)
        write_json(out / 'summary.json', summary)


def merge_shards(output, jobs, cfg, shards, limit):
    from attuner.io import read_jsonl, write_json
    from attuner.runner import data_path
    for dataset, method in jobs:
        examples = read_jsonl(data_path(cfg, dataset))
        if limit:
            examples = examples[:limit]
        ids = [r['instance_id'] for r in examples]
        records, protocols = {}, []
        for shard in range(shards):
            root = output / 'shards' / str(shard)
            protocol = json.loads((root / 'protocol.json').read_text())
            if protocol.pop('shard_index') != shard or protocol.pop('num_shards') != shards:
                raise ValueError('Unexpected shard assignment')
            protocols.append(protocol)
            rows = read_jsonl(root / dataset / method / 'timings.jsonl')
            if [r['instance_id'] for r in rows] != ids[shard::shards]:
                raise ValueError(f'Incomplete or mismatched timing shard: {root}')
            records.update({r['instance_id']: r for r in rows})
        if any(p != protocols[0] for p in protocols):
            raise ValueError('Timing shard protocols differ')
        out = output / dataset / method
        out.mkdir(parents=True, exist_ok=True)
        ordered = [records[i] for i in ids]
        (out / 'timings.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in ordered))
        write_json(output / 'protocol.json', dict(protocols[0], num_shards=shards))
        summary = dict(dataset=dataset, method=method, model=Path(cfg['model']).name,
            n=len(ordered), ttft_p50_ms=statistics.median(r['ttft_ms'] for r in ordered),
            gpu_uuids=sorted({r['gpu_uuid'] for r in ordered}), speedup=None,
            quality_evaluated=False, protocol='layerwise_online_v1')
        write_json(out / 'summary.json', summary)
        write_json(out / 'COMPLETE', dict(n=len(ordered)))
        emit('merged', **summary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--merge-adapter', action='store_true',
                        help='Merge query weights for online PIC; keep base artifact KV construction')
    parser.add_argument('--training-dir', type=Path)
    parser.add_argument('--config', type=Path, default=Path('configs/qwen3_4b.yaml'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--datasets', default='champ,musique,bigcodebench,personamem_v2')
    parser.add_argument('--methods', default='full,pic,position_oracle')
    parser.add_argument('--single-datasets', default='')
    parser.add_argument('--single-methods', default='full,pic,position_oracle')
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--ape-cache-layout', choices=('artifact_with_gaps', 'artifact_only'),
                        default='artifact_only')
    parser.add_argument('--cacheblend-mask', choices=('upstream_bottom_right', 'original_positions'),
                        default='original_positions')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--merge-shards', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    from attuner.config import methods as normalize_methods
    args.methods = ','.join(normalize_methods(args.methods)) if args.methods else ''
    args.single_methods = ','.join(normalize_methods(args.single_methods)) if args.single_methods else ''
    if args.shards < 1 or not 0 <= args.shard < args.shards:
        parser.error('Invalid shard assignment')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    jobs = [(d, m) for m in args.methods.split(',') for d in args.datasets.split(',') if d and m]
    jobs += [(d, m) for m in args.single_methods.split(',') for d in args.single_datasets.split(',') if d and m]
    if args.merge_adapter and (args.adapter is None or not any(m == 'pic_attuner' for _, m in jobs)):
        parser.error('--merge-adapter requires --adapter and pic_attuner among the methods')
    if args.dry_run:
        emit('plan', gpu=args.gpu, model=args.model, jobs=jobs, first_token_only=True,
             full_added=False, cacheblend_mask=args.cacheblend_mask, output=str(args.output),
             merge_adapter=args.merge_adapter, ape_cache_layout=args.ape_cache_layout,
             shard=args.shard, shards=args.shards, merge_shards=args.merge_shards)
        return
    if args.merge_shards:
        import yaml
        from attuner.config import load
        cfg = load(args.config)
        cfg['model'] = args.model
        merge_shards(args.output, jobs, cfg, args.shards, args.limit)
        summarize_speedups(args.output, jobs)
        return
    import torch
    import yaml
    from attuner.io import read_jsonl, append_jsonl, write_json
    from attuner.runner import data_path
    from baselines.efficiency_runtime import EfficiencyRuntime
    from baselines.trained_runtime import TrainedBaselineRuntime

    from attuner.config import load
    cfg = load(args.config)
    cfg.update(model=args.model, adapter=None, dtype='bfloat16', device='cuda:0',
        cache_dir=None, quality_only=False, attention_backend='sdpa', ape_kernel='flash',
        agentkvshift_check_layer=1, agentkvshift_ratio=.1,
        cacheblend_mask=args.cacheblend_mask, cacheblend_check_layer=1, cacheblend_ratio=.16)
    cfg['ape_cache_layout'] = args.ape_cache_layout
    if args.shards > 1:
        args.output = args.output / 'shards' / str(args.shard)
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = dict(version='layerwise_online_v1', model=args.model,
        adapter=str(args.adapter) if args.adapter else None, methods=args.methods,
        datasets=args.datasets, single_datasets=args.single_datasets, single_methods=args.single_methods,
        storage='pageable_cpu_ram', batch_size=1, generated_tokens=1, repetitions=1,
        limit=args.limit, warmup=args.warmup, fixed_prefix='dataset-common token prefix',
        gaps='online causal computation, batched by layer; never silently precached',
        artifact_encoding='unchanged from quality runtime',
        timing_includes=['RAM lookup', 'H2D', 'assembly', 'method-specific online repair',
                         'uncached online prefill', 'first token selection'],
        timing_excludes=['cache construction', 'tokenization', 'model loading'],
        speedup=None, note=('Full is remeasured in this run; derive speedups from matched rows'
                           if 'full' in args.methods.split(',') else
                           'Full was not remeasured; do not mix old speedups'))
    path = args.output/'protocol.json'
    if args.shards > 1:
        protocol.update(shard_index=args.shard, num_shards=args.shards)
    if args.ape_cache_layout != 'artifact_with_gaps':
        protocol['ape_cache_layout'] = args.ape_cache_layout
    if args.merge_adapter:
        protocol['adapter_execution'] = 'merged_q_projection'
        protocol['merge_accumulation_dtype'] = 'float32'
        protocol['artifact_encoder'] = 'original_base_weights; adapters disabled'
    if args.cacheblend_mask != 'upstream_bottom_right':
        protocol['cacheblend_mask'] = args.cacheblend_mask
        protocol['cacheblend_check_layer'] = 1
        protocol['cacheblend_ratio'] = .16
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError('Output belongs to another timing protocol; choose another output directory')
    write_json(path, protocol)
    runtime, loaded_key = None, None
    for dataset, method in jobs:
        key = method if method in {'kvlink','encoder_pic','pic_attuner'} else 'base'
        out = args.output/dataset/method
        out.mkdir(parents=True, exist_ok=True)
        examples = read_jsonl(data_path(cfg, dataset))
        if args.limit:
            examples = examples[:args.limit]
        records = {r['instance_id']: r for r in read_jsonl(out/'timings.jsonl', recover_tail=True)}
        assigned_count = len(examples[args.shard::args.shards])
        if len(records) == assigned_count and (out/'summary.json').exists():
            emit('already_complete', dataset=dataset, method=method, n=len(records))
            continue
        if key != loaded_key:
            runtime = None
            gc.collect()
            torch.cuda.empty_cache()
            emit('loading', method=method, gpu_state=gpu_snapshot(args.gpu))
            if key in {'kvlink','encoder_pic'}:
                if args.training_dir is None:
                    raise ValueError('Trained baseline requires --training-dir')
                runtime = TrainedBaselineRuntime(dict(cfg, method=method,
                    checkpoint=str(args.training_dir/method/'model')))
            else:
                if key == 'pic_attuner' and args.adapter is None:
                    raise ValueError('PIC+adapter requires --adapter')
                runtime_class = EfficiencyRuntime
                if args.merge_adapter and key == 'pic_attuner':
                    from baselines.merged_attuner import MergedAttunerRuntime
                    runtime_class = MergedAttunerRuntime
                runtime = runtime_class(dict(cfg, adapter=str(args.adapter) if key=='pic_attuner' else None))
            loaded_key = key
        emit('planning', dataset=dataset, method=method, total=len(examples))
        plans = [runtime.plan(row) for row in examples]
        if isinstance(runtime, EfficiencyRuntime):
            runtime.fixed_prefix_tokens = fixed_prefix(plans)
            emit('cache_layout', dataset=dataset, method=method,
                 fixed_prefix_tokens=len(runtime.fixed_prefix_tokens),
                 inter_artifact_gaps='kept online with original causal context')
        examples = examples[args.shard::args.shards]
        plans = plans[args.shard::args.shards]
        def measure(row, plan):
            runtime.clear_cache()
            runtime.require_warm_cache = False
            runtime.prepare_request(plan, method)
            runtime._sync()
            runtime.require_warm_cache = True
            try:
                return runtime.generate(row, method, prepared_plan=plan, max_new_tokens=1, temperature=0)
            finally:
                runtime.require_warm_cache = False
        for index in range(min(args.warmup, len(examples))):
            measure(examples[index], plans[index])
        started = time.monotonic()
        for index, (row, plan) in enumerate(zip(examples, plans)):
            iid = row['instance_id']
            if iid in records:
                continue
            result = measure(row, plan)
            record = dict(instance_id=iid, ttft_ms=result['ttft_seconds']*1000,
                output_tokens=result['output_tokens'], quality_evaluated=False,
                gpu_name=torch.cuda.get_device_name(),
                gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
                prompt_tokens=len(plan.token_ids), artifact_tokens=sum(s.token_count for s in plan.segments))
            for name in ('fixed_prefix_tokens','online_tokens','recompute_tokens','prefill_schedule'):
                if name in result:
                    record[name] = result[name]
            append_jsonl(out/'timings.jsonl', record)
            records[iid] = record
            if (index+1)%10 == 0 or index+1 == len(examples):
                emit('progress', dataset=dataset, method=method, completed=len(records),
                     total=len(examples), elapsed_s=round(time.monotonic()-started,1),
                     last_ttft_ms=round(record['ttft_ms'],2))
            if (index+1)%100 == 0:
                append_jsonl(args.output/'gpu_usage.jsonl', dict(time=time.strftime('%F %T'),
                    dataset=dataset, method=method, state=gpu_snapshot(args.gpu)))
        values = [records[r['instance_id']]['ttft_ms'] for r in examples]
        summary = dict(dataset=dataset, model=Path(args.model).name, method=method,
            n=len(values), ttft_p50_ms=statistics.median(values), speedup=None,
            gpu_uuids=sorted({r['gpu_uuid'] for r in records.values()}),
            quality_evaluated=False, protocol='layerwise_online_v1')
        write_json(out/'summary.json', summary)
        write_json(out/'COMPLETE', dict(n=len(values)))
        emit('complete', **summary)
        del plans
    summarize_speedups(args.output, jobs)


if __name__ == '__main__':
    main()
