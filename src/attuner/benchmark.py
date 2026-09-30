"""Quality evaluation and Full-teacher collection. Never reports efficiency."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

from .config import DATASETS, SOURCES, load, methods
from .io import append_jsonl, exact_coverage, read_jsonl, write_json


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/qwen3_4b.yaml')
    p.add_argument('--model', required=True, help='Local downloaded Qwen3 model directory')
    p.add_argument('--datasets', default=','.join(DATASETS))
    p.add_argument('--methods', default='full,pic,position_oracle')
    p.add_argument('--adapter', type=Path)
    p.add_argument('--training-dir', type=Path, help='Contains kvlink/model and encoder_pic/model')
    p.add_argument('--output', type=Path, default=Path('results/quality'))
    p.add_argument('--gpus', default='0')
    p.add_argument('--limit', type=int, help='Smoke-test subset, not a paper result')
    p.add_argument('--max-new-tokens', type=int, help='Override generation budget for a smoke test')
    p.add_argument('--collect', action='store_true', help='Generate Full training responses without scoring')
    p.add_argument('--generate-only', action='store_true')
    p.add_argument('--score-only', action='store_true', help='Score saved predictions without loading a model')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--shard', type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument('--shards', type=int, default=1, help=argparse.SUPPRESS)
    return p


def examples_for(args, dataset):
    split = 'train' if args.collect else 'test'
    rows = read_jsonl(Path('data/prepared') / dataset / f'{split}.jsonl')
    if not rows:
        raise FileNotFoundError(f'Prepare {dataset}/{split} first: python -m attuner.prepare --download')
    return rows[:args.limit] if args.limit else rows


def job_dir(args, dataset, method):
    return args.output / dataset if args.collect else args.output / method / dataset


def generation(cfg, dataset, args):
    result = dict(cfg['generation'])
    result.update(cfg.get('dataset_generation', {}).get(dataset, {}))
    if args.collect:
        result['max_new_tokens'] = cfg['dataset_training'][dataset]['response_tokens']
    if args.max_new_tokens is not None:
        result['max_new_tokens'] = args.max_new_tokens
    result['temperature'] = 0.0
    return result


def identity(args, cfg, dataset, method, examples):
    return dict(model=str(Path(args.model).resolve()), method=method, dataset=dataset,
        adapter=str(args.adapter.resolve()) if args.adapter else None,
        training_dir=str(args.training_dir.resolve()) if args.training_dir else None,
        generation=generation(cfg, dataset, args), collect=args.collect,
        instance_ids=[r['instance_id'] for r in examples],
        settings={k: cfg[k] for k in cfg if k.startswith(('ape_', 'epic_', 'cacheblend_', 'agentkvshift_'))},
        dtype=cfg.get('dtype'), attention_backend=cfg.get('attention_backend'),
        enable_thinking=cfg.get('enable_thinking', False))


def bind(path, value):
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(f'Settings differ from saved run: {path}. Choose a new output directory.')
    if not path.exists():
        write_json(path, value)


def worker(args):
    import torch
    from .runtime import Runtime
    from baselines.qwen3_runtime import Qwen3BaselineRuntime
    from baselines.trained_runtime import TrainedBaselineRuntime
    cfg = load(args.config)
    dataset, method = args.datasets, methods(args.methods)[0]
    rows = examples_for(args, dataset)[args.shard::args.shards]
    directory = job_dir(args, dataset, method)
    part = directory / 'parts' / f'{args.shard}.jsonl'
    saved = read_jsonl(part, recover_tail=True)
    done = {r['instance_id'] for r in saved}
    if not done <= {r['instance_id'] for r in rows} or len(done) != len(saved):
        raise ValueError('Saved shard does not match this cohort')
    pending = [r for r in rows if r['instance_id'] not in done]
    if not pending:
        return
    print(json.dumps(dict(event='loading_model', dataset=dataset, method=method,
                          shard=args.shard, remaining=len(pending))), flush=True)
    cfg.update(model=args.model, device='cuda:0', cache_dir=None,
               adapter=str(args.adapter) if args.adapter and method.endswith('attuner') else None)
    if method in {'kvlink', 'encoder_pic'}:
        cfg.update(method=method, checkpoint=str(args.training_dir / method / 'model'))
        runtime = TrainedBaselineRuntime(cfg)
    elif method in {'ape', 'epic', 'cacheblend', 'agentkvshift'}:
        runtime = Qwen3BaselineRuntime(cfg)
    else:
        runtime = Runtime(cfg)
    torch.manual_seed(cfg['seed'])
    for row in pending:
        runtime.clear_cache()
        result = runtime.generate(row, method, **generation(cfg, dataset, args))
        # Quality generation may run concurrently: these timings are not TTFT evidence.
        result = {k: v for k, v in result.items() if not k.endswith('_seconds')}
        result.update(instance_id=row['instance_id'], dataset=dataset, method=method,
                      model=getattr(runtime, 'model_name', Path(args.model).name),
                      model_identifier=getattr(runtime, 'model_identifier', str(Path(args.model).resolve())))
        append_jsonl(part, result)
        done.add(row['instance_id'])
        print(json.dumps(dict(event='progress', method=method, dataset=dataset,
                              shard=args.shard, completed=len(done), total=len(rows))), flush=True)


def score(args, dataset, method, examples):
    from .evaluate import evaluate, check_bigcode_environment
    cfg = load(args.config)
    directory = job_dir(args, dataset, method)
    env = check_bigcode_environment(cfg) if dataset == 'bigcodebench' else None
    write_json(directory / 'run_manifest.json', {'identity': {'request': {'evaluation_environment': env}}})
    result = evaluate(cfg, dataset, directory, examples)
    write_json(directory / 'summary.json', result['metrics'])
    print(json.dumps(dict(event='scored', dataset=dataset, method=method,
                          metrics=result['metrics'])), flush=True)


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    selected_methods = methods(args.methods)
    if args.collect:
        args.datasets, selected_methods = ','.join(SOURCES), ['full']
        if args.score_only:
            p.error('--collect cannot be combined with --score-only')
    datasets = args.datasets.split(',')
    if not set(datasets) <= set(DATASETS):
        p.error('Unknown dataset')
    if args.limit is not None and args.limit < 1:
        p.error('--limit must be positive')
    if any(m.endswith('attuner') for m in selected_methods) and not args.adapter:
        p.error('Attuner quality evaluation requires --adapter')
    if set(selected_methods) & {'kvlink', 'encoder_pic'} and not args.training_dir:
        p.error('Trained baseline evaluation requires --training-dir')
    if args.worker:
        # Parent already selects a single dataset for each collection worker.
        args.datasets = p.parse_args(argv).datasets
        return worker(args)
    gpus = args.gpus.split(',')
    if len(gpus) != len(set(gpus)) or any(not g.strip() for g in gpus):
        p.error('GPU IDs must be unique and nonempty')
    cfg = load(args.config)
    if args.dry_run:
        print(json.dumps(dict(datasets=datasets, methods=selected_methods, gpus=gpus,
                              collect=args.collect, output=str(args.output)), indent=2))
        return
    jobs = [(d, m) for m in selected_methods for d in datasets]
    scoring = []
    # CPU scoring of a completed dataset can overlap the next dataset's generation.
    with ThreadPoolExecutor(max_workers=1) as executor:
        for dataset, method in jobs:
            examples = examples_for(args, dataset)
            directory = job_dir(args, dataset, method)
            bind(directory / 'settings.json', identity(args, cfg, dataset, method, examples))
            prediction = directory / 'predictions.jsonl'
            if not args.score_only and not prediction.exists():
                bind(directory / 'sharding.json', {'shards': len(gpus)})
                processes, handles = [], []
                try:
                    for shard, gpu in enumerate(gpus):
                        command = [sys.executable, '-m', 'attuner.benchmark', '--worker',
                            '--config', args.config, '--model', args.model,
                            '--datasets', dataset, '--methods', method, '--output', str(args.output),
                            '--shard', str(shard), '--shards', str(len(gpus))]
                        for name in ['adapter', 'training_dir', 'limit', 'max_new_tokens']:
                            value = getattr(args, name)
                            if value is not None:
                                command += ['--' + name.replace('_', '-'), str(value)]
                        if args.collect:
                            command.append('--collect')
                        log = directory / f'worker-{shard}.log'
                        handle = log.open('a'); handles.append(handle)
                        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED='1')
                        processes.append(subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT))
                        print(f'{dataset}/{method}: GPU {gpu}, log={log}', flush=True)
                    statuses = [process.wait() for process in processes]
                    if any(statuses):
                        raise RuntimeError(f'Generation failed for {dataset}/{method}; see worker logs. Rerun to resume.')
                finally:
                    for process in processes:
                        if process.poll() is None:
                            process.terminate()
                            process.wait()
                    for handle in handles:
                        handle.close()
                records = [r for shard in range(len(gpus))
                           for r in read_jsonl(directory / 'parts' / f'{shard}.jsonl')]
                indexed = exact_coverage(records, [r['instance_id'] for r in examples])
                temporary = prediction.with_suffix('.tmp')
                temporary.write_text(''.join(json.dumps(indexed[r['instance_id']], ensure_ascii=False)+'\n' for r in examples))
                temporary.replace(prediction)
            exact_coverage(read_jsonl(prediction), [r['instance_id'] for r in examples])
            if args.collect:
                target = Path('data/teachers') / Path(args.model).name / f'{dataset}.jsonl'
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() and target.read_bytes() != prediction.read_bytes():
                    raise ValueError(f'Teacher file already exists with different responses: {target}')
                if not target.exists():
                    target.write_bytes(prediction.read_bytes())
                print(f'Teacher responses: {target}', flush=True)
            elif args.score_only or not args.generate_only:
                scoring.append(executor.submit(score, args, dataset, method, examples))
        for future in scoring:
            future.result()


if __name__ == '__main__':
    main()
