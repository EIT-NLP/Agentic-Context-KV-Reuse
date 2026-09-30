"""Small restart helpers shared by the independent baseline entry points."""
import json
import os
from pathlib import Path
import random


def bind_run(directory, manifest, name='run_manifest.json'):
    from attuner.io import write_json
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError(f'Run settings changed; use another output directory: {path}')
    else:
        if any(directory.iterdir()):
            raise ValueError(f'Existing output has no {name}; use another directory')
        write_json(path, manifest)


def pending_examples(directory, examples):
    from attuner.io import read_jsonl, indexed
    path = Path(directory) / 'predictions.jsonl'
    rows = read_jsonl(path, recover_tail=True)
    done = indexed(rows)
    expected = {str(row['instance_id']) for row in examples}
    if len(expected) != len(examples) or not set(done) <= expected:
        raise ValueError('Prediction IDs do not match this evaluation cohort')
    invalid = [row for row in rows if not isinstance(row.get('raw_output'), str)
               or row.get('status') in {'error','failed'}]
    if invalid:
        # Preserve failed records, then retry those examples instead of scoring them.
        with (Path(directory)/'incomplete_predictions.jsonl').open('a') as handle:
            for row in invalid: handle.write(json.dumps(row)+'\n')
        bad = {str(row['instance_id']) for row in invalid}
        rows = [row for row in rows if str(row['instance_id']) not in bad]
        temp = path.with_suffix('.tmp')
        temp.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows))
        temp.replace(path)
        done = indexed(rows)
    # An interrupted final write may be valid JSON without the terminating newline.
    if rows and path.stat().st_size:
        with path.open('rb+') as handle:
            handle.seek(-1, 2)
            if handle.read(1) != b'\n':
                handle.seek(0, 2)
                handle.write(b'\n')
    return [row for row in examples if str(row['instance_id']) not in done]


def final_model_exists(directory, method):
    path = Path(directory) / 'model'
    config = path / 'baseline_config.json'
    if not config.is_file():
        return False
    if method == 'encoder_pic':
        weights = [path / 'encoder_pic.pt']
    else:
        if not (path / 'config.json').is_file() or not (path / 'tokenizer_config.json').is_file():
            return False
        index = path / 'model.safetensors.index.json'
        weights = ([path / name for name in set(json.loads(index.read_text())['weight_map'].values())]
                   if index.exists() else [path / 'model.safetensors'])
    return bool(weights) and all(p.is_file() and p.stat().st_size > 0 for p in weights)


def save_training(directory, learner, optimizer, update, rank=0, world=1):
    """Atomic rolling checkpoint; retain trainable tensors, optimizer and each rank's RNG."""
    import torch
    import torch.distributed as dist
    rng = dict(python=random.getstate(), cpu=torch.get_rng_state(),
               cuda=torch.cuda.get_rng_state() if torch.cuda.is_available() else None)
    states = [None] * world if rank == 0 else None
    if world > 1:
        dist.gather_object(rng, states, dst=0)
    else:
        states = [rng]
    if rank == 0:
        directory = Path(directory)
        payload = dict(update=update,
                       parameters={n: p.detach() for n, p in learner.named_parameters() if p.requires_grad},
                       buffers=dict(learner.named_buffers()), optimizer=optimizer.state_dict(), rng=states)
        temp = directory / 'resume.pt.tmp'
        with temp.open('wb') as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        temp.replace(directory / 'resume.pt')
    if world > 1:
        dist.barrier()


def restore_training(directory, learner, optimizer, rank=0, world=1):
    import torch
    path = Path(directory) / 'resume.pt'
    if not path.exists():
        return 0
    state = torch.load(path, map_location='cpu', weights_only=False)
    if len(state['rng']) != world:
        raise ValueError('Resume requires the same GPU count; use another directory for a new run')
    parameters = {n: p for n, p in learner.named_parameters() if p.requires_grad}
    if parameters.keys() != state['parameters'].keys():
        raise ValueError('Trainable parameter layout differs from the checkpoint')
    with torch.no_grad():
        for name, value in state['parameters'].items():
            parameters[name].copy_(value)
        for name, value in state['buffers'].items():
            learner.get_buffer(name).copy_(value)
    optimizer.load_state_dict(state['optimizer'])
    rng = state['rng'][rank]
    random.setstate(rng['python'])
    torch.set_rng_state(rng['cpu'])
    if rng['cuda'] is not None:
        torch.cuda.set_rng_state(rng['cuda'])
    return state['update']


def trim_training_log(directory, update):
    from attuner.io import read_jsonl
    path = Path(directory) / 'train.jsonl'
    if path.exists():
        rows = read_jsonl(path, recover_tail=True)
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows if row['update'] <= update))
