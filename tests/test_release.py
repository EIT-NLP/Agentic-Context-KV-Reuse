"""Data boundaries, resumability and public entry-point contracts."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from attuner import benchmark
from attuner.config import load, methods
from attuner.io import append_jsonl, read_jsonl, write_json
from baselines.measure_efficiency import summarize_speedups

ROOT = Path(__file__).resolve().parents[1]


def test_dataset_download_respects_hf_endpoint(monkeypatch, tmp_path):
    from attuner.prepare import fetch
    calls, urls = [], []
    payload = tmp_path / 'downloaded.json'
    payload.write_text('[]')
    def hub_download(repo_id, filename, **kwargs):
        calls.append((repo_id, filename, kwargs))
        return str(payload)
    monkeypatch.setenv('HF_ENDPOINT', 'https://example.org/hf/')
    monkeypatch.setattr('huggingface_hub.hf_hub_download', hub_download)
    monkeypatch.setattr('attuner.prepare.download', lambda url, target: urls.append(url))
    (tmp_path / 'musique').mkdir()
    for split in ['train', 'dev']:
        (tmp_path / 'musique' / f'{split}.jsonl').touch()
    fetch(tmp_path)
    assert len(calls) == 8
    assert all(c[2] == {'repo_type':'dataset', 'endpoint':'https://example.org/hf'} for c in calls)
    assert len(urls) == 1 and urls[0].startswith('https://raw.githubusercontent.com/')
    fetch(tmp_path)
    assert len(calls) == 8  # Existing dataset files are reused.


def test_public_configs():
    for size in ['4b', '8b']:
        cfg = load(ROOT / f'configs/qwen3_{size}.yaml')
        assert cfg['training']['layer_start'] == 0
        assert cfg['training']['layer_end'] == 35
        assert cfg['training']['artifact_order'] == 'official'
        assert cfg['training']['global_batch_size'] == 16
        assert cfg['ape_cache_layout'] == 'artifact_only'
        assert cfg['cacheblend_mask'] == 'original_positions'
    assert methods('full,comb,pic-pos') == ['full', 'encoder_pic', 'position_oracle']


def test_synthetic_data_uses_training_skills_only():
    split = json.loads((ROOT / 'data/splits/bigcodebench.json').read_text())
    rows = read_jsonl(ROOT / 'data/synthetic/bigcodebench.jsonl')
    assert len(rows) == len({r['instance_id'] for r in rows}) == 4000
    assert not set(split['train_skills']) & set(split['eval_skills'])
    for row in rows:
        assert len(row['skill_ids']) >= 2
        assert set(row['skill_ids']) <= set(split['train_skills'])
        assert set(row) == {'instance_id', 'dataset', 'question', 'skill_ids'}


def test_collection_worker_keeps_single_dataset(monkeypatch, tmp_path):
    received = []
    monkeypatch.setattr(benchmark, 'worker', lambda args: received.append(args.datasets))
    benchmark.main(['--worker', '--collect', '--model', '/unused/Qwen3-4B',
                    '--datasets', 'bigcodebench', '--output', str(tmp_path)])
    assert received == ['bigcodebench']


def test_teacher_budget_differs_from_quality_budget():
    cfg = load(ROOT / 'configs/qwen3_4b.yaml')
    args = SimpleNamespace(collect=True, max_new_tokens=None)
    assert benchmark.generation(cfg, 'bigcodebench', args)['max_new_tokens'] == 1024
    assert benchmark.generation(cfg, 'musique', args)['max_new_tokens'] == 256
    args.collect = False
    assert benchmark.generation(cfg, 'bigcodebench', args)['max_new_tokens'] == 4096


def test_resume_rejects_changed_settings(tmp_path):
    path = tmp_path / 'settings.json'
    benchmark.bind(path, {'model': 'first'})
    benchmark.bind(path, {'model': 'first'})
    with pytest.raises(ValueError, match='Settings differ'):
        benchmark.bind(path, {'model': 'second'})


def test_speedup_uses_ratio_of_matched_medians(tmp_path):
    for method, times in [('full', [10, 50, 30]), ('pic', [5, 10, 15])]:
        out = tmp_path / 'musique' / method
        for i, t in enumerate(times):
            append_jsonl(out / 'timings.jsonl', dict(instance_id=str(i), ttft_ms=t, gpu_uuid='gpu0'))
        write_json(out / 'summary.json', {'n': 3})
    summarize_speedups(tmp_path, [('musique', 'full'), ('musique', 'pic')])
    assert json.loads((tmp_path / 'musique/pic/summary.json').read_text())['speedup'] == 3
    rows = read_jsonl(tmp_path / 'musique/pic/timings.jsonl')
    rows[0]['gpu_uuid'] = 'gpu1'
    (tmp_path / 'musique/pic/timings.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError, match='GPU assignment'):
        summarize_speedups(tmp_path, [('musique', 'pic')])


def test_score_only_never_starts_generation(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    cfg_path = ROOT / 'configs/qwen3_4b.yaml'
    row = {'instance_id': 'a'}
    append_jsonl(Path('data/prepared/musique/test.jsonl'), row)
    output = tmp_path / 'quality'
    append_jsonl(output / 'full/musique/predictions.jsonl', {'instance_id':'a','raw_output':'answer'})
    def forbidden(*args, **kwargs):
        raise AssertionError('Score-only attempted to launch a GPU worker')
    monkeypatch.setattr(benchmark.subprocess, 'Popen', forbidden)
    calls = []
    monkeypatch.setattr(benchmark, 'score', lambda *a: calls.append(a[1:3]))
    benchmark.main(['--score-only', '--config', str(cfg_path), '--model', '/unused/Qwen3-4B',
                    '--datasets', 'musique', '--methods', 'full', '--output', str(output)])
    assert calls == [('musique','full')]
