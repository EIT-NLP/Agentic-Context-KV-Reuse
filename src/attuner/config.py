"""Release configuration, independent of a particular machine or GPU count."""
from pathlib import Path
import yaml

DATASETS = ('theoremqa', 'logicbench', 'medcalcbench', 'champ',
            'bigcodebench', 'musique', 'personamem_v2')
SOURCES = ('musique', 'bigcodebench', 'personamem_v2')
METHODS = ('full', 'pic', 'position_oracle', 'pic_attuner', 'full_attuner',
           'ape', 'epic', 'cacheblend', 'agentkvshift', 'kvlink', 'encoder_pic')


def load(path):
    cfg = yaml.safe_load(Path(path).read_text())
    cfg.setdefault('data', {}).setdefault('prepared_paths', {})
    for dataset in DATASETS:
        cfg['data']['prepared_paths'][dataset] = {
            'test': f'data/prepared/{dataset}/test.jsonl',
            'dev': f'data/prepared/{dataset}/test.jsonl',
            'train': f'data/prepared/{dataset}/train.jsonl',
        }
    cfg['data'].update(sra_root='third_party/SR-Agents',
        musique={'evaluator_root': 'third_party/musique'},
        personamem_v2={'official_inference': 'data/external/PersonaMem-v2/inference.py',
                       'context_mode': 'gt_turns'})
    cfg['mixed_training'] = {'output_dir': 'data/prepared/mixed12k'}
    cfg['teacher_reuse'] = {'directory': 'data/teachers'}
    return cfg


def methods(value):
    aliases = {'comb': 'encoder_pic', 'pic-pos': 'position_oracle',
               'attuner': 'pic_attuner'}
    result = [aliases.get(x.strip().lower(), x.strip().lower()) for x in value.split(',')]
    if not result or any(x not in METHODS for x in result):
        raise ValueError(f'Choose methods from {METHODS}; comb and pic-pos are aliases')
    return result
