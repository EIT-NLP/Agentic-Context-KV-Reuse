"""Download public inputs and construct the fixed evaluation and mixed12k sets."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import urllib.request
import zipfile

from .io import indexed, read_jsonl, write_json


def download(url, target):
    target = Path(target)
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + '.partial')
    print(f'Download: {url}', flush=True)
    with urllib.request.urlopen(url, timeout=120) as src, temporary.open('wb') as dst:
        shutil.copyfileobj(src, dst)
    temporary.replace(target)
    return target


def download_hf(repo_id, filename, target):
    """Use the Hub client for redirects, authentication and mirror support."""
    target = Path(target)
    if target.exists():
        return target
    from huggingface_hub import hf_hub_download
    cached = hf_hub_download(repo_id, filename, repo_type='dataset',
        endpoint=os.environ.get('HF_ENDPOINT', 'https://huggingface.co').rstrip('/'))
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + '.partial')
    shutil.copyfile(cached, temporary)
    temporary.replace(target)
    return target


def fetch(root):
    for name in ['corpus/corpus.json'] + [f'instances/{d}.json' for d in
            ['theoremqa', 'logicbench', 'medcalcbench', 'champ', 'bigcodebench']]:
        download_hf('WeihangSu/SRA-Bench', name, root/'sra'/name)
    for name in ['benchmark.csv', 'train.csv']:
        download_hf('bowen-upenn/PersonaMem-v2', 'benchmark/text/'+name,
                    root/'personamem_v2'/name)
    download('https://raw.githubusercontent.com/bowen-upenn/PersonaMem-v2/main/inference.py',
             Path('data/external/PersonaMem-v2/inference.py'))
    missing = [split for split in ['train', 'dev'] if not (root/'musique'/f'{split}.jsonl').exists()]
    if missing:
        import gdown
        archive = root/'musique_v1.0.zip'
        if not archive.exists():
            temporary = str(archive)+'.partial'
            if not gdown.download(id='1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h', output=temporary, quiet=False):
                raise RuntimeError('MuSiQue download failed; see README for manual download')
            Path(temporary).replace(archive)
        with zipfile.ZipFile(archive) as z:
            for split in missing:
                candidates = [n for n in z.namelist() if n.endswith(f'musique_ans_v1.0_{split}.jsonl') and '__MACOSX' not in n]
                if len(candidates) != 1:
                    raise ValueError('Unexpected MuSiQue archive layout')
                target = root/'musique'/f'{split}.jsonl'
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(candidates[0]) as src, target.open('wb') as dst:
                    shutil.copyfileobj(src, dst)


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
    tmp.replace(path)


def sra_row(raw, corpus, synthetic=False):
    from sragents.prompts import build_prompt
    ids = raw.get('skill_annotations', raw.get('skill_ids'))
    skills = [corpus[i] for i in ids]
    system, user = build_prompt(raw, skills=[s['content'] for s in skills])
    return dict(instance_id=raw['instance_id'], dataset=raw['dataset'], raw=raw,
        messages=([{'role':'system', 'content':system}] if system else [])+[{'role':'user','content':user}],
        artifacts=[{'document_id':s['skill_id'], 'content':s['content']} for s in skills],
        synthetic=synthetic)


def musique_row(raw):
    artifacts=[]
    for i,p in enumerate(raw['paragraphs']):
        if p.get('is_supporting') is not True:
            continue
        index=int(p.get('idx',i))
        block=f"[Document {index}]\nTitle: {p.get('title','')}\n{p.get('paragraph_text','').strip()}\n"
        artifacts.append({'document_id':f"{raw['id']}:doc:{index}", 'content':block})
    user='Documents:\n\n'+'\n'.join(a['content'] for a in artifacts)
    user+=f"\nQuestion: {raw['question']}\n\nGive a concise answer. Put the final answer on its own last line exactly as `ANSWER: <answer>`."
    return dict(instance_id=raw['id'],dataset='musique',raw=raw,artifacts=artifacts,
        messages=[{'role':'system','content':'You answer multi-hop questions using only the provided documents.'},
                  {'role':'user','content':user}])


def build(raw_root, output):
    splits=Path('data/splits')
    fixed=json.loads((splits/'mixed12k.json').read_text())
    bcb=json.loads((splits/'bigcodebench.json').read_text())
    corpus={r['skill_id']:r for r in json.loads((raw_root/'sra/corpus/corpus.json').read_text())}
    training,evaluation={},{}
    for dataset in ['theoremqa','logicbench','medcalcbench','champ','bigcodebench']:
        rows=json.loads((raw_root/'sra/instances'/f'{dataset}.json').read_text())
        if dataset=='bigcodebench':
            allowed=set(bcb['eval_skills'])
            rows=[r for r in rows if set(r['skill_annotations'])<=allowed]
            if len(rows)!=667:
                raise ValueError('BigCodeBench evaluation partition no longer contains 667 examples')
        evaluation[dataset]=[sra_row(r,corpus) for r in rows]
    synthetic=read_jsonl('data/synthetic/bigcodebench.jsonl')
    allowed=set(bcb['train_skills'])
    if allowed & set(bcb['eval_skills']):
        raise ValueError('BigCodeBench skill partitions overlap')
    for row in synthetic:
        if not set(row['skill_ids'])<=allowed or len(row['skill_ids'])<2:
            raise ValueError('Synthetic task references a held-out skill or fewer than two skills')
    training['bigcodebench']=indexed([sra_row(r,corpus,True) for r in synthetic])
    training['musique']=indexed([musique_row(r) for r in read_jsonl(raw_root/'musique/train.jsonl')])
    evaluation['musique']=[musique_row(r) for r in read_jsonl(raw_root/'musique/dev.jsonl')]
    from .personamem import prepare_gt
    spec=dict(official_inference='data/external/PersonaMem-v2/inference.py',
              layout_manifest=str(splits/'personamem_layout.json'), option_seed=20260908)
    for split,filename in [('test','benchmark.csv'),('train','train.csv')]:
        path=raw_root/'personamem_v2'/filename
        with path.open(newline='') as f:
            count=sum(1 for _ in csv.DictReader(f))
        rows=prepare_gt(dict(spec,split=split,benchmark_csv=str(path),expected_count=count))['personamem_v2_test']
        if split=='test':evaluation['personamem_v2']=rows
        else:training['personamem_v2']=indexed(rows)
    test_personas={r['raw']['persona_id'] for r in evaluation['personamem_v2']}
    mixed=[]
    for dataset,pool in training.items():
        ids=fixed['selection_ids'][dataset]
        selected=[dict(pool[i],source_dataset=dataset) for i in ids]
        if len(selected)!=4000 or len(set(ids))!=4000:
            raise ValueError(f'{dataset}: expected 4000 unique training examples')
        if set(ids)&{r['instance_id'] for r in evaluation[dataset]}:
            raise ValueError('Training/evaluation sample overlap')
        if dataset=='personamem_v2' and any(r['raw']['persona_id'] in test_personas for r in selected):
            raise ValueError('PersonaMem training/evaluation persona overlap')
        write_rows(output/dataset/'train.jsonl',selected)
        mixed.extend(selected)
    # Match the stored mixture order; training itself shuffles each epoch.
    import random
    order={'musique':0,'bigcodebench':1,'personamem_v2':2}
    mixed.sort(key=lambda r:order[r['dataset']])
    random.Random(fixed['seed']).shuffle(mixed)
    write_rows(output/'mixed12k/train.jsonl',mixed)
    expected={'theoremqa':747,'logicbench':760,'medcalcbench':1100,'champ':223,
              'bigcodebench':667,'musique':2417,'personamem_v2':5000}
    for dataset,rows in evaluation.items():
        if len(rows)!=expected[dataset]:
            raise ValueError(f'{dataset}: expected {expected[dataset]} examples, received {len(rows)}')
        indexed(rows)
        write_rows(output/dataset/'test.jsonl',rows)
    write_json(output/'summary.json',dict(train={d:4000 for d in training},evaluation=expected,
                                         seed=fixed['seed'],teacher_responses_included=False))
    print(json.dumps({'prepared':str(output),'training':12000,'evaluation':expected},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--download',action='store_true')
    p.add_argument('--raw-dir',type=Path,default=Path('data/raw'))
    p.add_argument('--output',type=Path,default=Path('data/prepared'))
    args=p.parse_args()
    if args.download:fetch(args.raw_dir)
    build(args.raw_dir,args.output)


if __name__=='__main__':main()
