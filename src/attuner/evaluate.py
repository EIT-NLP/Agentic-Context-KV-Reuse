"""Invoke benchmark evaluators with documented safety/correctness patches and retain per-instance evidence."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

from .io import exact_coverage, read_jsonl, write_json, comparable_identity


def extract_answer(text):
    matches = re.findall(r"(?im)^\s*ANSWER\s*:\s*(.*?)\s*$", text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return (matches[-1] if matches else lines[-1] if lines else "").strip().strip("*").strip()


def evaluate(config, dataset, directory, examples):
    environment = check_bigcode_environment(config) if dataset == 'bigcodebench' else None
    if environment is not None:
        manifest = json.loads((Path(directory) / 'run_manifest.json').read_text())
        if comparable_identity(manifest['identity']['request'].get('evaluation_environment')) != comparable_identity(environment):
            raise ValueError('BigCodeBench evaluation environment differs from inference identity')
    directory = Path(directory).resolve()
    rows = read_jsonl(directory / "predictions.jsonl")
    ids = [str(row["instance_id"]) for row in examples]
    by_id = exact_coverage(rows, ids)
    official = directory / "official"
    official.mkdir(exist_ok=True)
    log = official / "evaluator.log"
    if dataset == 'personamem_v2':
        from .personamem import official_evaluate
        result = official_evaluate(config['data']['personamem_v2'], rows, examples)
        exact_coverage(result['details'], ids)
        if result.get('errors'):
            raise RuntimeError(f"official evaluation errors: {result['errors']}")
        write_json(official/'evaluation.json', result)
        write_json(official/'command.json', {'callable':'official PersonaBenchmarkEvaluator MCQ methods',
            'source':config['data']['personamem_v2']['official_inference'],
            'mode':'unchanged AST extraction, no API client initialization'})
        log.write_text(f"Scored {len(rows)} predictions with official MCQ evaluator.\n")
        return result
    if dataset == "musique":
        gold = official / "gold.jsonl"
        prediction = official / "predictions.jsonl"
        with gold.open("w") as gf, prediction.open("w") as pf:
            for row in examples:
                gf.write(json.dumps(row["raw"], ensure_ascii=False) + "\n")
                pf.write(json.dumps({"id": row["instance_id"],
                    "predicted_answer": extract_answer(by_id[row["instance_id"]]["raw_output"]),
                    "predicted_support_idxs": [], "predicted_answerable": True}) + "\n")
        root = Path(config["data"]["musique"]["evaluator_root"]).resolve()
        # Load the filename with a dot through importlib; evaluator code stays unchanged.
        script = '''import importlib.util,json,sys
from metrics.answer import AnswerMetric
spec=importlib.util.spec_from_file_location("official_eval", "evaluate_v1.0.py")
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
pred,gold,out=sys.argv[1:]
metrics=mod.evaluate(pred,gold)
details=[]
for p,g in zip(mod.read_jsonl(pred),mod.read_jsonl(gold)):
    metric=AnswerMetric();metric(p["predicted_answer"],[g["answer"]]+g["answer_aliases"])
    em,f1=metric.get_metric()
    details.append({"instance_id":g["id"],"correct":bool(em),"answer_em":em,"answer_f1":f1})
json.dump({"metrics":metrics,"details":details},open(out,"w"),indent=2)
'''
        command = [sys.executable, "-c", script, str(prediction), str(gold),
                   str(official / "evaluation.json")]
    else:
        root = Path(config["data"]["sra_root"]).resolve()
        gold = official / "instances.json"
        write_json(gold, [row["raw"] for row in examples])
        command = [sys.executable, "-m", "sragents.cli.main", "evaluate",
                   "--input", str(directory / "predictions.jsonl"), "--instances", str(gold),
                   "--output", str(official / "evaluation.json"), "--workers",
                   str(config.get("eval_workers", 32)), "--force"]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    write_json(official / "command.json", {"argv": command, "cwd": str(root)})
    with log.open("w") as handle:
        subprocess.run(command, cwd=root, env=env, stdout=handle,
                       stderr=subprocess.STDOUT, check=True)
    result = json.loads((official / "evaluation.json").read_text())
    exact_coverage(result["details"], ids)
    if result.get("errors"):
        raise RuntimeError(f"official evaluation errors: {result['errors']}")
    return result


def check_bigcode_environment(config):
    """Record the local evaluator and timeout; no container or installation checks."""
    return dict(backend='sr-agents-subprocess',
                evaluator_root=str(Path(config['data']['sra_root']).resolve()),
                timeout_seconds=float(os.environ.get('BIGCODEBENCH_TIMEOUT_PER_TASK', 240)))
