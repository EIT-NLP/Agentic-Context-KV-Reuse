"""``sragents evaluate`` — Stage 3."""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from sragents.cli._common import require_exists
from sragents.evaluate import evaluate as evaluate_one
from sragents.evaluate.metrics import compute_accuracy

DEFAULT_WORKERS = 32


def add_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "evaluate", help="Evaluate inference results (stage 3)",
        description="Extract answers from raw_output and score against ground truth.",
    )
    p.add_argument("--input", type=Path, required=True,
                   help="Inference JSONL file produced by `sragents infer`")
    p.add_argument("--instances", type=Path, required=True,
                   help="Bench instances JSON (provides ground truth)")
    p.add_argument("--output", type=Path, required=True,
                   help="Output JSON path for the eval summary")
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing output file")
    p.set_defaults(func=run)


def _one(result: dict, instances: dict) -> tuple[dict | None, str | None]:
    inst = instances.get(result["instance_id"])
    if inst is None:
        return None, f"instance {result['instance_id']} not found"

    raw = result.get("raw_output", "")
    out = evaluate_one(raw, inst)
    return ({
        "instance_id": result["instance_id"],
        "ground_truth": inst.get("eval_data", {}).get("answer", ""),
        "skill_annotations": inst.get("skill_annotations", []),
        **out,  # provides extracted_answer, correct, + any dataset-specific fields
    }, None)


def run(args) -> None:
    require_exists(args.input, "input")
    require_exists(args.instances, "instances")

    if args.output.exists() and not args.force:
        print(f"  already exists: {args.output} (use --force to overwrite)")
        return

    results = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))

    if not results:
        print(f"  no records in {args.input}")
        return

    instances = {i["instance_id"]: i
                 for i in json.loads(args.instances.read_text())}

    dataset = results[0]["dataset"]
    method = results[0].get("method", "unknown")
    model = results[0].get("model", "unknown")

    # All records must belong to the same (dataset, method, model) run.
    other_datasets = {r["dataset"] for r in results if r["dataset"] != dataset}
    if other_datasets:
        sys.exit(
            f"sragents: error: {args.input} mixes datasets "
            f"{{ {dataset}, {', '.join(sorted(other_datasets))} }} in one "
            "file. Evaluate each dataset separately."
        )
    print(f"  {dataset} × {model} × {method} — {len(results)} records")

    details: list[dict | None] = [None] * len(results)
    warnings: list[str] = []

    environment = None
    if dataset == 'bigcodebench':
        environment = dict(backend='sr-agents-subprocess',
            timeout_seconds=float(os.environ.get('BIGCODEBENCH_TIMEOUT_PER_TASK', 240)))
    if dataset == "bigcodebench" and args.workers > 1:
        # Each score launches a separate process with resource/time limits.
        # Executor failures propagate instead of being counted as wrong answers.
        pool = ThreadPoolExecutor(max_workers=args.workers)
        try:
            futures = {pool.submit(_one, r, instances): idx for idx, r in enumerate(results)}
            with tqdm(total=len(results), desc="  evaluating") as bar:
                for fut in as_completed(futures):
                    idx = futures[fut]
                    detail, warn = fut.result()
                    if warn or detail is None:
                        raise RuntimeError(warn or 'Missing evaluator result')
                    details[idx] = detail
                    bar.update(1)
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
    else:
        for idx, r in enumerate(tqdm(results, desc="  evaluating")):
            detail, warn = _one(r, instances)
            if warn:
                warnings.append(warn)
                continue
            details[idx] = detail

    for w in warnings:
        print(f"  WARN: {w}")

    if warnings or any(d is None for d in details):
        raise RuntimeError('Evaluation incomplete; no accuracy is published: ' + '; '.join(warnings))
    details_clean = [d for d in details if d is not None]
    metrics = compute_accuracy(details_clean)

    output = {
        "dataset": dataset, "method": method, "model": model,
        "metrics": metrics, "details": details_clean,
        "evaluation_environment": environment,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2))

    print(f"\n  {metrics['correct']}/{metrics['total']} correct "
          f"({metrics['accuracy']:.4f})")
    print(f"  Saved: {args.output}")

