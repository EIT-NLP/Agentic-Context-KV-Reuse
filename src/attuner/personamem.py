"""PersonaMem-v2 GT-message preparation and official MCQ evaluation."""
import ast
import copy
import csv
import json
from collections import Counter
from pathlib import Path

from .io import write_json


def _official(path):
    """Compile unchanged official methods, without importing API dependencies."""
    path = Path(path)
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                 and n.name == "PersonaBenchmarkEvaluator")
    names = {"create_mcq_options", "extract_final_answer", "check_mcq_correctness"}
    methods = [copy.deepcopy(n) for n in owner.body
               if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in methods} != names:
        raise ValueError("Official PersonaMem evaluator methods missing")
    isolated = ast.Module(body=[ast.ImportFrom(module="__future__",
        names=[ast.alias(name="annotations")], level=0), ast.ClassDef(
        name="OfficialMethods", bases=[], keywords=[], body=methods, decorator_list=[])],
        type_ignores=[])
    namespace = {}
    exec(compile(ast.fix_missing_locations(isolated), str(path), "exec"), namespace)
    row_method = next(n for n in owner.body if isinstance(n, ast.FunctionDef)
                      and n.name == "evaluate_row")
    suffixes = [n.value.value for n in ast.walk(row_method)
                if isinstance(n, ast.AugAssign) and isinstance(n.op, ast.Add)
                and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)
                and "Please recall my related preferences" in n.value.value]
    if len(suffixes) != 1:
        raise ValueError("Cannot identify unchanged official query suffix")
    return namespace["OfficialMethods"](), suffixes[0]



def prepare_gt(spec):
    seed = int(spec.get('option_seed', 20260908))
    official, suffix = _official(spec['official_inference'])
    with Path(spec['benchmark_csv']).open(encoding='utf-8', newline='') as f:
        sources = list(csv.DictReader(f))
    if len(sources) != int(spec.get('expected_count', 5000)):
        raise ValueError('GT-only preparation requires the full benchmark')
    rows, count_histogram, contents = [], Counter(), Counter()
    split = spec.get('split', 'test')
    frozen = None
    if spec.get('layout_manifest'):
        frozen = json.loads(Path(spec['layout_manifest']).read_text())[split]
        indices = [r['row_index'] for r in frozen]
        if len(indices) != len(set(indices)):
            raise ValueError('Duplicate source rows in PersonaMem layout manifest')
    else:
        indices = list(range(len(sources)))
    for ordinal, index in enumerate(indices):
        source = sources[index]
        layout = frozen[ordinal] if frozen is not None else None
        if layout and layout['persona_id'] != source['persona_id']:
            raise ValueError('PersonaMem CSV row order differs from frozen layout')
        snippet = json.loads(source['related_conversation_snippet'])
        if not snippet:
            raise ValueError(f'Empty GT snippet at row {index}')
        messages, artifacts = [], []
        for turn, original in enumerate(snippet):
            if original.get('role') not in {'user', 'assistant'} or not isinstance(original.get('content'), str):
                raise ValueError(f'GT row {index} turn {turn} is not a text user/assistant message')
            # Extra JSON metadata is not model-visible. Role and content stay verbatim.
            message = {k: original[k] for k in ('role', 'content')}
            messages.append(message)
            artifacts.append({'document_id': f"persona{source['persona_id']}:gt-message:{turn}",
                'message_start': turn, 'message_end': turn + 1,
                'grouping': 'one_official_gt_message', 'source_snippet_index': turn})
            contents[(source['persona_id'], message['role'], message['content'])] += 1
        count_histogram[len(messages)] += 1
        try:
            query = json.loads(source['user_query'])
        except json.JSONDecodeError:
            query = ast.literal_eval(source['user_query'])
        if query.get('content'):
            query['content'] += suffix
        if not isinstance(query.get('content'), str):
            raise ValueError('GT-only query must be text')
        row_seed = layout['official_row_seed'] if layout else seed + index
        incorrect = json.loads(source['incorrect_answers']) if source['incorrect_answers'] else []
        instruction, mapping = official.create_mcq_options(source['correct_answer'], incorrect, seed=row_seed)
        raw = {k: source[k] for k in ('correct_answer', 'topic_query', 'pref_type', 'updated', 'sensitive_info')}
        raw.update(persona_id=source['persona_id'], row_index=index, option_mapping=mapping,
            official_row_seed=row_seed, context_mode='gt_turns',
            gt_source='related_conversation_snippet', gt_message_count=len(snippet))
        if split == 'train':
            raw['split'] = 'train'
        if layout:
            if len(layout['artifact_ids']) != len(artifacts):
                raise ValueError('PersonaMem artifact count differs from frozen layout')
            for artifact, identifier in zip(artifacts, layout['artifact_ids']):
                artifact['document_id'] = identifier
        rows.append({'instance_id': layout['instance_id'] if layout else f"personamem-v2:{source['persona_id']}:{index:05d}",
            'dataset': 'personamem_v2', 'question': query['content'], 'raw': raw,
            'messages': messages + [query, {'role':'system', 'content':instruction}],
            'artifacts': artifacts})
    if len({r['instance_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate GT-only row IDs')
    if spec.get('audit_path'):
        write_json(spec['audit_path'], {'n':len(rows), 'context_mode':'gt_turns', 'images':False,
            'gt_source':'related_conversation_snippet', 'artifact_unit':'one role/content message',
            'persona_background_included':False, 'unannotated_history_included':False,
            'annotation_order_preserved':True, 'gt_messages_per_question':dict(count_histogram),
            'total_artifact_occurrences':sum(contents.values()), 'unique_persona_artifacts':len(contents),
            'artifact_occurrence_histogram':dict(Counter(contents.values())),
            'option_seed':seed, 'protocol':'GT-only ablation; official questions/options/MCQ evaluator',
            'history_matching_required':False})
    return {'personamem_v2_test': rows}


def official_evaluate(spec, predictions, examples):
    from .io import exact_coverage
    official, _ = _official(spec["official_inference"])
    by_id = exact_coverage(predictions, [row["instance_id"] for row in examples])
    details = []
    for row in examples:
        raw = row["raw"]
        letter = official.extract_final_answer(by_id[row["instance_id"]]["raw_output"])
        correct = official.check_mcq_correctness(letter, raw["correct_answer"], raw["option_mapping"])
        details.append({"instance_id": row["instance_id"], "correct": bool(correct),
                        "predicted_answer_mcq": letter, "persona_id": raw["persona_id"]})
    count = sum(d["correct"] for d in details)
    return {"metrics": {"accuracy": count / len(details), "correct": count, "total": len(details)},
            "details": details, "errors": []}
