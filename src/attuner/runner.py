"""Dataset and generation settings shared by release entry points."""
from pathlib import Path

def data_path(config, dataset, split=None):
    split = split or ("dev" if dataset == "musique" else "test")
    configured = config.get("data", {}).get("prepared_paths", {}).get(dataset, {}).get(split)
    if configured:
        return Path(configured)
    return Path(config["data"]["output_dir"]) / f"{dataset}_{split}.jsonl"

def shard_examples(examples, dataset, shard, shards):
    """Keep each PersonaMem history on one worker so its cache is built once."""
    if dataset != "personamem_v2":
        return examples[shard::shards]
    selected = []
    for row in examples:
        persona = str(row.get("raw", {}).get("persona_id", ""))
        if not persona:
            raise ValueError("PersonaMem row is missing persona_id")
        digits = "".join(character for character in persona if character.isdigit())
        owner = (int(digits) if digits else sum(map(ord, persona))) % shards
        if owner == shard:
            selected.append(row)
    return selected

def generation_config(config, dataset, rollout=False):
    result = dict(config["generation"])
    result.update(config.get("dataset_generation", {}).get(dataset, {}))
    if rollout:
        result["temperature"] = 0.0
        result["max_new_tokens"] = int(config["training"].get("response_tokens", result["max_new_tokens"]))
    return result
