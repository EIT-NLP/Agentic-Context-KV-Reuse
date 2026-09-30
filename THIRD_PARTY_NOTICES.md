# Third-party sources

Apache-2.0 applies to the original Attuner code, not to upstream datasets or
separately licensed source. Preserve the upstream notices when redistributing.

## Included evaluation code

- [SR-Agents / SRA-Bench](https://github.com/oneal2000/SR-Agents): reduced prompt
  and evaluator implementation; license retained at `third_party/SR-Agents/LICENSE`.
  Local changes are described in its README.
- [MuSiQue](https://github.com/StonyBrookNLP/musique): evaluation script and metrics;
  license retained at `third_party/musique/LICENSE`.
- The SR-Agents evaluators also contain code attributed to
  [BigCodeBench](https://github.com/bigcode-project/bigcodebench),
  [TheoremQA](https://github.com/TIGER-AI-Lab/TheoremQA), and
  [MedCalc-Bench](https://github.com/ncbi-nlp/MedCalc-Bench).
  Original attribution comments are retained.

## Downloaded separately

- [PersonaMem-v2](https://github.com/bowen-upenn/PersonaMem-v2): its original
  `inference.py` is fetched during preparation for its MCQ option/answer methods.
  It is not bundled or relicensed here. The data is downloaded from the authors'
  [dataset](https://huggingface.co/datasets/bowen-upenn/PersonaMem-v2).
- SRA-Bench data is downloaded from
  [WeihangSu/SRA-Bench](https://huggingface.co/datasets/WeihangSu/SRA-Bench).
- MuSiQue data uses the official download linked in the authors' repository.
- Qwen3 base weights are downloaded separately under their upstream terms.

## Baseline implementations

The Qwen3 adaptations and their upstream repositories are listed in
[baselines/README.md](baselines/README.md). AgentKVShift is a paper
reimplementation; the other five methods were adapted with reference to the
linked author code. No full upstream baseline checkout is bundled.

`data/splits/` records the experimental partitions and PersonaMem option layout.
`data/synthetic/bigcodebench.jsonl` contains the 4,000 synthetic training questions
and training-side skill IDs, without teacher responses or model weights.
Downloaded benchmark content remains subject to its own dataset terms.
