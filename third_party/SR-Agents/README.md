# SR-Agents evaluation subset

Source: https://github.com/oneal2000/SR-Agents

This subset retains the prompt builders, dataset evaluators and evaluation CLI
used by Attuner. The upstream license is in LICENSE. Retrieval, API inference,
ToolQA and experiment launchers are not included.

Local modifications cover bounded mathematical expression parsing, symbolic
equivalence and numerical list answers, and propagation of BigCodeBench
execution/environment failures. BigCodeBench runs through local subprocesses,
not Docker. Its resource guard is not an OS security sandbox; run generated code
in a disposable evaluation environment without credentials or valuable files.

The dependencies in pyproject.toml include the packages used by BigCodeBench tests.
