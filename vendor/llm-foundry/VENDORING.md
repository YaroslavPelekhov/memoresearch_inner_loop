# Vendored LLM Foundry provenance

This focused library copy was taken on 2026-08-20 from the remounted source
directory `remote/tools/llm-foundry-gigachat3.5` (package version `0.2.0`). The
source mount did not contain Git metadata, so there is no trustworthy source
commit hash to record.

Retained:

- the supplied `llmfoundry/` Python package;
- normal train, evaluation, and data-preparation scripts;
- the default Gigar configuration;
- license, package metadata, and upstream README.

Excluded from the focused copy:

- internal submission, Docker, benchmark, notebook, and test infrastructure;
- bundled evaluation data;
- generated caches and artifacts;
- large local evaluation fixtures under `scripts/eval/local_data`.

Autoresearch-specific patches are committed in this outer repository: an
external decoder registration seam, a one-GPU FSDP guard, and an optional
post-instantiation parameter-count check. Candidate mutations may not edit this
vendored subtree in the first experiment.

The source declares private `contrib/composer` and `contrib/streaming` Git
submodules, but their directories were empty in the mounted copy and anonymous
access to the GitLab remotes was denied. The remote runtime therefore uses
public `mosaicml==0.20.1` and `mosaicml-streaming==0.13.0`, plus the focused
`autoresearch.lmfoundry_compat` adapter. The adapter covers only the dense Gigar
path exercised by this experiment; unavailable private callbacks, optimizers,
and schedulers raise explicit errors. It is not represented as an exact copy of
the missing private runtime.
