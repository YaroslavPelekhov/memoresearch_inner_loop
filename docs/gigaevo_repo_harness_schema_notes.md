# GigaEvo Repo-Harness Schema Notes

Use `gigaevo_repo_harness_schema_simple.svg` for the main presentation slide.
Use `gigaevo_repo_harness_schema_simple.png` when the slide tool handles PNG
more reliably. The older `presentation` and DOT files are more detailed
reference versions.

## Talk Track

GigaEvo still owns the evolutionary loop: select parents, mutate, evaluate,
score, admit to the archive, then repeat. Repo-harness changes the candidate
artifact from an inline code string into a Git-backed repository snapshot.

The actual solution code is stored in Git. Each candidate is a commit on a
`gigaevo/candidate/...` branch. Redis stores a compact `RepoCandidateManifest`
in `Program.code`, with `repo_path`, `commit`, `parent_commit`, `branch`,
`entrypoint`, and `changed_files`.

Agents communicate through explicit artifacts rather than hidden state:

- `MetaRepoParentSelector` reads active parent summaries, metrics, reflections,
  archive roles, and compact Git diffs, then emits a selected parent set.
- `RepoHarnessMutationOperator` writes `mutation_brief.md`, creates a Git
  worktree at the primary parent commit, and runs the selected coding-agent CLI.
- The mutation agent edits repository files; GigaEvo runs smoke checks if
  configured and commits the changed worktree.
- `RepoReflectionStage` reads the child/parent diff, metric deltas, benchmark
  feedback, parent feedback, and artifact paths, then writes markdown guidance
  for the next mutation.

Feedback is organized in `Program.metadata`:

- `repo_benchmark_feedback`: benchmark command, return code, metrics,
  stdout/stderr tails, structured feedback, staged-validation details.
- `repo_evaluation_artifacts`: copied logs or benchmark-declared files.
- `repo_reflection`: diff summary, metric deltas, changed files, and LLM insight.
- `mutation_context`: formatted metrics and reflection text used in the next
  mutation brief.
- `repo_descriptors` and `archive_roles`: optional repo-archive admission state.

The archive strategy decides which evaluated programs become active parents.
The default repo-harness setup uses a single-island archive; the archive variant
uses repo-specific top-K roles such as `active_parent`, `hall_of_fame`,
`local_pareto`, `novelty`, `failure`, `superseded`, and `quarantined`.
