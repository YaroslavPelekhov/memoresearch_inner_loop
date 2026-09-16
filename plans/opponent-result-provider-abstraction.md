# Plan — OpponentResultProvider abstraction (cached vs executing)

**Goal:** Stop re-executing opponent programs in `FetchOpponentResultsStage` when the opponent's `CallProgramFunction` output is already stored in the opponent's Redis DB. Design a clean OOP abstraction so this is a generic framework feature, not a role-hardcoded shortcut.

**Target experiment:** `heilbron/adversarial-repro-v1` (stopped). Next launch must use the new provider for D runs and keep exec for G runs.

---

## 1. Problem recap

`FetchOpponentResultsStage._exec_one` (`gigaevo/adversarial/stages.py:157-171`) runs the opponent's `entrypoint()` in a fresh subprocess for every evaluation of the current program. This is the source of:

- **F-1**: 300 s per-opponent timeout eating into DAG budget on D runs.
- **F-4**: silent None-drop on exec failure → DGTrackerStage length mismatch.
- Wasted compute: the opponent was already evaluated in its own run and the output is persisted.

### Asymmetry

| Run role | Fetches | Opponent output type                        | Re-exec necessary? |
|----------|---------|---------------------------------------------|--------------------|
| D (improver)    | G constructors | `np.ndarray(11, 2)` — pure data     | No — cache it     |
| G (constructor) | D improvers    | closure `improve(points) -> points` | Yes — must invoke on G's fresh points |

For G-side, the opponent output is a callable. Even if `cloudpickle` could round-trip it from Redis, the closure's internal RNG/state would drift from a freshly-seeded invocation. Safer: re-exec each time. However, per the user's concern, D improvers can legitimately run long, so the per-opponent timeout on G side must be raised to `stage_timeout` to avoid starving legitimate work.

### Storage already exists

`ProgramStageResult.output` is serialized with `pickle_b64_serialize` (cloudpickle under the hood) and stored in the opponent program's JSON blob at `{prefix}:program:{pid}`. See `gigaevo/programs/core_types.py:148-162` and `gigaevo/database/redis_program_storage.py:155-156`. So fetching `program.stage_results["CallProgramFunction"].output.data` from the opponent's DB gives us the same value the subprocess would produce.

---

## 2. Design — strategy pattern

### 2.1 New interface

```python
# gigaevo/adversarial/opponent_result_provider.py (new file)

class OpponentResultProvider(ABC):
    """Produce opponent evaluation payloads for a list of opponent program IDs.

    Replaces the inline execution in FetchOpponentResultsStage. Two concrete
    implementations:
      - ExecOpponentResultProvider: runs opponent code in a subprocess
        (current behavior, extracted as-is). Required when opponent output
        is a non-reproducible callable.
      - CachedOpponentResultProvider: reads stored CallProgramFunction output
        from the opponent's Redis DB. Used when opponent output is a static
        value (ndarray, dict, scalar, etc.).

    Both implementations are responsible for aligning output length with
    the requested ID list: failed/missing opponents become None placeholders
    (fixes F-4 silent-drop at stages.py:145).
    """

    @abstractmethod
    async def produce(self, ids: list[str]) -> list[Any | None]:
        """Return one result per id in the same order; None for failure/miss."""
```

### 2.2 Two concrete implementations

**ExecOpponentResultProvider** — wraps the existing `run_exec_runner` path. Owns the archive_provider (for `get_codes_by_ids`), `per_opponent_timeout`, `python_path`, `max_memory_mb`. Identical semantics to today.

**CachedOpponentResultProvider** — talks directly to opponent Redis DB(s). For each id:
  1. GET `{prefix}:program:{pid}`
  2. Decode the program JSON, extract `stage_results["CallProgramFunction"]`
  3. If status == COMPLETED → deserialize `output` via `pickle_b64_deserialize`, return its `.data` field (the `Box[Any]` payload)
  4. Else → return None

No subprocess. No timeout. No import of fallback code.

Both providers share the same "None on failure" contract, so the stage becomes role-agnostic.

### 2.3 Thinned-out stage

```python
class FetchOpponentResultsStage(Stage):
    def __init__(self, *, result_provider: OpponentResultProvider,
                 fallback_codes: list[str] | None = None, ...):
        ...

    async def compute(self, program):
        ids = self.params.opponent_ids.data
        results = await self._result_provider.produce(ids)
        # Apply fallback only when ALL ids failed/empty (cold start)
        if not ids or all(r is None for r in results):
            if self._fallback_codes:
                results = await self._exec_fallbacks()  # same pattern as today
        return Box[Any](data=results)
```

The stage no longer owns subprocess semantics — it's a coordinator. `fallback_codes` stays here because cold-start fallback is orthogonal to caching (both modes need it when the archive is empty).

### 2.4 Pipeline wiring

`AdversarialPipelineBuilder.__init__` gains an `opponent_result_provider` parameter. A small factory in `AdversarialPipelineBuilder` (or a free function in `opponent_result_provider.py`) constructs the right provider from config:

```python
def build_opponent_result_provider(
    mode: Literal["exec", "cached"],
    *,
    archive_provider: OpponentArchiveProvider,
    host: str,
    port: int,
    sources: list[dict],
    per_opponent_timeout: float,
    python_path: list[Path],
    max_memory_mb: int | None,
) -> OpponentResultProvider: ...
```

Callers pass `mode` from the pipeline YAML; factory picks the impl. `AdversarialAsymmetricPipelineBuilder` threads it through.

### 2.5 Config surface

Add to `config/pipeline/adversarial_asymmetric.yaml`:

```yaml
# When "exec": run opponent code in subprocess (default; required when
# opponent output is a callable that must be re-invoked per evaluation).
# When "cached": read the opponent's CallProgramFunction output from
# their Redis DB. Use when opponent output is a static value (ndarray,
# dict, scalar) that was already computed in the opponent's own run.
opponent_result_mode: "exec"
```

In `heilbron_repro_v1.yaml` we do **not** set it globally. Instead, per-run overrides in `experiment.yaml.runs[*].extra_overrides` flip it:

- G runs (role=constructor, opponents are D improvers) → `opponent_result_mode=exec` + `per_opponent_timeout=${stage_timeout}` (≈3000 s)
- D runs (role=improver, opponents are G constructors) → `opponent_result_mode=cached`

This keeps the mechanism generic (no hardcoded `if role == "improver"` in framework code) while letting the experiment author declare the right strategy per run.

### 2.6 Timeout fix for G-side exec mode

In the `exec` path, change the default `per_opponent_timeout` to pull from `stage_timeout` when not explicitly set, OR set it per-run at the YAML level. Chosen approach: **explicit at the YAML level** (no implicit coupling). The experiment manifest launcher already knows per-run overrides; we add `per_opponent_timeout=${stage_timeout}` to G run `extra_overrides`.

---

## 3. Touchpoints (exact files & changes)

| # | File | Change |
|---|------|--------|
| 1 | `gigaevo/adversarial/opponent_result_provider.py` **(new)** | Abstract base + `ExecOpponentResultProvider` + `CachedOpponentResultProvider` + factory |
| 2 | `gigaevo/adversarial/stages.py` | Refactor `FetchOpponentResultsStage`: accept `result_provider`, drop `_exec_one`/`per_opponent_timeout`/`python_path`/`max_memory_mb` kwargs. Move exec path to `ExecOpponentResultProvider`. Fix F-4 by passing None placeholders (done naturally since provider returns aligned list). |
| 3 | `gigaevo/adversarial/pipeline.py` | `AdversarialPipelineBuilder.__init__` takes `opponent_result_provider`. Compose it in `_add_adversarial_stages`. Delete the `total_timeout = stage_timeout + 30` comment (it moves to ExecOpponentResultProvider). |
| 4 | `gigaevo/adversarial/asymmetric_pipeline.py` | Thread `opponent_result_provider` through `AdversarialAsymmetricPipelineBuilder.__init__`. |
| 5 | `gigaevo/adversarial/opponent_provider.py` | Optional: expose the Redis conn primitives so `CachedOpponentResultProvider` can reuse them (or just pass `host, port, sources` explicitly). |
| 6 | `config/pipeline/adversarial_asymmetric.yaml` | New `opponent_result_mode: "exec"` key. Hydra `pipeline_builder._target_` arg list grows by one. Add a new top-level `opponent_result_provider:` block invoked via `_target_` factory. |
| 7 | `config/pipeline/adversarial_coevo.yaml` (parent) | Add the same key + block, defaulted to exec, so inheritance works. |
| 8 | `config/pipeline/heilbron_repro_v1.yaml` | No change — per-run overrides carry it. |
| 9 | `experiments/heilbron/adversarial-repro-v1/experiment.yaml` | Per-run extras: G runs get `per_opponent_timeout=${stage_timeout}` + `opponent_result_mode=exec`; D runs get `opponent_result_mode=cached`. Add both keys to `pinned:` so treatment-verifier catches drift. |
| 10 | `experiments/heilbron/adversarial-repro-v1/01_design.md` **(append)** | Document the opponent_result_mode split and the G-side timeout bump. |
| 11 | `experiments/heilbron/adversarial-repro-v1/treatment_checks` (yaml field) | Add log patterns: `\[CachedOpponentResultProvider\] .* hit=\d+` (for D runs), `\[ExecOpponentResultProvider\]` (for G runs). |
| 12 | `tests/adversarial_pipeline/test_stages.py` | Keep existing tests; migrate to construct a mock result provider. Add one exec-provider integration-ish test. |
| 13 | `tests/adversarial_pipeline/test_opponent_result_provider.py` **(new)** | Unit tests: exec happy-path + failure; cached happy-path + missing program + missing stage_result + deserialize failure. |
| 14 | `tests/adversarial_pipeline/test_pipeline.py` | Update pipeline-builder tests that constructed `FetchOpponentResultsStage` directly. |

---

## 4. Behavioral contracts (invariants to keep)

1. **Length preservation.** `produce(ids)` returns `len(ids)` items; failures are None. Downstream: `DGTrackerStage` aligns `per_opp_delta` by index — no more silent-drop (fixes F-4 for free).
2. **Fallback path.** Cold start (ids empty OR everything None) falls back to `fallback_codes` executed via subprocess, regardless of mode. Reuse `ExecOpponentResultProvider.produce_from_codes(codes)` as a small extra method.
3. **Cache semantics of FetchOpponentResultsStage.** `InputHashCache` on opponent-ids still works: same ids hash → same stage result (both modes are deterministic given same ids + same opponent archive state).
4. **Error visibility.** Both providers must log at INFO when >0 results are None (with ids[:3] preview and counts). Critical for post-mortem diagnosis.
5. **Security/isolation.** `CachedOpponentResultProvider` never runs user code; `ExecOpponentResultProvider` preserves all current sandboxing (memory cap, output cap, python_path).

---

## 5. Testing strategy

- **Unit** — `CachedOpponentResultProvider`: stub Redis with pre-written JSON blobs covering (a) COMPLETED stage_result with ndarray output, (b) FAILED stage_result, (c) missing key, (d) malformed pickle. Assert aligned None output.
- **Unit** — `ExecOpponentResultProvider`: mirrors current `_exec_one` tests, with the stage-level test reduced to wiring.
- **Stage wiring** — `FetchOpponentResultsStage` invokes provider, propagates Box output, applies fallback on cold start.
- **Pipeline** — ensure both `opponent_result_mode` values build the pipeline cleanly, and the stage gets the right provider impl.
- **Smoke** — 3-gen run of `heilbron_repro_v1` with D=cached and G=exec(300 s) confirms:
  - D run log contains `[CachedOpponentResultProvider] hit=K miss=0`
  - G run log contains `[ExecOpponentResultProvider] ok=K fail=0`
  - DGTrackerStage runs without length mismatch
  - Wall-clock per-gen on D drops by ≥30% (no more subprocess exec for N=1 opponent)

---

## 6. Rollout

1. Land the refactor behind `opponent_result_mode="exec"` (default) — bit-for-bit backwards compatible. All existing experiments keep working.
2. Update `heilbron/adversarial-repro-v1/experiment.yaml` per §3.9.
3. Re-run smoke (`gigaevo launch --smoke`) for one G and one D run.
4. Inspect logs for the two markers in §5 bullet "Smoke".
5. Full relaunch with fresh Redis DBs 1–8.
6. Update `04_issues_log.md` with the F-1/F-4 resolution note.

---

## 7. Open questions / risk

1. **Stale cache vs archive churn.** If the opponent's `CallProgramFunction` output was recomputed since it was cached (new program version with same id?), cached output could be stale. Low risk: program IDs are immutable in this codebase — once persisted, output doesn't change. Verify by grepping for any call path that rewrites `stage_results["CallProgramFunction"]` on an existing id.
2. **Partial stage_results.** If the opponent program was stored before `CallProgramFunction` completed (intermediate save), we'll see SKIPPED/PENDING status. The provider returns None → fallback kicks in, same behavior as exec failure.
3. **Cross-DB latency.** `CachedOpponentResultProvider` reads from a different DB on the same Redis server. MGET on the id list makes this ≤1 RTT per call — orders of magnitude cheaper than subprocess spawn.
4. **Test matrix size.** New provider doubles the pipeline-config test matrix. Mitigate with parametrized tests.

---

## 8. Non-goals

- Caching D-side output (callables). Out of scope; explicitly documented as an exec-only path.
- Cross-experiment opponent sharing. Still one experiment at a time.
- Changing `OpponentArchiveProvider` interface (id list + code fetch stay put). We only ADD `OpponentResultProvider`.
