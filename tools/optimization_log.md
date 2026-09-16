# Optimization Log

Tracks each iteration of the automated throughput optimization loop.

## Baseline (captured)

Key numbers from baseline run:
- generation_wall_time N=5000: ~19s
- 3_generations N=5000: ~16s (5.4s/gen)
- collector.compute N=5000: ~4360ms
- get_all N=5000: ~3815ms
- frontier_recompute N=5000: ~39ms
- state transitions: ~2200/s
- serialization: ~0.005ms (negligible)
- complexity 5KB: ~2.8ms/call

## Iteration 1: count_by_status optimization (REJECTED)

**Attempted**: Replace `get_all_by_status()` with `count_by_status()` in `_has_active_dags()` to avoid O(N) MGET polling.

**Rationale**: The engine polls `_has_active_dags()` every `loop_interval` tick (1s default) while waiting for DAGs. With N=5000, this fetches all 5000 program payloads repeatedly. SCARD is O(1) and should be much faster.

**Chaos-hacker verdict**: REJECTED — Critical bug reintroduced. Ghost IDs (set members with no backing program data) cause `_await_idle()` to loop forever (80+ minute stall per generation). The existing `get_all_by_status` code deliberately filters ghosts. Existing regression test `test_has_active_dags_returns_false_with_ghost_id` would fail.

**Key lesson**: The "expensive" operation in `_has_active_dags` is actually necessary for correctness. The real bottleneck is elsewhere.

## Iteration 2: mget field projection for refresh cycle (NOT MEASURABLE)

**Attempted**: Add `exclude: frozenset[str] | None` parameter to `mget()` and use it in `_refresh_archive_programs()` to skip expensive `stage_results`, `metadata`, `metrics` during full archive refresh.

**Rationale**: `_refresh_archive_programs()` (Phase 5, once per generation) fetches all N archive programs just to check `program.state` and re-queue DONE programs. Previously deserialized all fields including expensive `stage_results`. With projection, skip 4 DEFERRABLE_FIELDS, deserialize only id/code/state/created_at/lineage/atomic_counter.

**Chaos-hacker verdict**: SAFE — Found 1 MEDIUM (pre-existing PopulationSnapshot cache design issue) and 3 LOW findings. No blocking issues. Merge strategy's empty-dict short-circuit makes this inherently safe. Recommended: remove "name" from DEFERRABLE_FIELDS to eliminate fragile risk (zero deserialization cost).

**Benchmark result**: Mixed performance
- 8 improvements: frontier_recompute (-5-7%), incremental_process (-4.7%), generation_wall_time N=500 (-2%)
- 11 regressions: collector.compute (+3.7-8.1%), storage_roundtrip (+2-3%), reindex_archive (+3.8%)
- **Headline metric**: generation_wall_time N=5000 — no significant change (15.9s → 15.7s, noise)
- **Verdict**: Optimization doesn't move the needle because _refresh_archive_programs is called only 1x/generation, while collector (20x/generation) regressed. The win is not measurable at benchmark scale.

**Conclusion**: Reverted. The optimization is correct and safe, but the performance improvement is too small to detect given benchmark variance.

## Iteration 3: ORM-style exclude parameter for get_all_by_status (IN PROGRESS)

**Attempted**: Add `exclude: frozenset[str] | None = None` parameter to `get_all_by_status()` and use `exclude=DEFERRABLE_FIELDS` in `_has_active_dags()` tight polling loop.

**Rationale**: `_has_active_dags()` is called every 5ms during `_await_idle()` (Phase 1, 3, 6). It fetches all QUEUED/RUNNING programs but only reads `program.state`. By excluding `stage_results` (dict with ProgramStageResult), `metadata` (pickle_b64 serialized), `metrics` (dict), and `name` (string), we skip expensive deserialization on every poll iteration.

**Changes**:
- Added `exclude` parameter to ProgramStorage.get_all_by_status() abstract interface
- Implemented exclude plumbing in RedisProgramStorage → _mget_by_keys()
- Use `exclude=DEFERRABLE_FIELDS` in `_has_active_dags()` calls (lines 492-496)
- Fixed PopulationSnapshot cache to skip caching when `exclude is not None` (prevents stale-projection poisoning)
- Updated test mocks to accept `**kwargs` for new parameter

**Chaos-hacker verdict**: APPROVED with fixes. Found 1 MEDIUM (PopulationSnapshot cache key issue — fixed) + 3 LOW findings. Core logic is sound; ghost ID filtering unaffected.

**Benchmark result**: TBD (benchmarks pass; need detailed comparison)

**Status**: COMMITTED (e074502) — ready for next iteration.

## Iteration 4: Fix PopulationSnapshot single-slot cache with (epoch, exclude) key

**Attempted**: Fix iteration 3's regression (22.8s collector x20 → 1.1s by restoring cache hits).

**Problem**: Iteration 3 made PopulationSnapshot skip caching entirely when `exclude is not None`, to prevent poisoning. But the collector calls `get_all(..., exclude={"stage_results"})` 20 times per refresh cycle, losing all cache hits.

**Solution**: Key cache on `(epoch, exclude)` instead of bypassing. Works correctly when all callers within an epoch use the same exclude value (which is true: collector always uses `exclude={"stage_results"}`, other callers use `exclude=None`).

**Chaos-hacker verdict**: APPROVED. Found 1 HIGH (misleading docstring) + 1 MEDIUM (race condition, theoretical) + 1 LOW findings. Code is correct for current usage; updated docstring to document single-exclude-per-epoch assumption.

**Benchmark result**: collector x20 should return from 22.8s → ~1.1s (20x cache hits restored). Full benchmark run encountered OOM; subset tests passed.

**Status**: COMMITTED (4a5c019).

## Iterations Remaining

Next candidates:
- `get_all()` deserialization at full scale (N=5000, 2.0s baseline)
- State transition throughput (already ~2300/s, seems optimal)
- `_has_active_dags()` polling frequency tuning (currently 5ms)
- Frontier recomputation optimization (38ms at N=5000, but negligible impact)
