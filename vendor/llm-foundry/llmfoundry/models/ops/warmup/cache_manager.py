import json
import os
import shutil
import sys
import threading
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

interval: int = 60 * 30
_MAX_LOCAL_RANKS: int = 8


def _copytree_multithreaded(src, dst, symlinks=False, ignore=None, copy_function=shutil.copy2,
                            ignore_dangling_symlinks=False, dirs_exist_ok=False, max_workers=8):
    """Recursively copy a directory tree using multiple threads."""
    sys.audit("shutil.copytree", src, dst)
    entries = list(os.scandir(src))
    os.makedirs(dst, exist_ok=dirs_exist_ok)
    errors = []

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for srcentry in entries:
            srcname = os.path.join(src, srcentry.name)
            dstname = os.path.join(dst, srcentry.name)
            if srcentry.is_dir():
                futures.append(executor.submit(
                    _copytree_multithreaded,
                    srcname, dstname,
                    symlinks=symlinks, ignore=ignore, copy_function=copy_function,
                    ignore_dangling_symlinks=ignore_dangling_symlinks,
                    dirs_exist_ok=dirs_exist_ok
                ))
            else:
                futures.append(executor.submit(
                    copy_function, srcname, dstname
                ))

        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(f"Failed to copy: {e}")
                raise

    try:
        shutil.copystat(src, dst)
    except OSError as why:
        if getattr(why, 'winerror', None) is None:
            errors.append((src, dst, str(why)))
    if errors:
        raise Exception(errors)
    return dst


def _rewrite_triton_group_files(cache_root: str, rank: int) -> None:
    """
    Rewrite Triton FileCacheManager __grp__* files in-place after cache relocation.

    Triton FileCacheManager stores absolute child_paths in group files.
    After copying cache to a new absolute path, these paths become stale.
    """
    root = Path(cache_root)
    if not root.exists():
        print(f"[cache][rank={rank}] Triton: cache root does not exist, skip rewrite: {cache_root}")
        return

    rewritten_groups = 0
    rewritten_children = 0

    for grp_path in root.rglob("__grp__*"):
        if not grp_path.is_file():
            continue

        try:
            with grp_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            print(f"[cache][rank={rank}] Triton: skip unreadable group {grp_path}: {exc}")
            continue

        child_paths = data.get("child_paths")
        if not isinstance(child_paths, dict):
            continue

        parent = grp_path.parent
        new_child_paths = {}
        changed = False

        for child_name, old_abs_path in child_paths.items():
            new_abs_path = str(parent / child_name)
            new_child_paths[child_name] = new_abs_path
            if old_abs_path != new_abs_path:
                changed = True
                rewritten_children += 1

        if not changed:
            continue

        tmp_path = grp_path.with_suffix(grp_path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump({"child_paths": new_child_paths}, f, indent=4, sort_keys=True)
            f.write("\n")
        os.replace(tmp_path, grp_path)
        rewritten_groups += 1

    print(
        f"[cache][rank={rank}] Triton: rewrote {rewritten_groups} group file(s), "
        f"{rewritten_children} child path(s) under {cache_root}"
    )


def _deepgemm_cache_manager(local_rank: int) -> None:
    global _MAX_LOCAL_RANKS

    propagate_cache = os.environ.get("PROPAGATE_DG_CACHE", "")

    if not local_rank:
        if propagate_cache:
            dg_cache_dir = os.environ.get("PROPAGATE_DG_CACHE_DIR", None)
            assert dg_cache_dir and os.path.exists(dg_cache_dir), (
                f"PROPAGATE_CACHE is set but DG_CACHE_DIR={dg_cache_dir!r} "
                + "does not exist or is not set. "
                + "Check /llmfoundry/models/ops/warmup/README.md (DeepGEMM section) "
                + "for more information."
            )
            for r in range(_MAX_LOCAL_RANKS):
                dst = f"/temp/.cache/rank_{r}/deepgemm"
                _copytree_multithreaded(dg_cache_dir, dst, dirs_exist_ok=True)
                n = sum(len(files) for _, _, files in os.walk(dst))
                print(f"[cache][rank={r}] DeepGEMM: transferred {n} file(s) from {dg_cache_dir} to {dst}")

        else:
            missing = [
                f"/temp/.cache/rank_{r}/deepgemm/cache"
                for r in range(_MAX_LOCAL_RANKS)
                if not os.path.exists(f"/temp/.cache/rank_{r}/deepgemm/cache")
            ]
            assert not missing, (
                "DeepGEMM cache directories missing: " + ", ".join(missing) + ". "
                + "Either set PROPAGATE_CACHE=1 with DG_CACHE_DIR or run warmup first. "
                + "Check /llmfoundry/models/ops/warmup/README.md (DeepGEMM section) "
                + "for more information."
            )

    deepgemm_cache_dir = f"/temp/.cache/rank_{local_rank}/deepgemm/"
    # os.makedirs(deepgemm_cache_dir, exist_ok=True)
    # assert os.path.isdir(deepgemm_cache_dir)

    os.environ["DG_JIT_CACHE_DIR"] = deepgemm_cache_dir
    n_deepgemm_files = sum(len(files) for _, _, files in os.walk(deepgemm_cache_dir))
    print(f"DeepGEMM: {n_deepgemm_files=} from {deepgemm_cache_dir}")


def _snapshot_cache(cache_dir: str) -> set[str]:
    p = Path(cache_dir)
    if not p.exists():
        return set()
    return {d.name for d in p.iterdir() if d.is_dir()}


def _watcher_loop(cache_dir: str, gather_dir: str, prop_cache: set, stop_event):
    print(f"[Watcher]: start watching {cache_dir}")
    while not stop_event.wait(interval):
        current_cache = _snapshot_cache(cache_dir)
        new_cache = current_cache - prop_cache
        if new_cache:
            for fname in new_cache:
                try:
                    shutil.copytree(os.path.join(cache_dir, fname),
                                os.path.join(gather_dir, fname), dirs_exist_ok=True)
                except Exception as e:
                    logger.warning(f"[Watcher]: Error: {e}")
                    continue
            print(f"[Watcher]: transferred {len(new_cache)} file(s) from {cache_dir} to {gather_dir}")
            prop_cache = prop_cache | new_cache
            print(f"[Watcher]:  local cache info updated. Wait {interval / 60} min until next watch session.")
        else:
            print(f"[Watcher]: new caches not found. Wait {interval / 60} min until next watch session.")


def _triton_and_deepgemm_gather_cache():
    print(f"[cache]: Start watching to cache folders.")
    gather_cache_dir = os.environ.get("DBG_CACHE_DIR", None)

    triton_cache_dir = os.environ.get("TRITON_CACHE_DIR", None)
    dg_cache_dir = os.environ.get("DG_JIT_CACHE_DIR", None)
    if not (gather_cache_dir \
            and (triton_cache_dir or dg_cache_dir)):
        return (None, None), (None, None)

    gather_triton_cache_dir = os.path.join(gather_cache_dir, "triton")
    gather_dg_cache_dir = os.path.join(gather_cache_dir, "deep_gemm")
    os.makedirs(gather_triton_cache_dir, exist_ok=True)
    os.makedirs(gather_dg_cache_dir, exist_ok=True)

    # all new compilated kernels in devices is sames
    rank = 0

    dg_watcher_thread = None
    dg_stop_event = None
    triton_watcher_thread = None
    triton_stop_event = None

    if dg_cache_dir:
        dg_cache_files = set(files for files in os.listdir(os.path.join(dg_cache_dir, 'cache')))
        tmp_dg_cache_dir = f"/temp/.cache/rank_{rank}/deepgemm/cache"
        dg_stop_event = threading.Event()
        if tmp_dg_cache_dir:
            dg_watcher_thread = threading.Thread(
                target=_watcher_loop,
                args=(tmp_dg_cache_dir, gather_dg_cache_dir,
                      dg_cache_files, dg_stop_event),
                daemon=True,
                name="cache-audit-dg-watcher",
            )
            dg_watcher_thread.start()

    if triton_cache_dir:
        triton_cache_files = set(files for files in os.listdir(triton_cache_dir))
        tmp_triton_cache_dir = f"/temp/.cache/rank_{rank}/triton/"
        if tmp_triton_cache_dir:
            triton_stop_event = threading.Event()
            triton_watcher_thread = threading.Thread(
                target=_watcher_loop,
                args=(tmp_triton_cache_dir, gather_triton_cache_dir,
                      triton_cache_files, triton_stop_event),
                daemon=True,
                name="cache-audit-triton-watcher",
            )
        triton_watcher_thread.start()

    return (dg_watcher_thread, dg_stop_event), \
           (triton_watcher_thread, triton_stop_event)
