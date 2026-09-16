#!/usr/bin/env python3
"""Keep one physical GPU reserved for an explicitly named experiment tree."""

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import secrets
import signal
import stat
import subprocess
import time

import psutil

LEASE_ENVIRONMENT_KEY = "AUTORESEARCH_GPU_LEASE_TOKEN"


def _gpu_uuid(index: int) -> str:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    gpu_by_index = {
        int(raw_index.strip()): uuid.strip()
        for line in completed.stdout.splitlines()
        for raw_index, uuid in [line.split(",", 1)]
    }
    if index not in gpu_by_index:
        raise RuntimeError(f"Physical GPU index {index} does not exist")
    return gpu_by_index[index]


def _occupants(uuid: str) -> list[int]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    pids: list[int] = []
    for line in completed.stdout.splitlines():
        process_uuid, raw_pid = (part.strip() for part in line.split(",", 1))
        if process_uuid == uuid:
            pids.append(int(raw_pid))
    return pids


def _command(pid: int) -> str | None:
    try:
        return (
            Path(f"/proc/{pid}/cmdline")
            .read_bytes()
            .replace(b"\0", b" ")
            .decode(errors="replace")
            .strip()
        )
    except FileNotFoundError:
        return None
    except OSError as exc:
        return f"<unreadable command: {type(exc).__name__}: {exc}>"


def _path_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _allowed_locations(cwd: Path, arguments: list[bytes], roots: list[Path]) -> bool:
    if not any(_path_within(cwd, root) for root in roots):
        return False
    for raw_argument in arguments:
        argument = raw_argument.decode(errors="replace")
        if not argument.startswith("/"):
            continue
        path = Path(argument)
        if any(_path_within(path, root) for root in roots):
            return True
    return False


def _allowed_by_process_location(pid: int, roots: list[Path]) -> bool:
    try:
        cwd = Path(f"/proc/{pid}/cwd").resolve()
        arguments = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return _allowed_locations(cwd, arguments, roots)


def _environment_has_token(environ: bytes, expected: str) -> bool:
    entry = f"{LEASE_ENVIRONMENT_KEY}={expected}".encode()
    return entry in environ.split(b"\0")


def _process_has_token(pid: int, expected: str) -> bool:
    try:
        environ = Path(f"/proc/{pid}/environ").read_bytes()
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return _environment_has_token(environ, expected)


def _ancestry_allowed(pid: int, parent_pids: list[int], owners: set[int]) -> bool:
    return pid in owners or any(parent_pid in owners for parent_pid in parent_pids)


def _process_has_allowed_ancestor(pid: int, owners: set[int]) -> bool:
    try:
        process = psutil.Process(pid)
        parent_pids = [parent.pid for parent in process.parents()]
    except psutil.NoSuchProcess:
        return True
    except psutil.Error:
        return False
    return _ancestry_allowed(pid, parent_pids, owners)


def _authorized_occupant(
    *, location_allowed: bool, token_allowed: bool, ancestry_allowed: bool
) -> bool:
    """Authorize token-bearing workers even after a launcher reparents them.

    Torch elastic and the DAG executor can intentionally detach a worker from
    the original tmux pane. The secret lease token remains inherited across
    that transition and, together with the allowed repository root, is the
    durable authorization. Ancestry remains diagnostic evidence for attached
    workers but is not a lifetime requirement.
    """

    del ancestry_allowed
    return location_allowed and token_allowed


def _load_or_create_token(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(descriptor, (secrets.token_hex(32) + "\n").encode())
        finally:
            os.close(descriptor)
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise PermissionError(f"GPU lease token permissions must be 0600: {path}")
    token = path.read_text(encoding="utf-8").strip()
    if len(token) != 64 or any(
        character not in "0123456789abcdef" for character in token
    ):
        raise ValueError(f"GPU lease token is invalid: {path}")
    return token


def _service_from_cgroup(lines: list[str]) -> str | None:
    services: list[str] = []
    for line in lines:
        path = line.split(":", 2)[-1]
        for component in path.split("/"):
            if component.endswith(".service"):
                if not component.startswith("user@"):
                    services.append(component)
    # A user service cgroup is nested below user@<uid>.service. The innermost
    # unit owns the process; never stop the outer user manager.
    return services[-1] if services else None


def _user_service(pid: int) -> str | None:
    try:
        lines = Path(f"/proc/{pid}/cgroup").read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return None
    return _service_from_cgroup(lines)


def _tmux_pane_root(pid: int) -> int | None:
    """Find the top process in the tmux pane that owns a detached GPU child."""

    try:
        current = psutil.Process(pid)
        for parent in current.parents():
            try:
                name = parent.name()
            except psutil.NoSuchProcess:
                return None
            if name == "tmux: server":
                return current.pid
            current = parent
    except psutil.NoSuchProcess:
        return None
    return None


def _stop(pid: int, *, runtime_mask_user_service: bool = False) -> str:
    service = _user_service(pid)
    if service is not None:
        try:
            subprocess.run(["systemctl", "--user", "stop", service], check=True)
            action = f"stopped user service {service}"
            if runtime_mask_user_service:
                try:
                    subprocess.run(
                        ["systemctl", "--user", "mask", "--runtime", service],
                        check=True,
                    )
                    action += " and runtime-masked it"
                except subprocess.CalledProcessError as exc:
                    action += f"; runtime mask failed with exit {exc.returncode}"
            return action
        except subprocess.CalledProcessError:
            pass
    root_pid = _tmux_pane_root(pid)
    targets = [pid, *([root_pid] if root_pid is not None else [])]
    process_groups: set[int] = set()
    for target in targets:
        try:
            process_groups.add(os.getpgid(target))
        except ProcessLookupError:
            continue
    for process_group in process_groups:
        os.killpg(process_group, signal.SIGTERM)
    if not process_groups:
        return "process already exited"
    return "terminated process groups " + ",".join(map(str, sorted(process_groups)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--allow-root", type=Path, action="append", required=True)
    parser.add_argument(
        "--allow-ancestor-pid", type=int, action="append", required=True
    )
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--runtime-mask-conflicting-services", action="store_true")
    args = parser.parse_args()
    uuid = _gpu_uuid(args.gpu_index)
    allowed_roots = [root.resolve() for root in args.allow_root]
    allowed_ancestors = set(args.allow_ancestor_pid)
    missing_ancestors = sorted(
        pid for pid in allowed_ancestors if pid <= 1 or not psutil.pid_exists(pid)
    )
    if missing_ancestors:
        raise ValueError(f"GPU lease ancestor PIDs are not alive: {missing_ancestors}")
    token_path = args.token_file.resolve()
    if not any(_path_within(token_path, root) for root in allowed_roots):
        raise ValueError("GPU lease token file must be under an allowed root")
    lease_token = _load_or_create_token(token_path)
    print(
        f"{datetime.now().astimezone().isoformat()} reserving physical GPU "
        f"{args.gpu_index} ({uuid}); allowed_roots={allowed_roots}; "
        f"allowed_ancestors={sorted(allowed_ancestors)}",
        flush=True,
    )
    while True:
        try:
            occupants = _occupants(uuid)
        except Exception as exc:
            print(
                f"{datetime.now().astimezone().isoformat()} gpu poll failed: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            if args.once:
                return 2
            time.sleep(args.poll_seconds)
            continue
        for pid in occupants:
            command = _command(pid)
            if command is None:
                continue
            location_allowed = _allowed_by_process_location(pid, allowed_roots)
            token_allowed = _process_has_token(pid, lease_token)
            ancestry_allowed = _process_has_allowed_ancestor(pid, allowed_ancestors)
            if _authorized_occupant(
                location_allowed=location_allowed,
                token_allowed=token_allowed,
                ancestry_allowed=ancestry_allowed,
            ):
                continue
            prefix = (
                f"{datetime.now().astimezone().isoformat()} conflict pid={pid} "
                f"location_allowed={location_allowed} token_allowed={token_allowed} "
                f"ancestry_allowed={ancestry_allowed} "
                f"command={command!r}"
            )
            if args.dry_run:
                print(f"{prefix}; dry-run", flush=True)
            else:
                try:
                    action = _stop(
                        pid,
                        runtime_mask_user_service=(
                            args.runtime_mask_conflicting_services
                        ),
                    )
                except Exception as exc:
                    action = f"enforcement failed: {type(exc).__name__}: {exc}"
                print(f"{prefix}; {action}", flush=True)
        if args.once:
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
