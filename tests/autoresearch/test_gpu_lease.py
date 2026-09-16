from __future__ import annotations

from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[2]
LEASE = runpy.run_path(str(ROOT / "tools/enforce-gpu-lease.py"))


def test_gpu_lease_selects_innermost_user_service() -> None:
    service_from_cgroup = LEASE["_service_from_cgroup"]

    assert (
        service_from_cgroup(
            [
                "0::/user.slice/user-123.slice/user@123.service/app.slice/"
                "autoresearch-rmm.service"
            ]
        )
        == "autoresearch-rmm.service"
    )


def test_gpu_lease_accepts_process_without_service() -> None:
    service_from_cgroup = LEASE["_service_from_cgroup"]

    assert service_from_cgroup(["0::/user.slice/session-42.scope"]) is None
    assert (
        service_from_cgroup(
            ["0::/user.slice/user-123.slice/user@123.service/tmux-spawn.scope"]
        )
        is None
    )


def test_gpu_lease_canonicalizes_allowed_paths(tmp_path: Path) -> None:
    path_within = LEASE["_path_within"]
    root = tmp_path / "autoresearch-system"
    root.mkdir()

    assert path_within(root / "runs/source/train.py", root) is True
    assert path_within(root / "../autoresearch-rmm/train.py", root) is False


def test_gpu_lease_requires_allowed_cwd_and_command_path(tmp_path: Path) -> None:
    allowed = tmp_path / "autoresearch-system"
    allowed.mkdir()
    external = tmp_path / "autoresearch-rmm"
    external.mkdir()
    arguments = [
        str(allowed / "runs" / "fla-python").encode(),
        str(external / "train.py").encode(),
    ]

    assert LEASE["_allowed_locations"](external, arguments, [allowed]) is False
    assert LEASE["_allowed_locations"](allowed, arguments, [allowed]) is True


def test_gpu_lease_token_is_private_stable_and_required(tmp_path: Path) -> None:
    token_path = tmp_path / "runs" / "gpu1-lease.token"
    token = LEASE["_load_or_create_token"](token_path)

    assert len(token) == 64
    assert token_path.stat().st_mode & 0o077 == 0
    assert LEASE["_load_or_create_token"](token_path) == token
    assert LEASE["_environment_has_token"](
        f"OTHER=x\0AUTORESEARCH_GPU_LEASE_TOKEN={token}\0".encode(), token
    )
    assert not LEASE["_environment_has_token"](
        b"AUTORESEARCH_GPU_LEASE_TOKEN=wrong\0", token
    )


def test_gpu_lease_requires_descendant_of_registered_owner() -> None:
    ancestry_allowed = LEASE["_ancestry_allowed"]

    assert ancestry_allowed(30, [20, 10, 1], {10}) is True
    assert ancestry_allowed(10, [], {10}) is True
    assert ancestry_allowed(30, [20, 1], {10}) is False


def test_token_and_location_survive_worker_reparenting() -> None:
    authorized = LEASE["_authorized_occupant"]

    assert authorized(
        location_allowed=True,
        token_allowed=True,
        ancestry_allowed=False,
    )
    assert not authorized(
        location_allowed=True,
        token_allowed=False,
        ancestry_allowed=True,
    )
    assert not authorized(
        location_allowed=False,
        token_allowed=True,
        ancestry_allowed=True,
    )
