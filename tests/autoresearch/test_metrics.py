from __future__ import annotations

from pathlib import Path

import autoresearch.metrics as metrics


def test_read_core_metric_falls_back_to_completed_console_log(
    tmp_path: Path,
    monkeypatch,
) -> None:
    event_file = tmp_path / "events.out.tfevents.test"
    event_file.write_bytes(b"not an event file")
    train_log = tmp_path / "train.stderr.log"
    train_log.write_text(
        "[batch=30518/30518]:\n"
        "\t Train metrics_gauntlet/core: 0.3554\n"
        "\t Train metrics_gauntlet/default_average: 0.3554\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        metrics,
        "_event_accumulator",
        lambda _: (_ for _ in ()).throw(RuntimeError("reader unavailable")),
    )

    value, tag, source = metrics.read_core_metric(tmp_path)

    assert value == 0.3554
    assert tag == "metrics_gauntlet/core"
    assert source == train_log


def test_console_fallback_uses_latest_core_value(tmp_path: Path) -> None:
    train_log = tmp_path / "train.stderr.log"
    train_log.write_text(
        "\t Train metrics_gauntlet/core: 0.3000\n"
        "\t Train metrics/metrics_gauntlet/core: 0.3554\n",
        encoding="utf-8",
    )

    value, tag, source = metrics.read_core_metric(tmp_path)

    assert value == 0.3554
    assert tag == "metrics/metrics_gauntlet/core"
    assert source == train_log
