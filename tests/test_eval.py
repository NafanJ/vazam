"""
test_eval.py — benchmark harness (eval.py)

Covers clip discovery, name resolution (against the fake Supabase db
fixture), scoring math, and the runner with a mocked pipeline. No ML, no
network, no real audio decoding.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from eval import (
    EvalClip,
    ModeOutcome,
    build_report,
    discover_clips,
    rank_of,
    resolve_actor_ids,
    resolve_show_ids,
    run_eval,
    summarize,
)
from pipeline import IdentificationResult


# ── Discovery ─────────────────────────────────────────────────────────────────

def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_discover_clips_layout(tmp_path):
    _touch(tmp_path / "Cowboy Bebop" / "Steve Blum" / "clip01.wav")
    _touch(tmp_path / "Cowboy Bebop" / "Steve Blum" / "clip02.M4A")
    _touch(tmp_path / "Cowboy Bebop" / "Wendee Lee" / "clip01.mp3")
    _touch(tmp_path / "Family Guy" / "Seth MacFarlane" / "a.flac")
    _touch(tmp_path / "Cowboy Bebop" / "Steve Blum" / "notes.txt")   # ignored
    _touch(tmp_path / "stray.wav")                                   # not in layout

    clips = discover_clips(str(tmp_path))

    assert len(clips) == 4
    assert clips[0].show_title == "Cowboy Bebop"
    assert clips[0].actor_name == "Steve Blum"
    assert {c.show_title for c in clips} == {"Cowboy Bebop", "Family Guy"}


def test_discover_clips_empty(tmp_path):
    assert discover_clips(str(tmp_path)) == []


# ── Resolution against the DB ─────────────────────────────────────────────────

def test_resolve_actor_ids_case_insensitive(db):
    aid = db.add_actor("Steve Blum")
    resolved = resolve_actor_ids(db, {"steve blum", "Nobody Real"})
    assert resolved == {"steve blum": aid}


def test_resolve_show_ids_exact_title(db):
    sid = db.add_show("Cowboy Bebop", media_type="anime", year=1998)
    db.add_show("Cowboy Bebop: The Movie", media_type="anime", year=2001)

    resolved = resolve_show_ids(db, {"cowboy bebop", "Unknown Show"})
    assert resolved == {"cowboy bebop": sid}


# ── Scoring ───────────────────────────────────────────────────────────────────

def test_rank_of():
    assert rank_of(7, [7, 2, 3]) == 1
    assert rank_of(3, [7, 2, 3]) == 3
    assert rank_of(9, [7, 2, 3]) is None
    assert rank_of(9, []) is None


def test_summarize_metrics():
    outcomes = [
        (1, ModeOutcome([1, 2, 3], top1_confident=True)),    # top-1, claimed, correct
        (2, ModeOutcome([9, 2, 3], top1_confident=True)),    # top-3, claimed, WRONG
        (3, ModeOutcome([9, 8, 7, 6, 3], top1_confident=False)),  # top-5 only
        (4, ModeOutcome([9, 8, 7], top1_confident=False)),    # miss
    ]
    s = summarize(outcomes)
    assert s["clips"] == 4
    assert s["top1"] == pytest.approx(1 / 4)
    assert s["top3"] == pytest.approx(2 / 4)
    assert s["top5"] == pytest.approx(3 / 4)
    assert s["confident_claimed"] == 2
    assert s["confident_precision"] == pytest.approx(1 / 2)


def test_summarize_empty():
    assert summarize([]) == {"clips": 0}


def test_summarize_no_confident_claims():
    s = summarize([(1, ModeOutcome([1], top1_confident=False))])
    assert s["confident_precision"] is None


# ── Runner ────────────────────────────────────────────────────────────────────

def _result(actor_id: int, confidence: float = 0.9) -> IdentificationResult:
    return IdentificationResult(actor_id, f"Actor {actor_id}", "Natural Voice", confidence)


def test_run_eval_global_and_show_aware(db):
    blum_id = db.add_actor("Steve Blum")
    show_id = db.add_show("Cowboy Bebop")

    clips = [EvalClip("/fake/clip01.wav", "Steve Blum", "Cowboy Bebop")]

    fake_pipeline = MagicMock()
    # Global call: wrong actor on top; show-aware call: correct actor wins
    fake_pipeline.identify.side_effect = [
        [_result(999), _result(blum_id)],
        [_result(blum_id)],
    ]

    run = run_eval(fake_pipeline, db, clips)

    assert len(run.records) == 1
    assert run.skipped == []
    rec = run.records[0]
    assert rec.truth_actor_id == blum_id
    assert rank_of(blum_id, rec.global_outcome.ranked_actor_ids) == 2
    assert rank_of(blum_id, rec.show_outcome.ranked_actor_ids) == 1

    # Show-aware call must pass the resolved show_id
    _, kwargs = fake_pipeline.identify.call_args
    assert kwargs.get("show_id") == show_id


def test_run_eval_skips_unknown_actor(db):
    db.add_show("Cowboy Bebop")
    clips = [EvalClip("/fake/clip.wav", "Nobody Real", "Cowboy Bebop")]

    fake_pipeline = MagicMock()
    run = run_eval(fake_pipeline, db, clips)

    assert run.records == []
    assert len(run.skipped) == 1
    fake_pipeline.identify.assert_not_called()


def test_run_eval_unknown_show_still_evaluates_globally(db):
    blum_id = db.add_actor("Steve Blum")
    clips = [EvalClip("/fake/clip.wav", "Steve Blum", "Not In DB")]

    fake_pipeline = MagicMock()
    fake_pipeline.identify.return_value = [_result(blum_id)]

    run = run_eval(fake_pipeline, db, clips)

    assert len(run.records) == 1
    assert run.records[0].show_outcome is None
    fake_pipeline.identify.assert_called_once()   # global only, no show-aware call


def test_run_eval_identify_failure_is_skipped(db):
    db.add_actor("Steve Blum")
    clips = [EvalClip("/fake/clip.wav", "Steve Blum", "Nowhere")]

    fake_pipeline = MagicMock()
    fake_pipeline.identify.side_effect = RuntimeError("decode error")

    run = run_eval(fake_pipeline, db, clips)
    assert run.records == []
    assert "identify failed" in run.skipped[0][1]


# ── Report ────────────────────────────────────────────────────────────────────

def test_build_report_shape(db):
    blum_id = db.add_actor("Steve Blum")
    db.add_show("Cowboy Bebop")
    clips = [EvalClip("/fake/clip.wav", "Steve Blum", "Cowboy Bebop")]

    fake_pipeline = MagicMock()
    fake_pipeline.identify.return_value = [_result(blum_id, confidence=0.95)]

    run = run_eval(fake_pipeline, db, clips)
    report = build_report(run, args_used={"verify": True})

    assert report["clips_evaluated"] == 1
    assert report["global"]["top1"] == 1.0
    assert report["show_aware"]["top1"] == 1.0
    assert report["per_show"]["Cowboy Bebop"]["global"]["clips"] == 1
    assert report["args"] == {"verify": True}
    assert report["embedding_count"] == 0
