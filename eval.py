"""
eval.py — Accuracy benchmark for the Vazam identification pipeline

Runs labelled audio clips through pipeline.identify() and reports top-k
accuracy, both global (open-set) and show-aware (closed-set), plus the
precision of "confident" claims. Run it before and after any pipeline or
data change — accuracy claims without this harness are vibes.

Benchmark directory layout (see benchmark/README.md for the recording
protocol — clips should be phone-mic recordings of a playing show, not
clean rips, or the numbers will flatter the pipeline):

    benchmark/
    └── Cowboy Bebop/            ← show title, must match a DB show
        ├── Steve Blum/          ← actor name, must match a DB actor
        │   ├── clip01.wav
        │   └── clip02.m4a
        └── Wendee Lee/
            └── clip01.wav

Usage
-----
    python eval.py benchmark/                   # full eval
    python eval.py benchmark/ --show "Cowboy Bebop"
    python eval.py benchmark/ --isolate         # Demucs on each clip
    python eval.py benchmark/ --no-verify       # disable window verification
    python eval.py benchmark/ --json out.json   # machine-readable results

Environment variables: SUPABASE_URL, SUPABASE_KEY, HF_TOKEN, DEVICE
(same as the API server).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
TOP_K = 5


# ── Benchmark discovery ───────────────────────────────────────────────────────

@dataclass
class EvalClip:
    path: str
    actor_name: str
    show_title: str


def discover_clips(root: str) -> list[EvalClip]:
    """Walk benchmark/<show>/<actor>/<clip.ext> and return labelled clips."""
    clips: list[EvalClip] = []
    root_path = Path(root)
    for show_dir in sorted(p for p in root_path.iterdir() if p.is_dir()):
        for actor_dir in sorted(p for p in show_dir.iterdir() if p.is_dir()):
            for f in sorted(actor_dir.iterdir()):
                if f.suffix.lower() in AUDIO_EXTENSIONS:
                    clips.append(EvalClip(
                        path=str(f),
                        actor_name=actor_dir.name,
                        show_title=show_dir.name,
                    ))
    return clips


# ── Name → id resolution ──────────────────────────────────────────────────────

def resolve_actor_ids(db, names: set[str]) -> dict[str, int]:
    """Case-insensitive actor name → id. Missing names are absent from the map."""
    all_actors = db.list_actors(limit=10000)
    by_name = {a["name"].lower(): a["id"] for a in all_actors}
    return {n: by_name[n.lower()] for n in names if n.lower() in by_name}


def resolve_show_ids(db, titles: set[str]) -> dict[str, int]:
    """Case-insensitive exact show title → id. Missing titles are absent."""
    resolved: dict[str, int] = {}
    for title in titles:
        for row in db.search_show(title):
            if row["title"].lower() == title.lower():
                resolved[title] = row["id"]
                break
    return resolved


# ── Scoring ───────────────────────────────────────────────────────────────────

@dataclass
class ModeOutcome:
    """What identify() returned for one clip in one mode (global/show-aware)."""
    ranked_actor_ids: list[int]
    top1_confident: bool


@dataclass
class ClipRecord:
    clip: EvalClip
    truth_actor_id: int
    global_outcome: ModeOutcome
    show_outcome: Optional[ModeOutcome]   # None when the show wasn't resolvable
    seconds: float = 0.0


def rank_of(truth_actor_id: int, ranked_actor_ids: list[int]) -> Optional[int]:
    """1-based rank of the true actor in the result list, or None if absent."""
    for i, aid in enumerate(ranked_actor_ids, 1):
        if aid == truth_actor_id:
            return i
    return None


def summarize(outcomes: list[tuple[int, ModeOutcome]]) -> dict:
    """Aggregate (truth_actor_id, outcome) pairs into accuracy metrics.

    confident_precision is the metric verification exists to protect: of the
    clips where the system claimed a confident match, how often was top-1 right.
    """
    n = len(outcomes)
    if n == 0:
        return {"clips": 0}

    top1 = top3 = top5 = 0
    claimed = claimed_correct = 0
    for truth, out in outcomes:
        r = rank_of(truth, out.ranked_actor_ids)
        if r is not None:
            top1 += r == 1
            top3 += r <= 3
            top5 += r <= 5
        if out.top1_confident:
            claimed += 1
            claimed_correct += r == 1

    return {
        "clips": n,
        "top1": top1 / n,
        "top3": top3 / n,
        "top5": top5 / n,
        "confident_claimed": claimed,
        "confident_precision": (claimed_correct / claimed) if claimed else None,
    }


def summarize_per_show(records: list[ClipRecord]) -> dict[str, dict]:
    by_show: dict[str, list[ClipRecord]] = {}
    for rec in records:
        by_show.setdefault(rec.clip.show_title, []).append(rec)

    out: dict[str, dict] = {}
    for show, recs in sorted(by_show.items()):
        out[show] = {
            "global": summarize([(r.truth_actor_id, r.global_outcome) for r in recs]),
            "show_aware": summarize([
                (r.truth_actor_id, r.show_outcome)
                for r in recs if r.show_outcome is not None
            ]),
        }
    return out


# ── Runner ────────────────────────────────────────────────────────────────────

@dataclass
class EvalRun:
    records: list[ClipRecord] = field(default_factory=list)
    skipped: list[tuple[EvalClip, str]] = field(default_factory=list)  # (clip, reason)
    embedding_count: int = 0


def run_eval(
    pipeline,
    db,
    clips: list[EvalClip],
    isolate: bool = False,
    verify: bool = True,
) -> EvalRun:
    """Identify every clip globally and show-aware; collect outcomes."""
    actor_ids = resolve_actor_ids(db, {c.actor_name for c in clips})
    show_ids  = resolve_show_ids(db, {c.show_title for c in clips})

    run = EvalRun(embedding_count=db.get_embedding_count())

    for i, clip in enumerate(clips, 1):
        truth = actor_ids.get(clip.actor_name)
        if truth is None:
            run.skipped.append((clip, f"actor '{clip.actor_name}' not in DB"))
            continue

        print(f"[{i}/{len(clips)}] {clip.show_title} / {clip.actor_name} / {Path(clip.path).name}")

        start = time.perf_counter()
        try:
            global_results = pipeline.identify(
                clip.path, top_k=TOP_K, isolate=isolate, verify=verify,
            )
        except Exception as exc:
            run.skipped.append((clip, f"identify failed: {exc}"))
            continue

        show_id = show_ids.get(clip.show_title)
        show_outcome: Optional[ModeOutcome] = None
        if show_id is not None:
            show_results = pipeline.identify(
                clip.path, top_k=TOP_K, isolate=isolate, show_id=show_id, verify=verify,
            )
            show_outcome = ModeOutcome(
                ranked_actor_ids=[r.actor_id for r in show_results],
                top1_confident=bool(show_results and show_results[0].confident),
            )

        run.records.append(ClipRecord(
            clip=clip,
            truth_actor_id=truth,
            global_outcome=ModeOutcome(
                ranked_actor_ids=[r.actor_id for r in global_results],
                top1_confident=bool(global_results and global_results[0].confident),
            ),
            show_outcome=show_outcome,
            seconds=time.perf_counter() - start,
        ))

    return run


# ── Reporting ─────────────────────────────────────────────────────────────────

def _pct(x: Optional[float]) -> str:
    return "  n/a" if x is None else f"{100 * x:5.1f}%"


def build_report(run: EvalRun, args_used: dict) -> dict:
    """Full machine-readable report (also drives the console output)."""
    records = run.records
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "args": args_used,
        "embedding_count": run.embedding_count,
        "clips_evaluated": len(records),
        "clips_skipped": [
            {"path": c.path, "reason": reason} for c, reason in run.skipped
        ],
        "avg_seconds_per_clip": (
            round(sum(r.seconds for r in records) / len(records), 2) if records else None
        ),
        "global": summarize([(r.truth_actor_id, r.global_outcome) for r in records]),
        "show_aware": summarize([
            (r.truth_actor_id, r.show_outcome)
            for r in records if r.show_outcome is not None
        ]),
        "per_show": summarize_per_show(records),
    }


def print_report(report: dict) -> None:
    print("\n── Vazam Eval ────────────────────────────────────────────")
    print(f"  Clips: {report['clips_evaluated']} evaluated, "
          f"{len(report['clips_skipped'])} skipped  "
          f"({report['embedding_count']} embeddings in DB, "
          f"avg {report['avg_seconds_per_clip']}s/clip)")

    for label, key in (("Global (open-set)", "global"),
                       ("Show-aware (closed-set)", "show_aware")):
        s = report[key]
        if not s.get("clips"):
            print(f"\n  {label}: no clips")
            continue
        print(f"\n  {label} — {s['clips']} clips")
        print(f"    top-1 {_pct(s['top1'])}   top-3 {_pct(s['top3'])}   top-5 {_pct(s['top5'])}")
        print(f"    confident: {s['confident_claimed']}/{s['clips']} claimed, "
              f"precision {_pct(s['confident_precision'])}")

    if report["per_show"]:
        print("\n  Per show (top-1, show-aware / global):")
        for show, s in report["per_show"].items():
            sa, gl = s["show_aware"], s["global"]
            sa_str = _pct(sa.get("top1")) if sa.get("clips") else "  n/a"
            print(f"    {show:<30} {sa_str} / {_pct(gl['top1'])}  ({gl['clips']} clips)")

    for clip_info in report["clips_skipped"]:
        print(f"  ⚠ skipped {clip_info['path']}: {clip_info['reason']}")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark identification accuracy against labelled clips",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("benchmark_dir", help="Root of the benchmark/<show>/<actor>/ tree")
    parser.add_argument("--show", default="", metavar="TITLE",
                        help="Only evaluate clips for one show")
    parser.add_argument("--isolate", action="store_true",
                        help="Run Demucs vocal isolation on each clip (slow)")
    parser.add_argument("--no-verify", action="store_true",
                        help="Disable multi-window verification")
    parser.add_argument("--json", default="", metavar="PATH",
                        help="Also write the full report as JSON")
    args = parser.parse_args()

    clips = discover_clips(args.benchmark_dir)
    if args.show:
        clips = [c for c in clips if c.show_title.lower() == args.show.lower()]
    if not clips:
        print(f"No audio clips found under {args.benchmark_dir} "
              "(expected benchmark/<show>/<actor>/<clip.wav>)")
        return

    from db import VazamDB
    from pipeline import VazamPipeline

    hf_token = os.environ.get("HF_TOKEN", "")
    db = VazamDB()
    pipeline = VazamPipeline(
        db=db,
        hf_token=hf_token,
        device=os.environ.get("DEVICE", "") or None,
        use_vad=bool(hf_token),
        use_diarization=False,
    )

    run = run_eval(pipeline, db, clips,
                   isolate=args.isolate, verify=not args.no_verify)

    report = build_report(run, args_used={
        "benchmark_dir": args.benchmark_dir,
        "show": args.show or None,
        "isolate": args.isolate,
        "verify": not args.no_verify,
    })
    print_report(report)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report written to {args.json}")

    db.close()


if __name__ == "__main__":
    main()
