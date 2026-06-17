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

Negatives (specificity): an optional `benchmark/_negatives/` folder holds clips
that should NOT match anyone in the DB — random noise, music, or real voices of
characters that were never logged. They have no ground-truth actor; they measure
the other half of the system — does it stay quiet when it should? Organize them
into subfolders (e.g. `_negatives/noise/`, `_negatives/unknown_voices/`) for a
per-category breakdown. The report shows the false-positive rate (how often a
negative produced a confident/possible claim — want ~0) and the score gap between
correct positive matches and negative top-scores, which is what you calibrate the
thresholds against.

    benchmark/
    └── _negatives/
        ├── noise/            ← traffic, music, room tone …
        │   └── clip01.wav
        └── unknown_voices/   ← real voices not in the DB
            └── clip01.m4a

Usage
-----
    python eval.py benchmark/                   # full eval (positives + negatives)
    python eval.py benchmark/ --show "Cowboy Bebop"
    python eval.py benchmark/ --isolate         # Demucs on each clip
    python eval.py benchmark/ --no-verify       # disable window verification
    python eval.py benchmark/ --no-negatives    # skip the _negatives/ folder
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

from pipeline import CONFIDENT_THRESHOLD, POSSIBLE_THRESHOLD

load_dotenv()

AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
TOP_K = 5
NEGATIVES_DIRNAME = "_negatives"


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
    # Skip _negatives/ and other "_"/"."-prefixed dirs — they aren't shows.
    for show_dir in sorted(
        p for p in root_path.iterdir()
        if p.is_dir() and not p.name.startswith(("_", "."))
    ):
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
    top1_confidence: Optional[float] = None   # similarity of the rank-1 match


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


# ── Negatives (specificity / false-positive rate) ─────────────────────────────

@dataclass
class NegativeRecord:
    """Outcome of running a should-not-match clip through identify() (global)."""
    path: str
    category: str                       # subfolder under _negatives/, or "(root)"
    top1_confidence: Optional[float]    # similarity of the (unwanted) top match
    top1_level: str                     # 'confident' | 'possible' | 'none' | 'error'
    top1_label: str                     # "Actor as Character" of the top match, or "—"
    seconds: float = 0.0


def discover_negatives(root: str) -> list[tuple[str, str]]:
    """Return (path, category) for every audio clip under <root>/_negatives/.

    category is the immediate subfolder name (e.g. 'noise'), or '(root)' for clips
    sitting directly in _negatives/.
    """
    neg_root = Path(root) / NEGATIVES_DIRNAME
    if not neg_root.is_dir():
        return []
    found: list[tuple[str, str]] = []
    for f in sorted(neg_root.rglob("*")):
        if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS:
            rel = f.relative_to(neg_root)
            category = rel.parts[0] if len(rel.parts) > 1 else "(root)"
            found.append((str(f), category))
    return found


def run_negatives(
    pipeline,
    negatives: list[tuple[str, str]],
    isolate: bool = False,
    verify: bool = True,
) -> list[NegativeRecord]:
    """Identify each negative clip globally; record the (unwanted) top match."""
    records: list[NegativeRecord] = []
    for i, (path, category) in enumerate(negatives, 1):
        print(f"[neg {i}/{len(negatives)}] {category} / {Path(path).name}")
        start = time.perf_counter()
        try:
            results = pipeline.identify(path, top_k=TOP_K, isolate=isolate, verify=verify)
        except Exception as exc:
            records.append(NegativeRecord(
                path, category, None, "error", f"identify failed: {exc}",
                time.perf_counter() - start,
            ))
            continue
        top = results[0] if results else None
        records.append(NegativeRecord(
            path=path,
            category=category,
            top1_confidence=top.confidence if top else None,
            top1_level=top.to_dict()["match_level"] if top else "none",
            top1_label=f"{top.actor_name} as {top.character_name}" if top else "—",
            seconds=time.perf_counter() - start,
        ))
    return records


def _neg_counts(recs: list[NegativeRecord]) -> dict:
    """Shared FP tally over a set of (non-error) negative records.

    Does not emit a 'clips' key — the caller sets that (total vs. per-category),
    so the FP rates here are always over len(recs).
    """
    n = len(recs)
    confident_fp = sum(r.top1_level == "confident" for r in recs)
    claimed_fp = sum(r.top1_level in ("confident", "possible") for r in recs)
    scores = [r.top1_confidence for r in recs if r.top1_confidence is not None]
    return {
        "confident_fp": confident_fp,
        "confident_fp_rate": confident_fp / n if n else None,
        "claimed_fp": claimed_fp,            # any claim at or above POSSIBLE_THRESHOLD
        "claimed_fp_rate": claimed_fp / n if n else None,
        "score_max": round(max(scores), 4) if scores else None,
        "score_mean": round(sum(scores) / len(scores), 4) if scores else None,
    }


def summarize_negatives(records: list[NegativeRecord]) -> dict:
    """False-positive metrics over the negative clips, overall and per category.

    `clips` = total negatives seen, `errors` = identify() failures (excluded from
    the rates), `scored` = clips that produced a usable result. FP rates are over
    `scored`.
    """
    scored = [r for r in records if r.top1_level != "error"]
    out = {"clips": len(records), "errors": len(records) - len(scored), "scored": len(scored)}
    if not scored:
        return out
    out.update(_neg_counts(scored))
    by_cat: dict[str, list[NegativeRecord]] = {}
    for r in scored:
        by_cat.setdefault(r.category, []).append(r)
    out["by_category"] = {
        cat: {"clips": len(recs), **_neg_counts(recs)}
        for cat, recs in sorted(by_cat.items())
    }
    return out


def positive_match_scores(records: list[ClipRecord]) -> list[float]:
    """Top-1 similarity on clips where the true actor actually won rank 1 (global)."""
    out: list[float] = []
    for r in records:
        if (rank_of(r.truth_actor_id, r.global_outcome.ranked_actor_ids) == 1
                and r.global_outcome.top1_confidence is not None):
            out.append(r.global_outcome.top1_confidence)
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
                top1_confidence=show_results[0].confidence if show_results else None,
            )

        run.records.append(ClipRecord(
            clip=clip,
            truth_actor_id=truth,
            global_outcome=ModeOutcome(
                ranked_actor_ids=[r.actor_id for r in global_results],
                top1_confident=bool(global_results and global_results[0].confident),
                top1_confidence=global_results[0].confidence if global_results else None,
            ),
            show_outcome=show_outcome,
            seconds=time.perf_counter() - start,
        ))

    return run


# ── Reporting ─────────────────────────────────────────────────────────────────

def _pct(x: Optional[float]) -> str:
    return "  n/a" if x is None else f"{100 * x:5.1f}%"


def _score_stats(scores: list[float], extreme: str) -> dict:
    """min/mean (positives) or max/mean (negatives) over a score list."""
    if not scores:
        return {"n": 0, extreme: None, "mean": None}
    return {
        "n": len(scores),
        extreme: round(min(scores) if extreme == "min" else max(scores), 4),
        "mean": round(sum(scores) / len(scores), 4),
    }


def build_report(
    run: EvalRun,
    args_used: dict,
    neg_records: Optional[list[NegativeRecord]] = None,
) -> dict:
    """Full machine-readable report (also drives the console output)."""
    records = run.records
    neg_records = neg_records or []
    pos_scores = positive_match_scores(records)
    neg_scores = [r.top1_confidence for r in neg_records
                  if r.top1_level != "error" and r.top1_confidence is not None]
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
        "negatives": summarize_negatives(neg_records),
        "negatives_detail": [
            {"path": r.path, "category": r.category, "score": r.top1_confidence,
             "level": r.top1_level, "matched": r.top1_label}
            for r in neg_records
        ],
        "calibration": {
            "confident_threshold": CONFIDENT_THRESHOLD,
            "possible_threshold": POSSIBLE_THRESHOLD,
            "positive_correct_scores": _score_stats(pos_scores, "min"),
            "negative_top_scores": _score_stats(neg_scores, "max"),
        },
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

    neg = report.get("negatives", {})
    if neg.get("scored"):
        scored = neg["scored"]
        print(f"\n  Negatives (should not match) — {scored} clips"
              + (f", {neg['errors']} errored" if neg.get("errors") else ""))
        print(f"    confident false-positives: {neg['confident_fp']}/{scored} "
              f"({_pct(neg['confident_fp_rate'])})   ← want 0")
        print(f"    any claim (≥ possible):    {neg['claimed_fp']}/{scored} "
              f"({_pct(neg['claimed_fp_rate'])})")
        print(f"    negative top-score: max {neg['score_max']}  mean {neg['score_mean']}")
        if neg.get("by_category"):
            for cat, c in neg["by_category"].items():
                print(f"      {cat:<20} confident {c['confident_fp']}/{c['clips']}, "
                      f"any {c['claimed_fp']}/{c['clips']}, max {c['score_max']}")

        cal = report["calibration"]
        pos, negs = cal["positive_correct_scores"], cal["negative_top_scores"]
        print("\n  Calibration (raise the bar into the gap between these):")
        print(f"    correct positive matches: min {pos['min']}  mean {pos['mean']}  (n={pos['n']})")
        print(f"    negative top-scores:      max {negs['max']}  mean {negs['mean']}  (n={negs['n']})")
        print(f"    current thresholds: confident {cal['confident_threshold']}  "
              f"possible {cal['possible_threshold']}")

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
    parser.add_argument("--no-negatives", action="store_true",
                        help="Skip the _negatives/ specificity folder")
    parser.add_argument("--json", default="", metavar="PATH",
                        help="Also write the full report as JSON")
    args = parser.parse_args()

    clips = discover_clips(args.benchmark_dir)
    if args.show:
        clips = [c for c in clips if c.show_title.lower() == args.show.lower()]
    negatives = [] if args.no_negatives else discover_negatives(args.benchmark_dir)
    if not clips and not negatives:
        print(f"No audio clips found under {args.benchmark_dir} "
              "(expected benchmark/<show>/<actor>/<clip.wav>, "
              "and/or benchmark/_negatives/<clip>)")
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

    neg_records = run_negatives(pipeline, negatives,
                                isolate=args.isolate, verify=not args.no_verify)

    report = build_report(run, args_used={
        "benchmark_dir": args.benchmark_dir,
        "show": args.show or None,
        "isolate": args.isolate,
        "verify": not args.no_verify,
        "negatives": len(negatives),
    }, neg_records=neg_records)
    print_report(report)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report written to {args.json}")

    db.close()


if __name__ == "__main__":
    main()
