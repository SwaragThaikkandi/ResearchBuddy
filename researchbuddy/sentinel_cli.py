"""
researchbuddy-sentinel — headless surveillance + overnight self-tuning.

Runs WITHOUT the UI: load graph → sentinel scan (watched topics → triage →
inbox + digest) → optionally N autotune rounds (the Karpathy loop) → save →
exit. Designed to be fired by Windows Task Scheduler while you sleep:

    researchbuddy-sentinel --install-task --time 03:00 --autotune 20
    researchbuddy-sentinel --uninstall-task

You wake up to a digest of new papers and a measurably better-tuned
recommender — the autoresearch pattern, applied to your literature.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys

logger = logging.getLogger("sentinel")

TASK_NAME = "ResearchBuddySentinel"


def build_task_command(time_str: str = "03:00",
                       autotune_rounds: int = 0) -> list[str]:
    """The schtasks invocation that registers the nightly run."""
    run = f'"{sys.executable}" -m researchbuddy.sentinel_cli'
    if autotune_rounds > 0:
        run += f" --autotune {int(autotune_rounds)}"
    return ["schtasks", "/Create", "/TN", TASK_NAME, "/SC", "DAILY",
            "/ST", time_str, "/TR", run, "/F"]


def build_uninstall_command() -> list[str]:
    return ["schtasks", "/Delete", "/TN", TASK_NAME, "/F"]


def _apply_core_key_from_prefs() -> None:
    """Same pref the CLI/UI use — headless runs get the fast lane too."""
    try:
        from researchbuddy.core import services as svc
        from researchbuddy.core import core_fetcher
        key = (svc.load_prefs().get("core_api_key") or "").strip()
        if key and not core_fetcher.has_api_key():
            core_fetcher.set_api_key(key)
    except Exception as e:                    # pragma: no cover - defensive
        logger.debug("CORE key pref load skipped: %s", e)


def run_headless(autotune_rounds: int = 0) -> int:
    from researchbuddy.core.state_manager import load, save
    from researchbuddy.core import sentinel as sn
    from researchbuddy.core import autotune as at

    graph = load()
    if graph is None:
        print("No saved graph (~/.researchbuddy). Run ResearchBuddy first.",
              file=sys.stderr)
        return 2
    _apply_core_key_from_prefs()
    at.apply_saved_tuning(graph)

    print(f"[sentinel] scanning watches "
          f"({len(graph.all_papers())} papers in graph) ...")
    report = sn.run_scan(graph, progress=lambda s, *a: print(f"  {s}"))
    print(f"[sentinel] {report['new']} new paper(s) filed to the inbox.")
    if report.get("digest"):
        print(f"[sentinel] digest: {report['digest']}")

    # Living-graph cycle (Bayesian scout) when the user enabled it.
    from researchbuddy.core import scout as sg
    if sg.load_state().get("enabled"):
        print("[scout] running a living-graph cycle ...")
        rep = sg.run_cycle(graph, progress=lambda s, *a: print(f"  {s}"))
        if rep.get("ok"):
            print(f"[scout] +{rep['acquired']} acquired, "
                  f"-{rep['pruned']} pruned, "
                  f"{len(rep['slate'])} paper(s) on the slate.")
        else:
            print(f"[scout] skipped: {rep.get('note')}")

    if autotune_rounds > 0:
        print(f"[autotune] running {autotune_rounds} experiment round(s) ...")
        result = at.run_session(
            graph, rounds=autotune_rounds,
            progress=lambda s, *a: print(f"  {s}"))
        if result.get("ready"):
            print(f"[autotune] {result['baseline']:.4f} → "
                  f"{result['best']:.4f}; kept: "
                  f"{', '.join(result['kept']) or 'nothing'}")
        else:
            print(f"[autotune] skipped: {result.get('note')}")

    save(graph)
    print("[sentinel] graph saved. Done.")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="researchbuddy-sentinel",
        description="Headless literature surveillance + overnight self-tuning.")
    p.add_argument("--autotune", type=int, default=0, metavar="N",
                   help="run N self-tuning experiment rounds after the scan")
    p.add_argument("--install-task", action="store_true",
                   help="register a daily Windows Task Scheduler job")
    p.add_argument("--uninstall-task", action="store_true",
                   help="remove the scheduled job")
    p.add_argument("--time", default="03:00", metavar="HH:MM",
                   help="daily run time for --install-task (default 03:00)")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="[%(name)s] %(message)s")

    if args.install_task:
        if sys.platform != "win32":
            print("--install-task uses Windows Task Scheduler (schtasks). "
                  "On Linux/macOS use cron:", file=sys.stderr)
            print(f"  0 3 * * * {sys.executable} -m researchbuddy.sentinel_cli"
                  + (f" --autotune {args.autotune}" if args.autotune else ""),
                  file=sys.stderr)
            return 2
        cmd = build_task_command(args.time, args.autotune)
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            print(f"Scheduled: '{TASK_NAME}' runs daily at {args.time}"
                  + (f" with {args.autotune} autotune round(s)."
                     if args.autotune else "."))
            print("Runs even when the UI is closed. Remove with: "
                  "researchbuddy-sentinel --uninstall-task")
        else:
            print(f"schtasks failed: {res.stderr.strip()}", file=sys.stderr)
        return res.returncode

    if args.uninstall_task:
        res = subprocess.run(build_uninstall_command(),
                             capture_output=True, text=True)
        print("Removed." if res.returncode == 0
              else f"schtasks failed: {res.stderr.strip()}")
        return res.returncode

    return run_headless(autotune_rounds=args.autotune)


if __name__ == "__main__":
    raise SystemExit(main())
