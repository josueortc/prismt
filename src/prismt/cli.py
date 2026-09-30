"""Command-line interface: ``python -m prismt <command> [options]``.

Every command accepts ``--json``: then stdout carries exactly one JSON object (what MATLAB
reads) and all logging goes to stderr. Exit codes: 0 ok, 1 internal error, 2 problem with
the data or settings, 3 cancelled, 4 problem with the Python environment.
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from collections.abc import Callable

from prismt import __version__
from prismt.errors import EXIT_INTERNAL, EXIT_OK, EXIT_USER, PrismtError
from prismt.jsonutil import dumps

log = logging.getLogger("prismt")


def _emit(args: argparse.Namespace, payload: dict, human: str | None = None) -> None:
    if getattr(args, "json", False):
        sys.stdout.write(dumps(payload, indent=None) + "\n")
    elif human is not None:
        sys.stdout.write(human.rstrip() + "\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------------------


def cmd_doctor(args: argparse.Namespace) -> int:
    from prismt.env import doctor

    report = doctor(require=args.require, out_dir=args.out)
    lines = [f"PRISMT {report['prismt']['version']} — {'ready' if report['ok'] else 'NOT ready'}"]
    for c in report["checks"]:
        lines.append(f"  [{'ok' if c['ok'] else 'FAIL'}] {c['name']}: {c['message']}")
        if not c["ok"] and c["hint"]:
            lines.append(f"         → {c['hint']}")
    for w in report["warnings"]:
        lines.append(f"  [warning] {w['message']}")
    _emit(args, report, "\n".join(lines))
    return EXIT_OK if report["ok"] else 4


def cmd_validate(args: argparse.Namespace) -> int:
    from prismt.io import validate_file

    report = validate_file(args.data)
    if report["ok"]:
        s = report["summary"]
        human = (f"OK: {s['n_trials']} trials, {s['n_channels']} channels x {s['n_time']} time points x "
                 f"{s['n_modalities']} modalities ({', '.join(m['name'] for m in s['modalities'])})")
    else:
        human = "\n".join(f"ERROR {e['code']}: {e['message']} {e.get('hint', '')}" for e in report["errors"])
    for w in report["warnings"]:
        human += f"\nwarning: {w['message']}"
    _emit(args, report, human)
    return EXIT_OK if report["ok"] else EXIT_USER


def cmd_synth(args: argparse.Namespace) -> int:
    from prismt.data.synthetic import write_synthetic

    path = write_synthetic(args.out, args.profile, args.difficulty, args.seed)
    _emit(args, {"ok": True, "path": str(path)}, f"Wrote {path}")
    return EXIT_OK


def cmd_defaults(args: argparse.Namespace) -> int:
    from prismt.config import schema_for_matlab

    payload = schema_for_matlab()
    _emit(args, payload, dumps(payload))
    return EXIT_OK


def _load(args: argparse.Namespace) -> tuple[dict, dict]:
    import json
    from pathlib import Path

    from prismt.config import load_config

    source = json.loads(Path(args.config).read_text(encoding="utf-8"))
    return load_config(args.config), source


def cmd_check(args: argparse.Namespace) -> int:
    from prismt.run import check

    cfg, _ = _load(args)
    report = check(cfg, timing=args.timing)
    s = report["selection"]
    lines = [f"{s['n_trials']} trials selected; {report['model']['tokens_per_trial']} tokens per trial; "
             f"{report['model']['parameters']:,} parameters.",
             f"Split: {report['split']['scheme']} on {report['split']['test_on']} ({report['split']['n_folds']} fold(s))."]
    if s["class_names"]:
        lines.append("Classes: " + ", ".join(f"{n} ({c})" for n, c in zip(s["class_names"], s["class_counts"])))
    if report["estimate"]:
        e = report["estimate"]
        lines.append(f"About {e['seconds_per_epoch']} s per epoch on {e['device']}; at most {e['max_total_minutes']} min.")
    for w in report["warnings"]:
        lines.append(f"warning: {w['message']}")
    _emit(args, report, "\n".join(lines))
    return EXIT_OK


def cmd_train(args: argparse.Namespace) -> int:
    from prismt.run import run

    cfg, source = _load(args)
    out = run(cfg, run_dir=args.run_dir, only_fold=args.fold, source=source)
    import json

    metrics = json.loads((out / "metrics.json").read_text()) if (out / "metrics.json").exists() else {}
    _emit(args, {"ok": True, "run_dir": str(out), "summary": metrics.get("summary_lines", [])},
          "\n".join([f"Run folder: {out}", *metrics.get("summary_lines", [])]))
    return EXIT_OK


def cmd_summarize(args: argparse.Namespace) -> int:
    from prismt.run import summarize

    info = summarize(args.run)
    lines = [info["run_dir"]]
    if "status" in info:
        lines.append(f"state: {info['status'].get('state')}")
    lines += info.get("metrics", {}).get("summary_lines", [])
    _emit(args, info, "\n".join(lines))
    return EXIT_OK


# ---------------------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="prismt", description="PRISMT command line")
    parser.add_argument("--version", action="version", version=f"prismt {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    def add(name: str, func: Callable, help_: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_, description=help_)
        p.add_argument("--json", action="store_true", help="print one JSON object on stdout")
        p.add_argument("-v", "--verbose", action="store_true", help="more log output (stderr)")
        p.set_defaults(func=func)
        return p

    p = add("doctor", cmd_doctor, "check that this Python can run PRISMT")
    p.add_argument("--require", default="auto", choices=["auto", "cpu", "cuda", "mps"],
                   help="fail unless this device is usable")
    p.add_argument("--out", default=None, help="also check that this folder is writable")

    p = add("validate", cmd_validate, "check a PRISMT dataset file and summarize it")
    p.add_argument("data", help="dataset file (.mat written by prismt.writeDataset, .h5 or .npz)")

    p = add("synth", cmd_synth, "write a synthetic dataset with known structure")
    p.add_argument("--profile", default="fast", choices=["tiny", "fast", "tutorial"])
    p.add_argument("--difficulty", default="medium", choices=["easy", "medium", "hard"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, help="output file (.mat)")

    add("defaults", cmd_defaults, "print the settings schema, presets and defaults (for MATLAB)")

    p = add("check", cmd_check, "check a run's settings against its dataset without training")
    p.add_argument("--config", required=True, help="run settings (JSON)")
    p.add_argument("--timing", action="store_true", help="also time a few training steps to estimate run time")

    p = add("train", cmd_train, "train and evaluate a model")
    p.add_argument("--config", required=True, help="run settings (JSON)")
    p.add_argument("--run-dir", default=None, help="run folder to write (default: a new folder in output.root)")
    p.add_argument("--fold", type=int, default=None, help="only this cross-validation fold (1-based)")

    p = add("summarize", cmd_summarize, "describe a run folder")
    p.add_argument("run", help="run folder")
    return parser


def main(argv: list[str] | None = None) -> int:
    import warnings

    # numpy 2.x with Apple's Accelerate BLAS reports spurious floating-point warnings in matmul.
    warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help(sys.stderr)
        return EXIT_USER
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    try:
        return int(args.func(args))
    except PrismtError as err:
        if args.json:
            _emit(args, {"ok": False, "error": err.to_dict()})
        log.error("%s", err.to_dict()["title"])
        log.error("%s", err)
        return err.exit_code
    except KeyboardInterrupt:
        return 3
    except Exception as exc:  # noqa: BLE001 - last resort: never die without a message
        detail = traceback.format_exc()
        log.error("Unexpected error: %s\n%s", exc, detail)
        if args.json:
            _emit(args, {"ok": False, "error": {"code": "E_INTERNAL", "title": "PRISMT hit an unexpected error",
                                                 "message": str(exc), "hint": "Please report this with the log.",
                                                 "field": "", "traceback": detail}})
        return EXIT_INTERNAL
