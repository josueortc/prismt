"""Command-line interface: ``python -m prismt <command>``."""

from __future__ import annotations

import argparse
import sys

from prismt import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="prismt", description=__doc__)
    parser.add_argument("--version", action="version", version=f"prismt {__version__}")
    parser.add_subparsers(dest="command", metavar="<command>")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help(sys.stderr)
        return 2
    return 0
