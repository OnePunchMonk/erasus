"""
Erasus CLI — Entry point for command-line usage.

Usage::

    erasus unlearn --config config.yaml
    erasus evaluate --checkpoint model.pt --metrics accuracy mia
    erasus --help

Or as a module::

    python -m erasus.cli.main unlearn --config config.yaml
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="erasus",
        description=(
            "ERASUS — Efficient Representative And Surgical Unlearning Selection\n"
            "Universal Machine Unlearning via Coreset Selection"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(
        title="commands",
        dest="command",
        help="Available commands",
    )

    # Register sub-commands
    from erasus.cli.unlearn import add_parser as add_unlearn
    from erasus.cli.evaluate import add_parser as add_evaluate

    add_unlearn(subparsers)
    add_evaluate(subparsers)

    # Legacy --config support (redirect to unlearn)
    parser.add_argument(
        "--config", "-c", type=str, default=None,
        help="(Legacy) YAML config — equivalent to 'erasus unlearn --config'.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="(Legacy) Validate only — equivalent to 'erasus unlearn --dry-run'.",
    )

    args = parser.parse_args()

    # Handle legacy --config flag
    if args.command is None and args.config:
        # Redirect to unlearn sub-command
        sys.argv = ["erasus", "unlearn", "--config", args.config]
        if args.dry_run:
            sys.argv.append("--dry-run")
        args = parser.parse_args(sys.argv[1:])

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Dispatch to the selected sub-command
    args.func(args)


if __name__ == "__main__":
    main()
