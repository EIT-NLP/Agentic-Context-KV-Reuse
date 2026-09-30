"""Evaluation-only CLI for the vendored SRA-Bench subset."""
import argparse
from sragents.cli import evaluate


def main(argv=None):
    parser = argparse.ArgumentParser(prog="sragents", description="SRA-Bench evaluation")
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate.add_parser(subparsers)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
