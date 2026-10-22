"""Build a comparison table from aggregate reports on an identical validation split."""

import argparse
from pathlib import Path

from derm_context.train_embeddings import comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_names", nargs="+")
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    for name in args.run_names:
        if Path(name).name != name or name in {"", ".", ".."}:
            parser.error("run names must be single filenames without directory components")
    comparison(args.reports_dir, args.run_names)


if __name__ == "__main__":
    main()
