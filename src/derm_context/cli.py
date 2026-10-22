"""Command line entry points; run from the repository root."""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Patient-context dermatology research pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    downloader = commands.add_parser("download", help="Download official full SLICE-3D data")
    downloader.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    preparer = commands.add_parser("prepare", help="Validate data and freeze patient splits")
    preparer.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    preparer.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    preparer.add_argument("--reports-dir", type=Path, default=Path("reports"))
    preparer.add_argument("--seed", type=int, default=42)
    trainer = commands.add_parser(
        "train", help="Train XGBoost; report validation, leave test sealed"
    )
    trainer.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    trainer.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    trainer.add_argument("--reports-dir", type=Path, default=Path("reports"))
    trainer.add_argument("--config", type=Path, default=Path("configs/baseline.json"))
    trainer.add_argument("--run-name", default="baseline-v1")
    predictor = commands.add_parser("predict", help="Score metadata using a locally trained model")
    predictor.add_argument("--input", type=Path, required=True)
    predictor.add_argument("--model-dir", type=Path, required=True)
    predictor.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "download":
        from derm_context.download import download

        download(args.raw_dir)
    elif args.command == "prepare":
        from derm_context.prepare import prepare

        prepare(args.raw_dir, args.processed_dir, args.reports_dir, args.seed)
    elif args.command == "train":
        from derm_context.train import train

        train(args.processed_dir, args.artifact_dir, args.reports_dir, args.config, args.run_name)
    elif args.command == "predict":
        from derm_context.train import predict

        predict(args.input, args.model_dir, args.output)


if __name__ == "__main__":
    main()
