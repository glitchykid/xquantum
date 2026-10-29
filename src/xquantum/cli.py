"""Commands for reproducible research runs; no patient data sent to remote services."""

import argparse
import json
import sys
from pathlib import Path

from xquantum.dataset import audit_manifest
from xquantum.ingest import fetch_bootstrap
from xquantum.model import ResearchModel, train
from xquantum.schemas import ClinicalRecord, Sample


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="XQuantum hair/scalp research, not clinical diagnosis"
    )
    commands = root.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch-bootstrap", help="Download 10 cataloged open-license images")
    fetch.add_argument("--output", type=Path, default=Path("data/commons"))
    for name in ("audit", "train"):
        command = commands.add_parser(name)
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--bootstrap", action="store_true")
        if name == "train":
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--multimodal", action="store_true")
            command.add_argument("--epochs", type=int, default=50)
            command.add_argument("--seed", type=int, default=42)
    analyze = commands.add_parser("analyze")
    analyze.add_argument("--model", type=Path, required=True)
    analyze.add_argument("--image", type=Path, required=True)
    analyze.add_argument("--clinical", type=Path)
    serve = commands.add_parser("serve", help="Bind a research-only API to localhost")
    serve.add_argument("--model", type=Path, required=True)
    serve.add_argument("--port", type=int, default=8000)
    schema = commands.add_parser("schema")
    schema.add_argument("--kind", choices=("clinical", "sample"), default="clinical")
    return root


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "fetch-bootstrap":
            result = fetch_bootstrap(args.output)
        elif args.command == "audit":
            result = {
                "samples": len(audit_manifest(args.manifest, bootstrap=args.bootstrap)),
                "audit": "passed",
            }
        elif args.command == "train":
            report = train(
                args.manifest,
                args.output,
                bootstrap=args.bootstrap,
                multimodal=args.multimodal,
                epochs=args.epochs,
                seed=args.seed,
            )
            result = {
                key: report[key]
                for key in (
                    "mode",
                    "training_samples",
                    "classes",
                    "held_out_metrics",
                    "clinically_validated",
                )
            }
        elif args.command == "analyze":
            clinical = (
                ClinicalRecord.model_validate_json(args.clinical.read_text())
                if args.clinical
                else ClinicalRecord()
            )
            result = ResearchModel.load(args.model).analyze(args.image.read_bytes(), clinical)
        elif args.command == "schema":
            result = (ClinicalRecord if args.kind == "clinical" else Sample).model_json_schema()
        else:
            import uvicorn

            from xquantum.api import create_app

            uvicorn.run(create_app(args.model), host="127.0.0.1", port=args.port, access_log=False)
            return 0
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError) as exc:
        print(f"XQuantum: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
