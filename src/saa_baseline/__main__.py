"""Command line: python -m saa_baseline --help."""
import argparse
import json
from pathlib import Path
import pandas as pd
from .analysis import evaluate_predictions, reproduce, write_event_tables
from .data import (ANALYSIS_END, CUT_TEST, analysis_data, load_annotations, load_raw,
                   normalize_annotations, partition_masks, sha256)
from .model import load_model, predict, save_model, train


def main():
    parser = argparse.ArgumentParser(description="Frozen retrospective annotation-reproduction baseline. Not a validated physical detector.")
    commands = parser.add_subparsers(dest="command", required=True)
    normal = commands.add_parser("normalize", help="Create private annotator=Martin copy and per-row provenance")
    normal.add_argument("--annotations", required=True)
    normal.add_argument("--output", required=True)
    normal.add_argument("--provenance", required=True)
    for name in ("train", "reproduce"):
        command = commands.add_parser(name)
        command.add_argument("--raw", required=True)
        command.add_argument("--annotations", required=True, help="Original hash-locked annotations, not normalized copy")
        command.add_argument("--output", required=True)
        if name == "reproduce":
            command.add_argument("--legacy-dir")
    pred = commands.add_parser("predict", help="Predict from channels9–27 only; labels/SRAG ignored")
    pred.add_argument("--model", required=True, help="Trusted local joblib artifact only")
    pred.add_argument("--raw", required=True)
    pred.add_argument("--output", required=True)
    pred.add_argument("--start", default=str(CUT_TEST))
    pred.add_argument("--end", default=str(ANALYSIS_END))
    ev = commands.add_parser("evaluate", help="Descriptive agreement; does not certify an independent test")
    ev.add_argument("--predictions", required=True)
    ev.add_argument("--annotations", required=True)
    ev.add_argument("--cadence-seconds", required=True, type=float)
    ev.add_argument("--end", required=True, help="Exclusive UTC evaluation window end")
    ev.add_argument("--output", required=True)
    ev.add_argument("--scope", choices=("retrospective", "candidate_external_not_certified"), default="retrospective")
    args = parser.parse_args()
    if args.command == "normalize":
        result = normalize_annotations(args.annotations, args.output, args.provenance)
        print(json.dumps({k: v for k, v in result.items() if k != "row_provenance"}, indent=2))
    elif args.command == "reproduce":
        result = reproduce(args.raw, args.annotations, args.output, args.legacy_dir)
        print(json.dumps(result, indent=2))
    elif args.command == "train":
        frame, _, human, cadence, hashes = analysis_data(args.raw, args.annotations)
        mask = partition_masks(frame, human)["train"]
        estimator = train(frame.loc[mask], frame.marked.to_numpy()[mask].astype(int))
        save_model(estimator, args.output, {"source_hashes": hashes, "cadence_seconds": cadence,
                   "train_end_exclusive": "2020-10-31T00:00:00Z", "independent_final_test_used": False})
        print(f"Saved frozen model to {args.output}; no validation/test labels used for fitting.")
    elif args.command == "predict":
        estimator, _ = load_model(args.model)
        frame = load_raw(args.raw, validate_srag=False)
        start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
        if start.tzinfo is None or end.tzinfo is None or end <= start:
            parser.error("start/end must be timezone-aware with start < end")
        frame = frame[(frame.UTC >= start) & (frame.UTC < end)]
        if frame.empty:
            parser.error("Prediction window contains no observations")
        probability, prediction = predict(estimator, frame)
        result = frame[["UTC"]].copy()
        result["model_probability"] = probability
        result["model_prediction"] = prediction.astype(int)
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(path, index=False)
        path.with_suffix(path.suffix + ".json").write_text(json.dumps({"raw_sha256": sha256(args.raw),
            "model_sha256": sha256(args.model), "start": str(start), "end_exclusive": str(end),
            "independence_certified": False}, indent=2) + "\n")
        print(f"Wrote {len(result)} predictions to {path}.")
    elif args.command == "evaluate":
        p = pd.read_csv(args.predictions)
        m = load_annotations(args.annotations)
        end = pd.Timestamp(args.end)
        if end.tzinfo is None:
            parser.error("end must include a timezone")
        sample, events, runs = evaluate_predictions(p, m, args.cadence_seconds, end)
        path = Path(args.output)
        path.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(events).to_csv(path / "supported_event_overlap.csv", index=False)
        write_event_tables(runs, path, "model")
        result = {"scope": args.scope, "independent_final_test_used": False,
                  "independence_certified": False, "sample_agreement": sample,
                  "prediction_sha256": sha256(args.predictions), "annotation_sha256": sha256(args.annotations),
                  "warning": "Agreement with incomplete marks, not physical accuracy. External status requires separate evaluator provenance; this command cannot establish it."}
        (path / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
