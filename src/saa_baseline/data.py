"""Validated private input loading and lossless-field annotator normalization."""
import csv
import hashlib
import io
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .intervals import merge_intervals, rasterize

CHANNELS = tuple(str(i) for i in range(9, 28))
EXPECTED_HASHES = {
    "raw": "351af6c6734b6e70bc835b11156c2d0b087f64f71cbcc8d72c0ca7dddaed8f1f",
    "marks": "d22a3fb51c50c2b2a77f2c9046738d17a8ef6df65d494944d74d37cebac47e17",
}
CUT_TRAIN = pd.Timestamp("2020-10-31", tz="UTC")
CUT_TEST = pd.Timestamp("2020-11-04", tz="UTC")
ANALYSIS_START = pd.Timestamp("2020-10-15", tz="UTC")
ANALYSIS_END = pd.Timestamp("2020-11-08", tz="UTC")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def normalize_annotations(source, destination, sidecar):
    """Only the annotator field changes; preserve every other byte in this schema.

    This intentionally fails closed for a different CSV representation instead
    of silently rewriting timestamps, quoting, row order or other columns.
    """
    source, destination, sidecar = map(Path, (source, destination, sidecar))
    if len({p.resolve() for p in (source, destination, sidecar)}) != 3:
        raise ValueError("Source, normalized copy and sidecar must be distinct.")
    original = source.read_bytes()
    rows = list(csv.reader(io.StringIO(original.decode("utf-8"), newline="")))
    if not rows or rows[0] != ["annotator", "date", "start", "end", "label", "source"]:
        raise ValueError("Unexpected annotation schema.")
    lines = original.splitlines(keepends=True)
    if len(lines) != len(rows):
        raise ValueError("Multiline CSV records are unsupported for byte-preserving normalization.")
    result = [lines[0]]
    provenance = []
    for i, (row, line) in enumerate(zip(rows[1:], lines[1:]), start=1):
        if len(row) != len(rows[0]) or row[0] not in {"Martin", "Martin Kakona"}:
            raise ValueError(f"Unexpected annotator or schema at data row {i}.")
        prefix = (row[0] + ",").encode("utf-8")
        if not line.startswith(prefix):
            raise ValueError("Quoted annotator representation requires explicit review.")
        result.append(b"Martin," + line[len(prefix):])
        provenance.append({"data_row_1_based": i, "original_annotator": row[0]})
    normalized = b"".join(result)
    check = list(csv.reader(io.StringIO(normalized.decode("utf-8"), newline="")))
    if check[0] != rows[0] or len(check) != len(rows):
        raise AssertionError("Normalization changed schema or row count.")
    if any(a[1:] != b[1:] or b[0] != "Martin" for a, b in zip(rows[1:], check[1:])):
        raise AssertionError("Normalization changed a non-annotator value.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(normalized)
    manifest = {"operation": "Canonical annotator=Martin; no new annotation or human review",
                "source_filename": source.name, "normalized_filename": destination.name,
                "source_sha256": hashlib.sha256(original).hexdigest(),
                "normalized_sha256": hashlib.sha256(normalized).hexdigest(),
                "rows": len(provenance), "columns_preserved": rows[0],
                "non_annotator_bytes_preserved": True,
                "row_provenance": provenance}
    sidecar.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_raw(path, validate_srag=True):
    with open(path, newline="") as source:
        header = next(csv.reader(source))
    if len(set(header)) != len(header):
        raise ValueError("Duplicate input column names are not allowed.")
    frame = pd.read_csv(path)
    if "UTC" not in frame:
        raise ValueError("Missing UTC column.")
    frame["UTC"] = pd.to_datetime(frame.UTC, utc=True, format="ISO8601")
    if frame.UTC.isna().any() or frame.UTC.duplicated().any():
        raise ValueError("Missing or duplicate timestamps.")
    frame = frame.sort_values("UTC").reset_index(drop=True)
    features(frame)  # Validate allowed detector inputs, not labels.
    if validate_srag and "SAA" in frame and not frame.SAA.isin([0, 1]).all():
        raise ValueError("Supplied SAA comparison mask must be binary.")
    return frame


def load_annotations(path):
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    for column in ("annotator", "start", "end", "label"):
        if column not in frame:
            raise ValueError(f"Missing annotation column {column}")
    if not set(frame.annotator) <= {"Martin", "Martin Kakona"}:
        raise ValueError("Unexpected annotator; normalization needs review.")
    for col in ("start", "end"):
        frame[col] = pd.to_datetime(frame[col], utc=True, format="ISO8601")
    if not (frame.end > frame.start).all() or not frame.label.eq("SAA").all():
        raise ValueError("Annotations must be finite positive-duration SAA intervals.")
    return frame


def features(frame):
    if frame.columns.has_duplicates:
        raise ValueError("Duplicate feature column names are not allowed.")
    missing = set(CHANNELS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing detector channels: {sorted(missing)}")
    x = frame.loc[:, list(CHANNELS)].to_numpy(dtype=float)
    if not np.isfinite(x).all() or (x < 0).any():
        raise ValueError("Detector channels must be finite and nonnegative.")
    return np.log1p(x)


def analysis_data(raw_path, marks_path, verify_originals=True):
    hashes = {"raw": sha256(raw_path), "marks": sha256(marks_path)}
    if verify_originals and hashes != EXPECTED_HASHES:
        raise ValueError("Source version differs from frozen 21-Oct inputs; do not compare silently.")
    raw = load_raw(raw_path)
    marks = load_annotations(marks_path)
    human = merge_intervals(zip(marks.start, marks.end))
    start, end = ANALYSIS_START, ANALYSIS_END
    if (marks.start < start).any() or (marks.end > end).any():
        raise ValueError("Marks outside the frozen analysis window.")
    frame = raw[(raw.UTC >= start) & (raw.UTC < end)].copy().reset_index(drop=True)
    if len(frame) < 2:
        raise ValueError("Too few observations.")
    cadence = float(frame.UTC.diff().dt.total_seconds().median())
    frame["marked"] = rasterize(frame.UTC, human)
    frame["segment_id"] = frame.UTC.diff().dt.total_seconds().gt(30.).cumsum()
    return frame, marks, human, cadence, hashes


def partition_masks(frame, human):
    if any(s < c < e for s, e in human for c in (CUT_TRAIN, CUT_TEST)):
        raise ValueError("A human event crosses a frozen split boundary.")
    train = (frame.UTC < CUT_TRAIN).to_numpy()
    validation = ((frame.UTC >= CUT_TRAIN) & (frame.UTC < CUT_TEST)).to_numpy()
    retrospective = (frame.UTC >= CUT_TEST).to_numpy()
    if not train.any() or not validation.any() or not retrospective.any():
        raise ValueError("All frozen temporal partitions must be nonempty.")
    if not np.all(train.astype(int) + validation + retrospective == 1):
        raise AssertionError("Partitions overlap or omit observations.")
    return {"train": train, "validation_not_retuned": validation, "retrospective_holdout": retrospective}
