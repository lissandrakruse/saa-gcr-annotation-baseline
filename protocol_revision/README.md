# Global temporal matching protocol

Code-only local revision of 1 November 2026. This is not Hitaeshi's upstream v0.2 release or a group-approved standard.

## Run

Use the repository's Python 3.12 environment and pinned dependencies. CSV inputs must contain `event_id,start_utc,end_utc`, unique IDs and positive-duration timezone-aware intervals. Inputs are supplied locally and are not included or downloaded automatically.

```sh
python -m unittest discover -s protocol_revision/tests -v
python protocol_revision/src/run_protocol.py \
  --reference-events /private/reference_events.csv \
  --prediction-events /private/srag_events.csv \
  --out /private/revision_results
```

The optional `--raw /private/SPACEDOS_ISS2020_slice.csv` mode is deliberately restricted to the original source SHA-256 and the fixed 15 October–7 November 2020 UTC comparison window. It checks re-extracted intervals against the supplied prediction inputs; it is not a generic raw-data detector. Omitting it leaves raw verification explicitly false. Avoid concurrent runs using the same output directory.

## Contract

- Expand references by 60 seconds on each side; inclusive intersection/contact establishes eligibility.
- Maximize matched-pair count first, then total **expanded** overlap, with exact integer costs. Deterministic residual-path ordering resolves ties; a unique optimal matching is not guaranteed for arbitrary inputs. Confirm expanded versus original overlap with collaborators before treating this as an agreed protocol.
- Export signed boundary offsets from original intervals separately. Offset sign is prediction minus reference. No requirement that both boundaries be within 60 seconds is imposed by matching.
- Acquisition gaps >30 seconds split positive runs, exactly 30 seconds does not; midnight does not split. The endpoint is capped by the next negative observation, last positive + median cadence, and window end. A terminal positive sample is capped by cadence/window end. These computational conventions do not establish physical boundary precision.
- The new entry point is separate from the existing supported-IoU pipeline. Existing commands and retrospective model settings are unchanged.

## Verification and distribution

Tests use synthetic inputs only, including an exhaustive oracle for 100 small random matching graphs, a greedy counterexample, overlap priority, contact, gap thresholds, midnight and terminal samples. The reviewer package separately contains private inputs, outputs and English interpretation. This public directory contains no annotations, measured results, raw files, private source links or executed data-bearing notebooks. The existing repository MIT license covers this code, not data supplied separately.

Agreement metrics are not physical accuracy, and unmatched events are not verified physical false positives. This revision does not train ML models, create human annotations or evaluate an independent detector.
