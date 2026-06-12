"""SzCORE entrypoint: python -m mvafa_szcore <input.edf> <output.tsv>

This is the file SzCORE calls inside the Docker container:
    CMD python3 -m mvafa_szcore "/data/${INPUT}" "/output/${OUTPUT}"
"""
import argparse
import datetime
import sys
from pathlib import Path

import pyedflib

from .inference import run_inference, probs_to_events


def _write_tsv(out_path: str, events: list, edf_path: str) -> None:
    """Write SzCORE-compatible BIDS annotation TSV."""
    with pyedflib.EdfReader(edf_path) as edf:
        date_time = edf.getStartdatetime()
        duration = float(edf.getFileDuration())

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    rows = []
    if events:
        for onset, offset in events:
            rows.append({
                "onset": onset,
                "duration": offset - onset,
                "eventType": "sz",
                "confidence": "n/a",
                "channels": "n/a",
                "dateTime": date_time.isoformat() if hasattr(date_time, "isoformat") else str(date_time),
                "recordingDuration": duration,
            })
    else:
        # SzCORE requires at least one row; use bckg to indicate no seizures
        rows.append({
            "onset": 0.0,
            "duration": duration,
            "eventType": "bckg",
            "confidence": "n/a",
            "channels": "n/a",
            "dateTime": date_time.isoformat() if hasattr(date_time, "isoformat") else str(date_time),
            "recordingDuration": duration,
        })

    header = "onset\tduration\teventType\tconfidence\tchannels\tdateTime\trecordingDuration\n"
    with open(out_path, "w") as f:
        f.write(header)
        for r in rows:
            f.write(
                f"{r['onset']}\t{r['duration']}\t{r['eventType']}\t"
                f"{r['confidence']}\t{r['channels']}\t{r['dateTime']}\t"
                f"{r['recordingDuration']}\n"
            )
    print(f"[INFO] Wrote {len(rows)} annotation(s) → {out_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="MV-AFA seizure detection — SzCORE submission"
    )
    parser.add_argument("input",  help="Input EDF file path")
    parser.add_argument("output", help="Output TSV annotation file path")
    args = parser.parse_args()

    print(f"[INFO] Running MV-AFA on: {args.input}", flush=True)
    probs, sfreq, window_sec, window_starts_sec = run_inference(args.input)

    if len(probs) == 0:
        print("[WARN] No windows extracted — writing bckg annotation", flush=True)
        events = []
    else:
        events = probs_to_events(probs, window_starts_sec, window_sec)
        print(f"[INFO] Detected {len(events)} seizure event(s)", flush=True)

    _write_tsv(args.output, events, args.input)


if __name__ == "__main__":
    main()
