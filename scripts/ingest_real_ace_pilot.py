"""
Turnkey CLI for Ingesting Raw Kayser ACE / MAT Pilot Reactor Logs into Auto-Catalyst 2.0.
Usage:
  python scripts/ingest_real_ace_pilot.py [path_to_csv]
Default:
  data/benchmarks/kayser_ace_pilot_run_sample.csv
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))
sys.path.append(str(PROJECT_ROOT / "src" / "analysis"))

from ace_pilot_ingestor import ACEPilotIngestor


def main():
    target_csv = Path(sys.argv[1]) if len(sys.argv) > 1 else (PROJECT_ROOT / "data" / "benchmarks" / "kayser_ace_pilot_run_sample.csv")

    if not target_csv.exists():
        print(f"[Error] Target pilot log file not found: {target_csv}")
        sys.exit(1)

    print("=" * 80)
    print("  KAYSER ACE / MAT PILOT REACTOR DATA INGESTION UTILITY")
    print("=" * 80)
    print(f"Reading target file: {target_csv.name}")

    ingestor = ACEPilotIngestor()
    records = ingestor.ingest_csv(target_csv)
    print(f"Validated {len(records)} runs conforming to ASTM mass balance closure standards.")

    committed = ingestor.commit_to_active_learning(records)
    print(f"Committed {committed} new high-fidelity (s=1.0) ground-truth runs to active learning storage.")
    print(f"Active learning history location: {ingestor.history_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
