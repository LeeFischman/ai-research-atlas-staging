"""
migrate_text_column.py
──────────────────────
One-time migration: renames the 'text' column to 'text_vector' in database.parquet
and moves it to the last position.

Usage:
    python migrate_text_column.py                        # uses database.parquet in place
    python migrate_text_column.py path/to/database.parquet

The original file is backed up to database.parquet.bak before any changes are made.
"""

import sys
import os
import shutil
import pandas as pd


def migrate(path: str) -> None:
    if not os.path.exists(path):
        print(f"ERROR: File not found: {path}")
        sys.exit(1)

    # ── Load ─────────────────────────────────────────────────────────────────
    print(f"Loading {path}...")
    df = pd.read_parquet(path)
    print(f"  {len(df)} rows, {len(df.columns)} columns.")
    print(f"  Columns: {list(df.columns)}")

    # ── Validate ─────────────────────────────────────────────────────────────
    if "text" not in df.columns:
        print("ERROR: 'text' column not found — already migrated or wrong file.")
        sys.exit(1)

    if "text_vector" in df.columns:
        print("ERROR: 'text_vector' column already exists — refusing to overwrite.")
        sys.exit(1)

    # ── Backup ───────────────────────────────────────────────────────────────
    backup_path = path + ".bak"
    shutil.copy2(path, backup_path)
    print(f"  Backup written to {backup_path}")

    # ── Migrate ──────────────────────────────────────────────────────────────
    # 1. Drop 'text' from current position
    # 2. Append as 'text_vector' at the end
    text_data = df["text"]
    df = df.drop(columns=["text"])
    df["text_vector"] = text_data

    print(f"  Renamed 'text' → 'text_vector' (now last column).")
    print(f"  Columns after migration: {list(df.columns)}")

    # ── Save ─────────────────────────────────────────────────────────────────
    df.to_parquet(path, index=False)
    print(f"  Saved migrated parquet to {path}")
    print(f"Done. Original backed up at {backup_path} — delete it once verified.")


if __name__ == "__main__":
    db_path = sys.argv[1] if len(sys.argv) > 1 else "database.parquet"
    migrate(db_path)
