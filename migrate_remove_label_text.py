"""
migrate_remove_label_text.py
────────────────────────────
One-time migration: removes the 'label_text' column from database.parquet.

Usage:
    python migrate_remove_label_text.py                        # uses database.parquet in place
    python migrate_remove_label_text.py path/to/database.parquet

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
    if "label_text" not in df.columns:
        print("'label_text' column not found — already migrated or wrong file. Nothing to do.")
        sys.exit(0)

    # ── Backup ───────────────────────────────────────────────────────────────
    backup_path = path + ".bak"
    shutil.copy2(path, backup_path)
    print(f"  Backup written to {backup_path}")

    # ── Migrate ──────────────────────────────────────────────────────────────
    df = df.drop(columns=["label_text"])
    print(f"  Dropped 'label_text' column.")
    print(f"  Columns after migration: {list(df.columns)}")

    # ── Save ─────────────────────────────────────────────────────────────────
    df.to_parquet(path, index=False)
    print(f"  Saved migrated parquet to {path}")
    print(f"Done. Original backed up at {backup_path} — delete it once verified.")


if __name__ == "__main__":
    db_path = sys.argv[1] if len(sys.argv) > 1 else "database.parquet"
    migrate(db_path)
