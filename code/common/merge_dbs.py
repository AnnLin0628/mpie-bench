#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Merge multiple shards manifest sqlite → a general library(Use it after running multiple cards in parallel)。

Each shard DB primary key(kf_id/track_id/ref_id/video_id...)All included video_id prefix, Globally unique,
so INSERT OR IGNORE Just stack it directly, No conflicts.crop/frame use absolute path, Don't move files.

usage: python merge_dbs.py --out ~/mpie_data/manifests/mpie_cc0.db \
        ~/mpie_data/manifests/mpie_cc0_0.db ... _3.db
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.manifest import connect  # noqa: E402

TABLES = ["videos", "shots", "keyframes", "persons", "refs", "captions", "samples"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("shards", nargs="+")
    args = ap.parse_args()
    conn = connect(args.out)                 # Create table
    for t in TABLES:                         # Clear output table, Ensure each merge is a clean rebuild(Avoid accumulation of old residues)
        conn.execute(f"DELETE FROM {t}")
    conn.commit()
    total = {t: 0 for t in TABLES}
    for sh in args.shards:
        if not Path(sh).exists():
            print(f"skip missing {sh}"); continue
        conn.execute("ATTACH DATABASE ? AS s", (sh,))
        for t in TABLES:
            cur = conn.execute(f"INSERT OR IGNORE INTO {t} SELECT * FROM s.{t}")
            total[t] += cur.rowcount
        conn.commit()
        conn.execute("DETACH DATABASE s")
        print(f"merged {sh}")
    print("The combined total:")
    for t in TABLES:
        n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"  {t:10s} {n}")


if __name__ == "__main__":
    main()
