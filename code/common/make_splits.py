#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""identity+Video double isolation train/val/test split(3.2.8 design).

rule:
  1. by identity For nodes, "with sample "Co-occurrence" maps edges, The connected components are all assigned to the same split
     (same character/All samples of the same video never span split → put an end to ID/background leak)
  2. test priority: Press first 13class interaction × C0-C3 × 2/3people Tiered quota filled test(~2500),
     and cc0 license_tier Priority in test core subset
  3. The remaining connected components are divided proportionally train / val

usage: python make_splits.py --db ~/mpie_data/manifests/mpie.db \
        --test-size 2500 --val-size 500 [--dry-run]
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.manifest import connect, rows  # noqa: E402


def union_find_components(samples):
    """identity Connected components of co-occurrence graphs: return {component_id: [sample,...]}。"""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    for s in samples:
        ids = json.loads(s["identity_ids"] or "[]") + [f"vid::{s['video_id']}"]
        for x in ids[1:]:
            union(ids[0], x)
        s["_root_key"] = ids[0] if ids else s["sample_id"]
    comps = defaultdict(list)
    for s in samples:
        comps[find(s["_root_key"])].append(s)
    return comps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--test-size", type=int, default=2500)
    ap.add_argument("--val-size", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    conn = connect(args.db)
    samples = rows(conn, "SELECT * FROM samples WHERE qc_pass=1")
    if not samples:
        print("samples Table is empty or None qc_pass=1, Run first Stage 9/10")
        return
    comps = list(union_find_components(samples).values())
    rng.shuffle(comps)
    print(f"{len(samples)} samples in {len(comps)} identity-connected components")

    # Tiered quotas: (interaction_type, density_level, n_person)
    def stratum(s):
        return (s["interaction_type"], s["density_level"], min(s["n_person"] or 2, 3))

    target_per_stratum = defaultdict(int)
    for s in samples:
        target_per_stratum[stratum(s)] += 1
    total = len(samples)
    quota = {k: max(1, round(v / total * args.test_size)) for k, v in target_per_stratum.items()}

    test_fill = defaultdict(int)
    assign = {}
    # cc0 Portion priority test(Core subset pixels publishable)
    comps.sort(key=lambda c: -sum(1 for s in c if s["license_tier"] == "cc0"))
    for comp in comps:
        n_test = sum(sum(v for v in test_fill.values()) for _ in [0])
        if sum(test_fill.values()) < args.test_size:
            gain = sum(1 for s in comp if test_fill[stratum(s)] < quota[stratum(s)])
            if gain >= len(comp) * 0.5:      # Only if more than half of the samples can fill the quota will they be admitted test
                for s in comp:
                    assign[s["sample_id"]] = "test"
                    test_fill[stratum(s)] += 1
                continue
        assign.update({s["sample_id"]: None for s in comp})

    rest = [s for s in samples if assign.get(s["sample_id"]) is None]
    rng.shuffle(rest)
    # val Also divided as a whole according to the weight: Re-gather by weight rest
    rest_comps = list(union_find_components(rest).values())
    n_val = 0
    for comp in rest_comps:
        split = "val" if n_val < args.val_size else "train"
        for s in comp:
            assign[s["sample_id"]] = split
        if split == "val":
            n_val += len(comp)

    counts = defaultdict(int)
    for v in assign.values():
        counts[v] += 1
    print("split sizes:", dict(counts))
    if args.dry_run:
        return
    for sid, sp in assign.items():
        conn.execute("UPDATE samples SET split=? WHERE sample_id=?", (sp, sid))
    conn.commit()
    # Leak self-check: any identity Should only appear in a split
    leak = rows(conn, """SELECT identity_ids FROM samples GROUP BY identity_ids
                         HAVING COUNT(DISTINCT split) > 1 LIMIT 5""")
    print("Leak self-check(Should be empty):", leak)


if __name__ == "__main__":
    main()
