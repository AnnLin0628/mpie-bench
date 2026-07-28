#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 10: assemble train/eval samples from QC-passed keyframes → sample triplets.

Each sample = best ref per identity (lower tier better; tie-break diversity_score),
not from target frame) + edit instruction + target frame.
license_tier from video (cc0 → public benchmark; restricted → train/calibration only).
split empty; common/make_splits.py does identity+video isolated splits.

Human override: if human_bindings table exists (scene board export via apply_scene_map.py
), use manual identity/refs; auto pick prefers is_primary=1
(starred clean refs) over tier/diversity. Unchanged when no human data (e.g. Harmony4D GT).

Usage: python build_samples.py --db ~/mpie_data/manifests/mpie.db
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.manifest import connect, upsert, rows  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--max-tier", type=int, default=4)
    args = ap.parse_args()
    conn = connect(args.db)

    has_hb = bool(rows(conn, "SELECT name FROM sqlite_master WHERE type='table' AND name='human_bindings'"))
    has_primary = has_hb and any(
        r["name"] == "is_primary" for r in rows(conn, "SELECT name FROM pragma_table_info('refs')"))
    primary_order = "is_primary DESC, " if has_primary else ""

    kfs = rows(conn, """
        SELECT k.*, c.edit_instruction, c.interaction_type,
               c.contact_density_level, c.n_person AS cap_n
        FROM keyframes k JOIN captions c ON c.kf_id = k.kf_id
        WHERE k.selected=1 AND c.needs_review=0""")
    n_full = n_partial = n_human = 0
    for kf in kfs:
        hb = rows(conn, "SELECT identity_ids FROM human_bindings WHERE kf_id=?",
                  (kf["kf_id"],)) if has_hb else []
        if hb:                                   # human binding from scene board
            import json as _json
            idents = _json.loads(hb[0]["identity_ids"])
            n_human += 1
        else:
            idents = [r["identity_id"] for r in rows(conn, """
                SELECT DISTINCT identity_id FROM persons
                WHERE track_id LIKE ? AND identity_id IS NOT NULL""",
                (kf["kf_id"] + "_p%",))]
        ref_ids = []
        for ident in idents:                     # one best ref per identity; ★ is_primary first
            best = rows(conn, f"""
                SELECT ref_id FROM refs
                WHERE identity_id=? AND kf_id != ? AND tier <= ?
                ORDER BY {primary_order}tier ASC, diversity_score DESC LIMIT 1""",
                (ident, kf["kf_id"], args.max_tier))
            if best:
                ref_ids.append(best[0]["ref_id"])
        # all identities have refs → qc_pass=1; partial → qc_pass=0 (train only)
        full = len(ref_ids) == len(idents) and len(ref_ids) >= 2
        if not (len(ref_ids) >= 2):
            continue
        vid = rows(conn, "SELECT license_tier FROM videos WHERE video_id=?",
                   (kf["video_id"],))
        upsert(conn, "samples", {
            "sample_id": f"s_{kf['kf_id']}", "kf_id": kf["kf_id"],
            "video_id": kf["video_id"], "identity_ids": idents,
            "ref_ids": ref_ids, "instruction": kf["edit_instruction"],
            "target_path": kf["frame_path"],
            "density_level": kf["contact_density_level"],
            "interaction_type": kf["interaction_type"],
            "n_person": kf["cap_n"],
            "license_tier": vid[0]["license_tier"] if vid else "restricted",
            "qc_pass": int(full), "split": None,
        })
        n_full += int(full)
        n_partial += int(not full)
    conn.commit()
    by = rows(conn, """SELECT interaction_type, density_level, COUNT(*) c
                       FROM samples WHERE qc_pass=1
                       GROUP BY 1,2 ORDER BY c DESC""")
    print(f"samples: {n_full} full(qc_pass=1) + {n_partial} partial-ref (human {n_human})")
    print("Stratified cell counts (qc_pass=1):")
    for r in by:
        print(f"  {r['interaction_type'] or '?':22s} {r['density_level'] or '?':4s} {r['c']}")


if __name__ == "__main__":
    main()
