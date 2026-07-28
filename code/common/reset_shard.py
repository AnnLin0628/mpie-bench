#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Safely reset a shard Stage4→6 output, For repeated parameter adjustment and re-run(Alternative to manual work UPDATE, Avoid path pollution)。

background: crop_targets.py The target frame will be frame_path Overlay into clipping path(Designed like this, convenient
Stage10 Take it directly frame_path when target_path). but simply put selected reset back 0 It won't
recover frame_path —— it still points to the crop, Once the crop directory is deleted, Follow-up Stage4 Re-scan will read
dead path(imread fail)。kf_id Encoded in naming rules video_id + frame_idx, Therefore, we can accurately infer
original Stage2 frame path to fix; reference pool(_r prefix)The rows themselves are candidates for regeneration on each rerun.,
Direct deletion is the cleanest method, No need to "restore", No reservations required.

usage: python reset_shard.py --db <db> --frames <frames_dir>
"""
import argparse
import sqlite3
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--frames", required=True, help="Stage2 Frame directory(Contains <vid>/<frame>.jpg)")
    args = ap.parse_args()
    conn = sqlite3.connect(args.db)

    # 1) peak frame(_f): recover frame_path for Stage2 original path + selected Return to zero
    peak_rows = conn.execute(
        "SELECT kf_id, video_id, frame_idx FROM keyframes WHERE kf_id LIKE '%\\_f%' ESCAPE '\\'"
    ).fetchall()
    n_restored = 0
    for kf_id, vid, fidx in peak_rows:
        raw = Path(args.frames) / vid / f"{fidx:07d}.jpg"
        if raw.exists():
            conn.execute("UPDATE keyframes SET frame_path=?, selected=0 WHERE kf_id=?",
                         (str(raw), kf_id))
            n_restored += 1

    # 2) reference pool frame(_r): New candidates are generated every time, Delete directly, Depend on ref_pool_extract.py re-produce
    r_kfs = [r[0] for r in conn.execute(
        "SELECT kf_id FROM keyframes WHERE kf_id LIKE '%\\_r%' ESCAPE '\\'").fetchall()]
    conn.execute("DELETE FROM keyframes WHERE kf_id LIKE '%\\_r%' ESCAPE '\\'")

    # 3) persons Clear all(Regardless of peak value/reference pool): peak frame persons Depend on identity_cluster.py of
    #    extract() step(Without --cluster)Regenerate, Otherwise, "previous round" will remain Stage4 Frame selection results"
    #    The corresponding old identity_id, and this new round selected=1 The collection does not match the numbers(The pits that have been stepped on:
    #    1209 article a few weeks ago top-8 old times persons, Mistaken as a valid candidate by a new round of clustering)。
    n_persons = conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0]
    conn.execute("DELETE FROM persons")

    # 4) refs Clear table(downstream Stage6 output, Re-run must be rebuilt)
    conn.execute("DELETE FROM refs")
    conn.commit()
    print(f"Peak frame recovery: {n_restored}/{len(peak_rows)}; Reference pool frame deletion: {len(r_kfs)}; "
          f"persons Clear all: {n_persons}; refs Cleared")
    print("Next, manually delete the disk directory: targets/ crops/persons/ crops/embeddings/ "
          "crops/refs_clean/ crops/refs_raw/ (Script does not touch the file system)")
    print("Next step sequence(Note that it contains Stage5 of extract, I missed this step before):")
    print("  1. interaction_density.py --top-per-video N          (Select peak frame)")
    print("  2. crop_targets.py                                    (Cutting target image)")
    print("  3. identity_cluster.py (Without --cluster)               (peak frame: Layoffs+feature, forStage10Identify who is in the target frame)")
    print("  4. ref_pool_extract.py                                (Trough frame: Layoffs+feature, reference candidate)")
    print("  5. identity_cluster.py --cluster                      (For the above two batches persons cluster together)")
    print("  6. ref_crop.py                                        (Only select reference images from the valley frame candidates)")


if __name__ == "__main__":
    main()
