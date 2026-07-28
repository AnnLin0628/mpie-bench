#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export the front-end scene board cc0_scene_map_<category>.json Apply to category manifest library.

scene mappingJSONis an artificial truth value(equivalenceHarmony4Dposture true value identity), Application content:
  1. Actor identity merge: All reference pictures of an actor(Possibly across videos) → unified canonical identity_id
  2. Deleted by✕reference picture: deleted_refs explicit list + The secret of being an actor who is not on the reserved list
  3. Deleted by✕target map: deleted_targets → keyframes.selected=0(Exit sample pool, Optional file deletion)
  4. ★Main reference picture: refs add is_primary Column and set, build_samples take it first
  5. Target graph binding: Write human_bindings surface(kf_id → identity_ids/ref_ids),
     build_samples Use manual conclusions when you see the table entries., no longer from persons Detection and derivation
JSONPlease file it as is manifests/scene_maps/ Don't delete it.

self contained(Standard library only), Can be copied directly to any machine and run.
usage: python apply_scene_map.py --db mpie_cc0_<cat>.db --map cc0_scene_map_<cat>.json \
        [--delete-files --refs-glob '~/mpie_data/cc0_full/<cat>_shard*/refs_clean' \
         --targets-glob '~/mpie_data/cc0_full/<cat>_shard*/targets']
"""
import argparse
import glob
import json
import sqlite3
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--delete-files", action="store_true", help="At the same time delete the✕The physical file of the image")
    ap.add_argument("--refs-glob", default="", help="Reference Picture Catalogglob(Cooperate--delete-files)")
    ap.add_argument("--targets-glob", default="", help="Target map directoryglob(Cooperate--delete-files)")
    args = ap.parse_args()

    def rm_file(dir_glob, stem):
        if not (args.delete_files and dir_glob):
            return
        for d in glob.glob(str(Path(dir_glob).expanduser())):
            p = Path(d) / f"{stem}.jpg"
            if p.exists():
                p.unlink()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    smap = json.loads(Path(args.map).read_text())

    cols = [r["name"] for r in conn.execute("PRAGMA table_info(refs)")]
    if "is_primary" not in cols:
        conn.execute("ALTER TABLE refs ADD COLUMN is_primary INTEGER DEFAULT 0")
    conn.execute("""CREATE TABLE IF NOT EXISTS human_bindings (
        kf_id TEXT PRIMARY KEY, identity_ids TEXT, ref_ids TEXT, scene INTEGER)""")
    conn.execute("UPDATE refs SET is_primary=0")
    conn.execute("DELETE FROM human_bindings")

    def ident_of(ref_stem):
        r = conn.execute("SELECT identity_id FROM refs WHERE ref_id=?", (ref_stem,)).fetchone()
        return r["identity_id"] if r else None

    n_merge = n_del = n_star = n_bind = n_miss = n_tgt = 0
    for sc in smap["scenes"]:
        for fn in sc.get("deleted_refs", []):           # explicit✕reference picture(Contains actors whose entire lines were deleted)
            stem = fn[:-4]
            if conn.execute("DELETE FROM refs WHERE ref_id=?", (stem,)).rowcount:
                n_del += 1
            rm_file(args.refs_glob, stem)
        for fn in sc.get("deleted_targets", []):        # explicit✕target map → Exit sample pool
            stem = fn[:-4]
            conn.execute("UPDATE keyframes SET selected=0 WHERE kf_id=?", (stem,))
            n_tgt += 1
            rm_file(args.targets_glob, stem)
        actor_ident = {}                     # label -> canonical identity_id
        actor_star = {}                      # label -> star ref_id(stem)
        for a in sc["actors"]:
            kept = [fn[:-4] for fn in a["refs"]]           # go.jpg = ref_id
            idents = sorted({i for i in (ident_of(s) for s in kept) if i})
            if not idents:
                n_miss += 1
                continue
            canon = idents[0]
            for other in idents[1:]:                        # Cross-video actor merging
                conn.execute("UPDATE persons SET identity_id=? WHERE identity_id=?", (canon, other))
                conn.execute("UPDATE refs SET identity_id=? WHERE identity_id=?", (canon, other))
                n_merge += 1
            # Reference pictures under the merged identity and not in the reserved list = quilt✕of → Delete line
            for r in conn.execute("SELECT ref_id FROM refs WHERE identity_id=?", (canon,)).fetchall():
                if r["ref_id"] not in kept:
                    conn.execute("DELETE FROM refs WHERE ref_id=?", (r["ref_id"],))
                    n_del += 1
                    rm_file(args.refs_glob, r["ref_id"])
            star = a.get("star")
            if star:
                conn.execute("UPDATE refs SET is_primary=1 WHERE ref_id=?", (star[:-4],))
                actor_star[a["label"]] = star[:-4]
                n_star += 1
            actor_ident[a["label"]] = canon
        for tgt_fn, labels in sc.get("bindings", {}).items():
            idents = [actor_ident[lb] for lb in labels if lb in actor_ident]
            refids = [actor_star.get(lb) for lb in labels if actor_star.get(lb)]
            if not idents:
                continue
            conn.execute("INSERT OR REPLACE INTO human_bindings VALUES (?,?,?,?)",
                         (tgt_fn[:-4], json.dumps(idents), json.dumps(refids), sc["scene"]))
            n_bind += 1
    conn.commit()
    print(f"Application completed: merge identities{n_merge}Second-rate, Delete reference picture{n_del}, Delete target image{n_tgt}, "
          f"Main reference picture{n_star}, target binding{n_bind}, Unable to locate actor{n_miss}")


if __name__ == "__main__":
    main()
