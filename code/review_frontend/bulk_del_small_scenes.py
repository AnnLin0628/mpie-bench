#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bulk soft-delete scenes with visible refs ≤ N (whole scene: refs + targets).
Usage: python bulk_del_small_scenes.py <category> [<category>...]

Same as board "Delete scene": structure from embedded export JSON in scene_board.html
(matches frontend, incl. cross-move placeholders) plus board_state.json deletes and
in-category merges (smerge). Writes del/dtgt + stack; board undo restores whole scene.
Note: refresh open board tabs after run (stale tab may overwrite server state).
"""
import json
import sys
from pathlib import Path

MAX_REFS = 1     # delete whole scene if visible refs <= this

ROOT = Path(".") / "data" / "cc0_review_full"


def load_scenes(cat):
    """Extract embedded scene export JSON from scene_board.html (same as frontend)."""
    html = (ROOT / cat / "scene_board.html").read_text()
    for line in html.splitlines():
        s = line.strip()
        if s.startswith("let scenes=") and s.endswith(";"):
            return json.loads(s[len("let scenes="):-1])
    sys.exit(f"{cat}: no scene JSON in scene_board.html; run gen_scene_board.py first")


def apply_smerge(scenes, smerge):
    """Replay in-category merges (same as frontend exp()); skip cross-category (3-tuple) records."""
    by = {s["anchor"]: s for s in scenes}
    for m in smerge:
        s = by.get(m[0])
        if len(m) == 3:
            if s is not None:
                s["_x"] = True
            continue
        d = by.get(m[1])
        while d and d.get("_into"):
            d = by.get(d["_into"])
        if not s or not d or s is d or s.get("_into") or s.get("_x") or d.get("_x"):
            continue
        d["actors"] += s["actors"]
        d["targets"] += s["targets"]
        s["_into"] = m[1]
    return [s for s in scenes if not s.get("_into") and not s.get("_x")]


def run(cat):
    sp = ROOT / cat / "board_state.json"
    st = json.loads(sp.read_text()) if sp.exists() else {}
    dels = set(st.get("del") or [])
    dtgt = set(st.get("dtgt") or [])
    star = st.get("star") or {}
    stack = st.get("stack") or []
    smerge = st.get("smerge") or []

    scenes = apply_smerge(load_scenes(cat), smerge)
    n_sc = n_ref = n_tgt = 0
    for s in scenes:
        vis_refs = [fn for a in s["actors"] for fn in a["refs"] if fn not in dels]
        vis_tgts = [fn for fn in s["targets"] if fn not in dtgt]
        if len(vis_refs) > MAX_REFS or not (vis_refs or vis_tgts):
            continue
        dels.update(vis_refs)
        dtgt.update(vis_tgts)
        stack.append(["scene", vis_refs, vis_tgts])
        n_sc += 1
        n_ref += len(vis_refs)
        n_tgt += len(vis_tgts)
    star = {aid: fn for aid, fn in star.items() if fn not in dels}

    st.update({"del": sorted(dels), "dtgt": sorted(dtgt), "star": star,
               "stack": stack, "smerge": smerge, "bind": st.get("bind") or {}})
    sp.write_text(json.dumps(st, ensure_ascii=False))
    remain = sum(1 for s in scenes
                 if [fn for fn in s["targets"] if fn not in dtgt])
    print(f"{cat}: deleted {n_sc} scenes (refs {n_ref} + targets {n_tgt}), remaining {remain}")


for c in sys.argv[1:]:
    run(c)
