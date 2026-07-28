#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cross-category scene move (physical migration).

Requirement (2026-07-10): cross-category merge must appear live on target board for rebinding.
Move actor groups + images + manual state into target category; write scene_merges.json;
structured merge into target scene; all edits continue natively on target category.

aid stability: mark moved members {"moved_to": <cat>} in src ref_clusters (keep slot),
gen_scene_board keeps empty slots → remaining a<i> ids stable (star/bind by aid).
New actor groups append at end on target; existing aids stable.
Called by Flask /cc0xmove; CLI for legacy records.
"""
import json
from pathlib import Path

ROOT = Path(".") / "data" / "cc0_review_full"
MAX_GROUP, MIN_SIM = 12, 0.32      # must match gen_scene_board.py


class SceneNotFound(Exception):
    pass


def _empty():
    return {"del": [], "dtgt": [], "star": {}, "bind": {}, "stack": [], "smerge": []}


def _expand(data):
    """Expand groups→actors like gen_scene_board; empty slot for moved members."""
    actors = []
    for g in data["groups"]:
        ms = g["members"]
        if len(ms) > 1 and (len(ms) > MAX_GROUP or g["avg_sim"] < MIN_SIM):
            for m in ms:
                actors.append([] if m.get("moved_to") else [m])
        else:
            actors.append([m for m in ms if not m.get("moved_to")])
    return actors


def _vid(fn):
    return fn.split("_r")[0].split("_f")[0]


def _resolve(anchor, smerge):
    seen = set()
    cur = anchor
    while True:
        nxt = next((p[1] for p in smerge if len(p) == 2 and p[0] == cur), None)
        if nxt is None or cur in seen:
            return cur
        seen.add(cur)
        cur = nxt


def _load(cat):
    root = ROOT / cat
    data = json.loads((root / "ref_clusters.json").read_text())
    actors = _expand(data)
    videos = sorted({_vid(p.name) for p in (root / "flat").glob("*.jpg")})
    parent = {v: v for v in videos}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for mem in actors:
        vs = sorted({m["video_id"] for m in mem})
        for v in vs[1:]:
            if vs[0] in parent and v in parent:
                parent[find(vs[0])] = find(v)
    mf = root / "scene_merges.json"
    if mf.exists():
        for grp in json.loads(mf.read_text()):
            grp = [v for v in grp if v in parent]
            for v in grp[1:]:
                parent[find(grp[0])] = find(v)
    sp = root / "board_state.json"
    state = {**_empty(), **json.loads(sp.read_text())} if sp.exists() else _empty()
    return root, data, actors, videos, find, state


def scene_number(cat, anchor):
    """Scene number (1-based) for anchor after board regen; show after reorder."""
    try:
        _, _, _, videos, find, _ = _load(cat)
        if anchor not in videos:
            return None
        root_v = find(anchor)
        comp_anchor = sorted(v for v in videos if find(v) == root_v)[0]
        idx = json.loads((ROOT / cat / "scene_index.json").read_text())
        return idx.index(comp_anchor) + 1
    except Exception:
        return None


def move_scene(src_cat, anchor, dst_cat, dst_anchor):
    """Move whole scene (incl. in-category merges) from src anchor to dst anchor scene."""
    sroot, sdata, sactors, svideos, sfind, sstate = _load(src_cat)
    if anchor not in svideos:
        raise SceneNotFound(anchor)
    droot, ddata, dactors, dvideos, dfind, dstate = _load(dst_cat)

    # effective scene = connected component ∪ in-category merge chain
    smg = [m for m in sstate["smerge"] if len(m) == 2]
    comps = {}
    for v in svideos:
        comps.setdefault(sfind(v), []).append(v)
    comp_anchor = {r: sorted(vs)[0] for r, vs in comps.items()}
    term = _resolve(comp_anchor[sfind(anchor)], smg)
    moved = set()
    for r, vs in comps.items():
        if _resolve(comp_anchor[r], smg) == term:
            moved.update(vs)

    dst_anchor = _resolve(dst_anchor, [m for m in dstate["smerge"] if len(m) == 2])
    if dst_anchor not in dvideos:
        raise ValueError(f"target anchor {dst_anchor} not in {dst_cat}")

    # migrate actor groups: append copy on dst, mark moved_to on src
    base = len(dactors)
    aid_map = {}
    n_ref = 0
    for i, mem in enumerate(sactors):
        if mem and {m["video_id"] for m in mem} <= moved:
            ddata["groups"].append({"avg_sim": 0.99, "members": [dict(m) for m in mem]})
            for m in mem:
                m["moved_to"] = dst_cat
            aid_map[f"a{i}"] = f"a{base + len(aid_map)}"
            n_ref += len(mem)
    if not aid_map:
        raise ValueError("no actors to migrate in this scene")

    snf = sdata.get("no_face", [])
    sdata["no_face"] = [fn for fn in snf if _vid(fn) not in moved]
    ddata.setdefault("no_face", []).extend(fn for fn in snf if _vid(fn) in moved)

    n_tgt = 0
    for p in sorted((sroot / "flat").glob("*.jpg")):
        if _vid(p.name) in moved:
            if "_r" not in p.name:
                n_tgt += 1
            p.rename(droot / "flat" / p.name)

    # structured merge into target scene (forced same scene at gen time)
    dmf = droot / "scene_merges.json"
    groups = json.loads(dmf.read_text()) if dmf.exists() else []
    groups.append([dst_anchor] + sorted(moved))
    dmf.write_text(json.dumps(groups, ensure_ascii=False))
    smf = sroot / "scene_merges.json"
    if smf.exists():
        gs = [[v for v in g if v not in moved] for g in json.loads(smf.read_text())]
        smf.write_text(json.dumps([g for g in gs if len(g) > 1], ensure_ascii=False))

    # migrate manual state
    for key in ("del", "dtgt"):
        gone = [fn for fn in sstate[key] if _vid(fn) in moved]
        sstate[key] = [fn for fn in sstate[key] if _vid(fn) not in moved]
        dstate[key].extend(fn for fn in gone if fn not in dstate[key])
    for aid, fn in list(sstate["star"].items()):
        if aid in aid_map:
            dstate["star"][aid_map[aid]] = fn
            del sstate["star"][aid]
    for fn, aids in list(sstate["bind"].items()):
        if _vid(fn) in moved:
            na = [aid_map[a] for a in aids if a in aid_map]
            if na:
                dstate["bind"][fn] = na
            del sstate["bind"][fn]
    sstate["stack"] = [e for e in sstate["stack"] if _vid(e[1]) not in moved]
    sstate["smerge"] = [m for m in sstate["smerge"]
                        if m[0] not in moved and not (len(m) == 2 and m[1] in moved)]

    (sroot / "ref_clusters.json").write_text(json.dumps(sdata, ensure_ascii=False))
    (droot / "ref_clusters.json").write_text(json.dumps(ddata, ensure_ascii=False))
    (sroot / "board_state.json").write_text(json.dumps(sstate, ensure_ascii=False))
    (droot / "board_state.json").write_text(json.dumps(dstate, ensure_ascii=False))
    return {"videos": len(moved), "actors": len(aid_map), "refs": n_ref,
            "targets": n_tgt, "src_state": sstate}
