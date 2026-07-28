#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MPIE-Bench review / pilot portal.

Serves local pilot outputs:
  /            -> pilot HTML report (build_report.py)
  /images/<fn> -> local crop originals
Refresh the page after rebuilding the report; no restart needed.
Start: nohup setsid python app.py > app.log 2>&1 &
"""
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from flask import Flask, send_from_directory, abort, request

sys.path.insert(0, str(Path.home() / "cc0_review"))
import xmove  # noqa: E402  cross-category scene physical move

BENCH = Path(".")
REPORT = BENCH / "docs/01_research/pilot_case_study_report.html"
WALKTHROUGH = BENCH / "docs/01_research/pipeline_walkthrough.html"
IMG_DIR = BENCH / "data/crops/pilot_case_study"
CC0_REVIEW = Path.home() / "cc0_review" / "review.html"
CC0_FLAT = Path.home() / "cc0_review" / "flat"
CC0_FULL_ROOT = BENCH / "data" / "cc0_review_full"
H4D_REVIEW = Path.home() / "h4d_board" / "h4d_review.html"
H4D_FLAT = Path.home() / "h4d_board" / "h4d_board_pkg" / "flat"

app = Flask(__name__)


_STYLE = """<style>body{font-family:-apple-system,"PingFang SC",sans-serif;background:#f4f5f8;color:#1f2430;margin:0;padding:24px}
h2{margin:4px 0 18px} .card{display:block;background:#fff;border:1px solid #e8eaee;border-radius:12px;padding:16px 20px;margin:10px 0;
max-width:640px;text-decoration:none;color:#1f2430} .card:hover{border-color:#4f7cff}
.card b{font-size:16px} .card .d{color:#6b7280;font-size:13px;margin-top:4px} .card{position:relative}
.done{position:absolute;top:14px;right:16px;background:#22a06b;color:#fff;font-size:12px;padding:3px 12px;border-radius:999px}
.pill{background:#22a06b;color:#fff;font-size:12px;padding:2px 10px;border-radius:999px}
table{border-collapse:collapse;background:#fff;border-radius:12px;overflow:hidden;border:1px solid #e8eaee}
th,td{padding:10px 16px;font-size:14px;text-align:left;border-bottom:1px solid #f0f2f5}
th{background:#fafbfc;color:#6b7280;font-size:12px} a{color:#4f7cff;text-decoration:none} a:hover{text-decoration:underline}
.na{color:#c0c4cc}</style>"""


FINAL_ROOT = BENCH / "data" / "final"


def _final_targets(name):
    """Final target image count; None if not finalized."""
    d = FINAL_ROOT / name / "targets"
    if not d.is_dir():
        return None
    return sum(1 for p in d.iterdir() if p.suffix == ".jpg")


def _done(cat):
    """Done if board marked complete or dataset finalized."""
    return _final_targets(cat) is not None or (CC0_FULL_ROOT / cat / "reviewed_done").exists()


def _cc0_targets(cat):
    """Live target count = flat/ jpgs without _r minus board deletes (board_state.json dtgt).
    Board edits only update shared state, not files; subtract deletes for live count."""
    flat = CC0_FULL_ROOT / cat / "flat"
    if not flat.is_dir():
        return 0
    dtgt = set()
    p = CC0_FULL_ROOT / cat / "board_state.json"
    if p.exists():
        try:
            dtgt = set(json.loads(p.read_text()).get("dtgt") or [])
        except Exception:
            pass
    return sum(1 for p in flat.iterdir()
               if p.suffix == ".jpg" and "_r" not in p.name and p.name not in dtgt)


def _h4d_targets():
    if not H4D_FLAT.is_dir():
        return 0
    return sum(1 for p in H4D_FLAT.iterdir() if "__t" in p.name)


_FOREIGN_DATASETS = {"chi3d", "panoptic", "egohumans", "ava", "harmony4d"}   # co-located but listed separately on portal


# ============ test set (benchmark) split ============
# Board "Move to test set" writes scene anchors to board_state.json tset;
# summarize live from scenes_export.json (gen_scene_board.py); train/test counts split.

def _board_state(cat):
    p = CC0_FULL_ROOT / cat / "board_state.json"
    try:
        return json.loads(p.read_text()) if p.exists() else {}
    except Exception:
        return {}


def _test_split(cat):
    """(test target count, test scene list). If any member of merge chain marked, whole scene is test."""
    st = _board_state(cat)
    tset = set(st.get("tset") or [])
    exp = CC0_FULL_ROOT / cat / "scenes_export.json"
    if not tset or not exp.exists():
        return 0, []
    scenes = json.loads(exp.read_text())
    dtgt = set(st.get("dtgt") or [])
    dref = set(st.get("del") or [])
    star = st.get("star") or {}
    into = {m[0]: m[1] for m in (st.get("smerge") or []) if len(m) == 2}

    def fin(a):
        seen = set()
        while a in into and a not in seen:
            seen.add(a)
            a = into[a]
        return a

    marked = {fin(a) for a in tset}
    picked = {}
    for s in scenes:
        root = fin(s["anchor"])
        if root not in marked:
            continue
        g = picked.setdefault(root, {"anchor": root, "videos": [], "targets": [], "refs": []})
        g["videos"] += s["videos"]
        g["targets"] += [t for t in s["targets"] if t not in dtgt]
        for a in s["actors"]:
            live = [fn for fn in a["refs"] if fn not in dref]
            if live:
                chosen = star.get(a["id"])
                g["refs"].append(chosen if chosen in live else live[0])
    out = [g for g in picked.values() if g["targets"]]
    return sum(len(g["targets"]) for g in out), out


def _h4d_class(name):
    if name.startswith("ballroom"):
        return "dance"
    if name.startswith("grappling"):
        return "wrestle_grapple"
    return "fight_combat"       # karate / mma*


_CHI3D_ACT = {"Grab": "wrestle_grapple", "Handshake": "handshake", "Hit": "fight_combat",
              "Kick": "fight_combat", "Push": "fight_combat", "HoldingHands": "hand_hold",
              "Hug": "hug", "Posing": "face_to_face_talk"}

# Test quota: (class, contact tier, target quota, source note) — total 2350, C3 overweight (59%)
# 2026-07-18: former C3∪C4 merged to C3 (weight-bearing + grappling/combat)
_QUOTA = [
    ("face_to_face_talk", "C1", 100, "CC0; no-contact control (incl. CHI3D Posing)"),
    ("handshake", "C1", 120, "CC0 + CHI3D Handshake"),
    ("high_five", "C1", 100, "CC0"),
    ("hand_hold", "C2", 140, "CC0 + CHI3D HoldingHands"),
    ("arm_around_shoulder", "C2", 150, "CC0"),
    ("dance", "C2", 160, "CC0 + H4D ballroom"),
    ("hug", "C3", 240, "CC0 + CHI3D Hug"),
    ("piggyback", "C3", 180, "CC0"),
    ("carry_lift", "C3", 200, "CC0"),
    ("dance_lift", "C3", 180, "CC0"),
    ("wrestle_grapple", "C3", 260, "CC0 + H4D grappling + CHI3D Grab"),
    ("fight_combat", "C3", 260, "CC0 + H4D karate/mma + CHI3D Hit/Kick/Push"),
    ("restrain_pin", "C3", 60, "CC0 pool only 67; gap filled by wrestle"),
    ("other_multi_person", "3+", 200, "CC0 3+ scenes + Panoptic (Count dim)"),
]


def _quota_counts():
    """Class -> test target count (CHI3D by action, H4D by scene, Panoptic -> 3+)."""
    got = defaultdict(int)
    for c in _cc0_cats():
        got[c] += _test_split(c)[0]
    for g in _test_split("chi3d")[1]:
        for t in g["targets"]:
            m = re.search(r"_f([A-Za-z]+)-", t)
            got[_CHI3D_ACT.get(m.group(1) if m else "", "other_multi_person")] += 1
    got["other_multi_person"] += _test_split("panoptic")[0]
    for g in _test_split("harmony4d")[1]:               # scene name (= video id/anchor) -> interaction class
        got[_h4d_class(g["anchor"])] += len(g["targets"])
    return got


_PROMPT_DIST_SUMMARY = BENCH / "data/manifests/prompt_distribution/summary.json"


def _prompt_dist_summary():
    """Load prompt_distribution/summary.json (C0–C3 x natural interaction clustering)."""
    if not _PROMPT_DIST_SUMMARY.is_file():
        return None
    try:
        return json.loads(_PROMPT_DIST_SUMMARY.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _cc0_cats():
    """14 CC0 interaction categories (exclude co-located foreign datasets)."""
    if not CC0_FULL_ROOT.exists():
        return []
    return sorted(p.name for p in CC0_FULL_ROOT.iterdir()
                  if p.is_dir() and p.name not in _FOREIGN_DATASETS)


@app.route("/")
def index():
    cats = _cc0_cats()
    n_cc0_targets = sum(_cc0_targets(c) for c in cats)
    # Finalized datasets: count from data/final/<ds>/targets; show "Done" badge
    fin_h4d = _final_targets("harmony4d")
    fin_chi3d = _final_targets("chi3d")
    fin_pano = _final_targets("panoptic")
    fin_ava = _final_targets("ava")
    h4d_board = (CC0_FULL_ROOT / "harmony4d" / "scene_board.html").exists()
    # H4D finalized but uses scene board: live count from board when present
    n_h4d = _cc0_targets("harmony4d") if h4d_board else (fin_h4d if fin_h4d is not None else _h4d_targets())
    n_chi3d = fin_chi3d if fin_chi3d is not None else _cc0_targets("chi3d")
    n_pano = fin_pano if fin_pano is not None else _cc0_targets("panoptic")
    n_ava = fin_ava if fin_ava is not None else _cc0_targets("ava")
    h4d_ok = H4D_REVIEW.exists()
    chi3d_ok = (CC0_FULL_ROOT / "chi3d" / "scene_board.html").exists()
    pano_ok = (CC0_FULL_ROOT / "panoptic" / "scene_board.html").exists()
    ava_ok = (CC0_FULL_ROOT / "ava" / "scene_board.html").exists()
    chi3d_href = ' href="/cc0scene/chi3d"' if chi3d_ok else ''
    pano_href = ' href="/cc0scene/panoptic"' if pano_ok else ''
    ava_href = ' href="/cc0scene/ava"' if ava_ok else ''
    badge = lambda ok: '<span class=done>Done</span>' if ok else ''
    n_done = sum(1 for c in cats if _done(c))
    # Train/test split: test targets deducted from training prep count
    t_cc0 = sum(_test_split(c)[0] for c in cats)
    t_h4d = _test_split("harmony4d")[0]
    t_chi3d = _test_split("chi3d")[0]
    t_pano = _test_split("panoptic")[0]
    t_ava = _test_split("ava")[0]
    n_test = t_cc0 + t_h4d + t_chi3d + t_pano + t_ava
    dist = _prompt_dist_summary()
    n_dist = (dist or {}).get("N") or n_test
    n_types = len((dist or {}).get("by_interaction_type") or {})
    type_label = f"{n_types} natural interaction types" if n_types else "C0–C3 × natural interaction"
    tt = lambda n, t: (f'train: <b>{n - t}</b> / test: <b style="color:#8b5cf6">{t}</b>')
    grand = n_h4d + n_cc0_targets + n_chi3d + n_pano + n_ava - n_test
    return f"""<html><head><meta charset=utf-8><title>MPIE-Bench dataset review</title>{_STYLE}</head><body>
<h2>MPIE-Bench review portal <span style="font-size:14px;color:#6b7280">training prep <b style="color:#1f2430">{grand}</b> · <a href="/testset" style="color:#8b5cf6">test set <b>{n_test}</b></a> target images</span></h2>
<a class=card href="/testset" style="border-color:#8b5cf6"><b>🧪 Test benchmark</b><div class=d>{type_label} · <b style="color:#8b5cf6">{n_dist}</b> targets assigned · classes from prompt distribution · view distribution & scenes</div></a>
<a class=card{' href="/cc0scene/harmony4d"' if h4d_board else ' href="/h4dreview"'}><b>Harmony4D</b>{badge(fin_h4d is not None)}<div class=d>multi-view GT / {tt(n_h4d, t_h4d)}</div></a>
<a class=card href="/cc0"><b>CC0 video mining</b>{badge(cats and n_done == len(cats))}<div class=d>14 interaction types / {tt(n_cc0_targets, t_cc0)}</div></a>
<a class=card{chi3d_href}><b>CHI3D</b>{badge(_done("chi3d"))}<div class=d>lab contact GT / {tt(n_chi3d, t_chi3d)}</div></a>
<a class=card{pano_href}><b>CMU Panoptic</b>{badge(_done("panoptic"))}<div class=d>multi-person social GT (3-8 people) / {tt(n_pano, t_pano)}</div></a>
<a class=card{ava_href}><b>AVA film interaction</b>{badge(_done("ava"))}<div class=d>top-100 films · 8 core contact classes / {tt(n_ava, t_ava)}</div></a>
</body></html>"""


@app.route("/pilot")
def pilot_report():
    if not REPORT.exists():
        abort(404, "report not built yet — run build_report.py")
    return send_from_directory(REPORT.parent, REPORT.name)


@app.route("/cc0")
def cc0_index():
    cats = _cc0_cats()

    def row(c):
        fin = _final_targets(c)
        n = fin if fin is not None else _cc0_targets(c)
        t = _test_split(c)[0]
        mark = ' <span class=pill>Done</span>' if _done(c) else ''
        if (CC0_FULL_ROOT / c / "scene_board.html").exists():
            link = f'<a href="/cc0scene/{c}">Open scene board</a>'
        else:
            link = '<span class=na>Board not generated (re-ingest)</span>'
        return f'<tr><td><b>{c}</b>{mark}</td><td>{n - t}</td><td>{link}</td></tr>'

    rows_html = "".join(row(c) for c in cats) or '<tr><td colspan=3>No review packages uploaded yet</td></tr>'
    total = sum((_final_targets(c) if _final_targets(c) is not None else _cc0_targets(c)) - _test_split(c)[0]
                for c in cats)
    return f"""<html><head><meta charset=utf-8><title>CC0 category portal</title>{_STYLE}</head><body>
<h2>CC0 video mining — categories <span style="font-size:14px;color:#6b7280">(training total {total} · <a href="/testset" style="color:#8b5cf6">test overview</a>)</span></h2>
<table><tr><th>Category</th><th>Training (targets)</th><th>Scene board</th></tr>{rows_html}</table>
<p style="font-size:13px;color:#6b7280">New categories appear after ingest_review_pkg.sh. Cross-category merge: type "hug 12" on any board.</p>
</body></html>"""


@app.route("/walkthrough")
def walkthrough():
    if not WALKTHROUGH.exists():
        abort(404, "walkthrough page not found")
    return send_from_directory(WALKTHROUGH.parent, WALKTHROUGH.name)


@app.route("/images/<path:fn>")
def images(fn):
    return send_from_directory(IMG_DIR, fn)


@app.route("/cc0review")
def cc0review():
    if not CC0_REVIEW.exists():
        abort(404, "cc0 review page not generated")
    return send_from_directory(CC0_REVIEW.parent, CC0_REVIEW.name)


@app.route("/cc0img/<path:fn>")
def cc0img(fn):
    return send_from_directory(CC0_FLAT, fn)


@app.route("/cc0review/<category>")
def cc0review_full(category):
    p = CC0_FULL_ROOT / category / "review.html"
    if not p.exists():
        abort(404, f"{category}: review.html not generated yet — run gen_review_full.py {category} on dev machine first")
    return send_from_directory(p.parent, p.name)


@app.route("/cc0img/<category>/<path:fn>")
def cc0img_full(category, fn):
    return send_from_directory(CC0_FULL_ROOT / category / "flat", fn)


@app.route("/cc0cluster/<category>")
def cc0cluster_full(category):
    p = CC0_FULL_ROOT / category / "cluster_review.html"
    if not p.exists():
        abort(404, f"{category}: cluster_review.html not generated yet — run gen_cluster_review.py {category} on dev machine first")
    return send_from_directory(p.parent, p.name)


@app.route("/cc0scene/<category>")
def cc0scene_full(category):
    p = CC0_FULL_ROOT / category / "scene_board.html"
    if not p.exists():
        abort(404, f"{category}: scene_board.html not generated yet — run gen_scene_board.py {category} on dev machine first")
    return send_from_directory(p.parent, p.name)


@app.route("/cc0state/<category>", methods=["GET", "POST"])
def cc0state(category):
    """Scene board shared state: GET on open, POST after each action. Saved under category dir;
    all browsers share progress (previously per-user localStorage)."""
    if not re.fullmatch(r"[a-z0-9_]+", category) or not (CC0_FULL_ROOT / category).is_dir():
        abort(404)
    p = CC0_FULL_ROOT / category / "board_state.json"
    if request.method == "POST":
        data = request.get_json(force=True, silent=True)
        if not isinstance(data, dict):
            abort(400)
        p.write_text(json.dumps(data, ensure_ascii=False))
        return {"ok": True}
    return app.response_class(p.read_text() if p.exists() else "{}", mimetype="application/json")


_GEN = Path.home() / "cc0_review/gen_scene_board.py"
_PY = Path.home() / "miniconda3/bin/python"


@app.route("/cc0done/<category>", methods=["GET", "POST"])
def cc0done(category):
    """Category done flag: toggled from board header; writes reviewed_done; shared team-wide."""
    if not re.fullmatch(r"[a-z0-9_]+", category) or not (CC0_FULL_ROOT / category).is_dir():
        abort(404)
    p = CC0_FULL_ROOT / category / "reviewed_done"
    if request.method == "POST":
        want = bool((request.get_json(force=True, silent=True) or {}).get("done"))
        if want:
            p.write_text("")
        else:
            p.unlink(missing_ok=True)
        return {"ok": True, "done": want}
    return {"done": _done(category)}


@app.route("/cc0xmove", methods=["POST"])
def cc0xmove():
    """Cross-category scene move: actor groups + images + manual state into target category/scene."""
    req = request.get_json(force=True, silent=True) or {}
    src, dst = str(req.get("src") or ""), str(req.get("dst") or "")
    anchor = str(req.get("anchor") or "")
    for c in (src, dst):
        if not re.fullmatch(r"[a-z0-9_]+", c) or not (CC0_FULL_ROOT / c).is_dir():
            return {"ok": False, "err": f"Invalid category: {c}"}
    if src == dst:
        return {"ok": False, "err": "Same category: merge by scene number locally"}
    if req.get("dstAnchor"):
        dst_anchor = str(req["dstAnchor"])
    else:
        idx = json.loads((CC0_FULL_ROOT / dst / "scene_index.json").read_text())
        n = req.get("n")
        if not isinstance(n, int) or not 1 <= n <= len(idx):
            return {"ok": False, "err": f"{dst} has only {len(idx)} scenes"}
        dst_anchor = idx[n - 1]
    try:
        info = xmove.move_scene(src, anchor, dst, dst_anchor)
    except xmove.SceneNotFound:
        return {"ok": False, "err": "anchor-gone"}    # already moved; frontend treats as done
    except Exception as e:
        return {"ok": False, "err": f"{type(e).__name__}: {e}"}
    for _ in range(2):                                # regen src+dst twice (refresh scene indices)
        for cat in (src, dst):
            subprocess.run([str(_PY), str(_GEN), cat], check=False, capture_output=True)
    subprocess.Popen(                                 # regen other categories twice (XIDX snapshot)
        ["bash", "-c",
         f'for i in 1 2; do for d in "{CC0_FULL_ROOT}"/*/; do c=$(basename "$d"); '
         f'[ -f "$d/ref_clusters.json" ] && "{_PY}" "{_GEN}" "$c" >/dev/null 2>&1; done; done'],
        start_new_session=True)
    # merged scenes may sort earlier; return new scene number
    return {"ok": True, "dst_scene": xmove.scene_number(dst, dst_anchor), **info}


@app.route("/cc0merge/<category>")
def cc0merge_full(category):
    p = CC0_FULL_ROOT / category / "scene_merge_review.html"
    if not p.exists():
        abort(404, f"{category}: scene_merge_review.html not generated yet")
    return send_from_directory(p.parent, p.name)


@app.route("/cc0review_full")
def cc0review_full_index():
    cats = sorted(p.name for p in CC0_FULL_ROOT.iterdir() if p.is_dir()) if CC0_FULL_ROOT.exists() else []
    links = "".join(f'<li><a href="/cc0review/{c}">{c}</a></li>' for c in cats) or "<li>(No review packages uploaded yet)</li>"
    return f"<html><body style='font-family:sans-serif'><h3>CC0 full review — ready categories</h3><ul>{links}</ul></body></html>"


@app.route("/h4dreview")
def h4dreview():
    if not H4D_REVIEW.exists():
        abort(404, "h4d review page not generated")
    return send_from_directory(H4D_REVIEW.parent, H4D_REVIEW.name)


@app.route("/h4dimg/<path:fn>")
def h4dimg(fn):
    return send_from_directory(H4D_FLAT, fn)


@app.route("/testset")
def testset():
    """Test set overview: clustering distribution + assigned test scenes per board."""
    dist = _prompt_dist_summary()
    # ---- category table from prompt_distribution clustering ----
    tier_color = {
        "C0": "#64748b", "C1": "#94a3b8", "C2": "#3b82f6",
        "C3": "#f59e0b",  # high-contact (merged former C3∪C4)
    }
    qrows = []
    got_total = 0
    n_scenes = 0
    if dist:
        got_total = int(dist.get("N") or 0)
        n_scenes = int(dist.get("n_scenes") or 0)
        tax = dist.get("taxonomy") or {}
        c_labels = tax.get("c_labels") or {}
        by_c = dist.get("by_c_action") or {}
        order = tax.get("interaction_types") or []
        action_to_c = tax.get("action_to_c") or {}
        tiers = [t for t in ("C0", "C1", "C2", "C3") if t in by_c] or sorted(by_c)
        shown = set()
        for tier in tiers:
            acts = by_c.get(tier) or {}
            names = [a for a in order if a in acts] + [a for a in acts if a not in order]
            for cls in names:
                n = int(acts[cls])
                pct = round(n * 1000 / got_total) / 10 if got_total else 0
                bar_w = min(100, round(n * 100 / got_total)) if got_total else 0
                bar = (f'<div style="background:#eef0f3;border-radius:4px;width:140px;height:10px">'
                       f'<div style="background:#8b5cf6;width:{bar_w}%;height:10px;border-radius:4px"></div></div>')
                tier_tip = c_labels.get(tier, tier)
                qrows.append(
                    f'<tr><td><span title="{tier_tip}" style="background:{tier_color.get(tier, "#6b7280")};'
                    f'color:#fff;font-size:11px;padding:2px 8px;border-radius:999px">{tier}</span></td>'
                    f'<td><b>{cls}</b></td>'
                    f'<td style="color:#8b5cf6;font-weight:700">{n}</td>'
                    f'<td>{pct}%</td><td>{bar}</td></tr>')
                shown.add((tier, cls))
        by_it = dist.get("by_interaction_type") or {}
        for cls, n in by_it.items():
            tier = action_to_c.get(cls, "")
            if (tier, cls) in shown:
                continue
            if any(cls in (by_c.get(t) or {}) for t in by_c):
                continue
            n = int(n)
            pct = round(n * 1000 / got_total) / 10 if got_total else 0
            bar_w = min(100, round(n * 100 / got_total)) if got_total else 0
            bar = (f'<div style="background:#eef0f3;border-radius:4px;width:140px;height:10px">'
                   f'<div style="background:#8b5cf6;width:{bar_w}%;height:10px;border-radius:4px"></div></div>')
            qrows.append(
                f'<tr><td><span style="background:{tier_color.get(tier, "#6b7280")};'
                f'color:#fff;font-size:11px;padding:2px 8px;border-radius:999px">{tier or "—"}</span></td>'
                f'<td><b>{cls}</b></td>'
                f'<td style="color:#8b5cf6;font-weight:700">{n}</td>'
                f'<td>{pct}%</td><td>{bar}</td></tr>')
    else:
        got = _quota_counts()
        got_total = sum(got.values())
        for cls, tier, quota, src in _QUOTA:
            n = got.get(cls, 0)
            pct = round(n * 1000 / got_total) / 10 if got_total else 0
            qrows.append(
                f'<tr><td><span style="background:{tier_color.get(tier, "#6b7280")};'
                f'color:#fff;font-size:11px;padding:2px 8px;border-radius:999px">{tier}</span></td>'
                f'<td><b>{cls}</b></td><td style="color:#8b5cf6;font-weight:700">{n}</td>'
                f'<td>{pct}%</td><td style="color:#6b7280;font-size:12px">{src}</td></tr>')

    # ---- test scenes assigned on each board ----
    sections = []
    board_cats = ["harmony4d"] + _cc0_cats() + ["chi3d", "panoptic", "ava"]
    for c in board_cats:
        n, groups = _test_split(c)
        if not groups:
            continue
        warn = ' <span style="color:#ef4444;font-size:12px">⚠ AVA is restricted tier (film pixels not redistributable); do not add to public benchmark</span>' if c == "ava" else ''
        cards = []
        for g in sorted(groups, key=lambda g: -len(g["targets"])):
            refs = "".join(f'<img loading=lazy src="/cc0img/{c}/{fn}" style="height:72px;border-radius:5px">' for fn in g["refs"][:4])
            tgts = "".join(f'<img loading=lazy src="/cc0img/{c}/{fn}" style="height:72px;border-radius:5px;border:2px solid #f59e0b">' for fn in g["targets"][:4])
            cards.append(
                f'<div style="display:inline-block;background:#fff;border:1px solid #e8eaee;border-radius:10px;padding:8px;margin:5px;vertical-align:top">'
                f'<div style="font-size:12px;color:#6b7280;margin-bottom:4px"><code>{g["anchor"]}</code> · {len(g["videos"])} videos · {len(g["refs"])} refs · <b style="color:#1f2430">{len(g["targets"])}</b> targets</div>'
                f'<div style="display:flex;gap:4px;flex-wrap:wrap">{refs}<span style="color:#d1d5db;align-self:center">→</span>{tgts}</div></div>')
        sections.append(f'<h3 style="margin:18px 0 4px">{c} <span style="font-size:13px;color:#8b5cf6">{len(groups)} scenes / {n} targets</span>'
                        f' <a href="/cc0scene/{c}" style="font-size:13px">Adjust on board</a>{warn}</h3>{"".join(cards)}')

    prompts_ok = (BENCH / "data/manifests/prompts_full/index.html").exists()
    dist_ok = (BENCH / "data/manifests/prompt_distribution/index.html").exists()
    btn = (
        '<span style="display:inline-flex;gap:8px;flex-wrap:wrap;margin-left:10px;vertical-align:middle">'
        + (f'<a href="/testset/prompts/" style="font-size:12px;padding:4px 10px;border:1px solid #c4b5fd;border-radius:999px;background:#f5f3ff;color:#6d28d9;text-decoration:none">Full prompt review</a>' if prompts_ok else '')
        + (f'<a href="/testset/prompt_distribution/" style="font-size:12px;padding:4px 10px;border:1px solid #c4b5fd;border-radius:999px;background:#f5f3ff;color:#6d28d9;text-decoration:none">Prompt distribution</a>' if dist_ok else '')
        + '</span>'
    )
    scene_note = f" · {n_scenes} scenes" if n_scenes else ""
    src_note = ('Class names and counts from <a href="/testset/prompt_distribution/">prompt distribution</a>'
                '(C0–C3 × layers.interaction clustering, not board cat).'
                if dist else
                '⚠ prompt_distribution/summary.json missing; falling back to board quota counts.')
    return f"""<html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Test set overview</title>{_STYLE}</head><body>
<h2>🧪 Test benchmark overview <span style="font-size:14px;color:#6b7280"><b style="color:#8b5cf6">{got_total}</b> targets assigned{scene_note} · <a href="/">Home</a></span>{btn}</h2>
<p style="font-size:13px;color:#6b7280;max-width:900px">{src_note}
Click purple "Move to test set" on a scene board to assign the whole scene (videos leave training; click again to undo).
Selection: ①≥1 ★ clean frontal ref per actor ② clear pose & contact ③ avoid duplicate models within class ④ ≤20 targets per scene
⑤ 3+ person scenes 15–20% ⑥ AVA/K700 restricted tier — training only, do not assign.</p>
<table style="margin-bottom:20px"><tr><th>Tier</th><th>Class (natural interaction)</th><th>Count</th><th>Share</th><th>Bar</th></tr>
{''.join(qrows)}
<tr style="background:#fafbfc"><td></td><td><b>Total</b></td><td style="color:#8b5cf6;font-weight:700">{got_total}</td><td>100%</td><td></td></tr></table>
{''.join(sections) or '<p style="color:#9ca3af">No scenes in test set yet — open Harmony4D / CC0 / CHI3D / Panoptic boards and click "Move to test set".</p>'}
</body></html>"""


@app.route("/finalimg/<ds>/<sub>/<path:fn>")
def finalimg(ds, sub, fn):
    if sub not in ("refs", "targets") or not re.fullmatch(r"[a-z0-9_]+", ds):
        abort(404)
    return send_from_directory(FINAL_ROOT / ds / sub, fn)


def _safe_manifest_file(root: Path, fn: str):
    """Serve a file under root; block path escape."""
    root = root.resolve()
    target = (root / fn).resolve()
    if root not in target.parents and target != root:
        abort(404)
    if not target.is_file():
        abort(404)
    return send_from_directory(target.parent, target.name)


@app.route("/testset/prompts/")
@app.route("/testset/prompts/<path:fn>")
def testset_prompts(fn="index.html"):
    """Full prompt caption review page (was evaluation dashboard/manifests/prompts_full)."""
    return _safe_manifest_file(BENCH / "data/manifests/prompts_full", fn)


@app.route("/testset/prompt_distribution/")
@app.route("/testset/prompt_distribution/<path:fn>")
def testset_prompt_distribution(fn="index.html"):
    """Test-set prompt distribution (was evaluation dashboard/manifests/prompt_distribution)."""
    return _safe_manifest_file(BENCH / "data/manifests/prompt_distribution", fn)


@app.route("/captiontrial")
def captiontrial():
    p = BENCH / "data/manifests/caption_trial/trial_v1.html"
    if not p.exists():
        abort(404, "trial not run yet — python code/pipeline/07_caption/caption_trial_v1.py")
    return send_from_directory(p.parent, p.name)


@app.route("/health")
def health():
    return {"ok": True, "report_exists": REPORT.exists()}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
