#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""human consistency Anat/Inter Concept annotation front-end (port 8080）。

protocol v4: construct validity (see eval_construct_validity_principle.md）
need 3 Annotated by:ann_01 / ann_02 / ann_03(majority vote gold standard).
Placement:$PACK/judgments/human_consistency/human/<ann_id>/<key>.json

usage:
  PACK="$MPIE_TEST_PACK" python app.py
  bash ../run_annot_frontend.sh
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory, abort

_EVAL = Path(__file__).resolve().parents[1]
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))

from checklist_common import (  # noqa: E402
    ANALYSIS_PROTOCOL_ID,
    ANAT_ITEMS,
    ANAT_QUESTIONS as _ANAT_Q,
    DECISION_TREE,
    INTER_ITEMS,
    INTER_QUESTIONS as _INTER_Q,
    OVERALL_ITEMS,
    OVERALL_QUESTIONS as _OVERALL_Q,
    PROTOCOL_ID,
    SCHEME_VERSION,
    anat_pass_from_items,
    apply_inter_dependencies,
    atomic_write_json,
    construct_scores,
    inter_pass_from_items,
    normalize_inter_item,
    normalize_code,
    normalize_overall,
    pair_key,
)
from pack_io import pack_root  # noqa: E402

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"

ANNOTATORS = [
    {"id": "ann_01", "label": "annotator ann_01"},
    {"id": "ann_02", "label": "annotator ann_02"},
    {"id": "ann_03", "label": "annotator ann_03"},
]
N_REQUIRED = 3
SPLITS_ORDER = ("guide", "pilot", "holdout")

GUIDELINES = [
    f"question {PROTOCOL_ID} · analyze {ANALYSIS_PROTOCOL_ID}（{SCHEME_VERSION}): Quality construct, non-acceptance check.",
    "The sample has been difficult to oversample+Weak models are overweighted; please score carefully and do not default to a high score.",
    "Take the overall picture first Q_inter / Q_anat（1–5), then fill in the item level. Overall is the main anchor of human preference.",
    "S_inter_req：Ic=0 Full marks only 2/3——Pair of people+No mold wear≠High interaction score.",
    "Ic Only comment on fit;Ir Only the parts will be evaluated (no judgment will be made for not sticking) 0). People can be separated only by hand→A5=0, don’t hit I1=0。",
    "I0 Only the characters named in the command will be counted. Not sure to choose U。",
    *DECISION_TREE,
    "Forbidden to see mesh/VLM。ann_01/02/03 independent; gold label=The item-level majority vote is then counted S。",
]


def _hc(pack: Path) -> Path:
    return pack / "judgments" / "human_consistency"


def load_split(pack: Path) -> dict:
    p = _hc(pack) / "_split.json"
    if not p.is_file():
        raise FileNotFoundError(p)
    return json.loads(p.read_text(encoding="utf-8"))


def load_prompt_zh(pack: Path) -> dict:
    p = _hc(pack) / "prompt_zh.json"
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def list_items(pack: Path, split: str | None = None) -> list[dict]:
    data = load_split(pack)
    zh_map = load_prompt_zh(pack)
    out = []
    for name in SPLITS_ORDER:
        if split and name != split:
            continue
        for it in data.get("splits", {}).get(name) or []:
            d = dict(it)
            d["split"] = name
            d.setdefault("key", pair_key(d["sample_id"], d["model_id"]))
            rel = d.get("img_relpath") or ""
            if not rel.startswith("judgments/") and not rel.startswith("outputs/"):
                mid, sid = d["model_id"], d["sample_id"]
                for e in (".png", ".jpg", ".jpeg", ".webp"):
                    cand = f"judgments/human_consistency/media/{mid}/{sid}{e}"
                    if (pack / cand).is_file():
                        rel = cand
                        break
            d["img_relpath"] = rel
            z = zh_map.get(d["sample_id"]) or {}
            d["prompt_zh"] = d.get("prompt_zh") or z.get("prompt_zh") or ""
            d["prompt_en"] = d.get("prompt") or z.get("prompt_en") or ""
            out.append(d)
    return out


def ann_path(pack: Path, ann_id: str, key: str) -> Path:
    return _hc(pack) / "human" / ann_id / f"{key}.json"


def progress_for(pack: Path, ann_id: str, split: str | None = None) -> dict:
    items = list_items(pack, split=split)
    done = 0
    root = _hc(pack) / "human" / ann_id
    if root.is_dir():
        have = {p.stem for p in root.glob("*.json") if not p.name.startswith("_")}
        done = sum(1 for it in items if it["key"] in have)
    return {"done": done, "total": len(items)}


def create_app(pack: Path) -> Flask:
    app = Flask(__name__, static_folder=str(STATIC), static_url_path="/static")

    @app.get("/")
    def index():
        return send_from_directory(STATIC, "index.html")

    @app.get("/media/<path:rel>")
    def media(rel: str):
        # rel relatively pack root
        full = (pack / rel).resolve()
        if not str(full).startswith(str(pack.resolve())):
            abort(403)
        if not full.is_file():
            abort(404)
        return send_from_directory(full.parent, full.name)

    @app.get("/api/meta")
    def api_meta():
        items = list_items(pack)
        by_split = {s: sum(1 for it in items if it["split"] == s) for s in SPLITS_ORDER}
        ann_prog = {a["id"]: progress_for(pack, a["id"]) for a in ANNOTATORS}
        inter_qs = []
        for row in _INTER_Q:
            # (id, text, cond, scale)
            inter_qs.append(
                {"id": row[0], "text": row[1], "cond": row[2], "scale": row[3]}
            )
        anat_qs = []
        for row in _ANAT_Q:
            anat_qs.append({"id": row[0], "text": row[1], "scale": row[2]})
        overall_qs = [{"id": i, "text": t, "scale": "likert5"} for i, t in _OVERALL_Q]
        return jsonify(
            {
                "protocol": PROTOCOL_ID,
                "scheme": SCHEME_VERSION,
                "analysis_protocol": ANALYSIS_PROTOCOL_ID,
                "pack": str(pack),
                "n_annotators_required": N_REQUIRED,
                "annotators": ANNOTATORS,
                "guidelines": GUIDELINES,
                "decision_tree": DECISION_TREE,
                "overall_questions": overall_qs,
                "inter_questions": inter_qs,
                "anat_questions": anat_qs,
                "splits": by_split,
                "n_total": len(items),
                "progress": ann_prog,
                "encoding": {
                    "bin_1": "This picture is normal (passed)",
                    "bin_0": "There is something wrong with this item in this picture (it does not pass)",
                    "Ic_2": "Fit established",
                    "Ic_1": "There is contact but no sticking",
                    "Ic_0": "contactless",
                    "U": "Unable to judge",
                },
                "encoding_note": "Claiming construct validity: a subjective view S_inter/S_anat Confusion;AND Pass rate is appendix only.",
            }
        )

    @app.get("/api/progress/<ann_id>")
    def api_progress(ann_id: str):
        if ann_id not in {a["id"] for a in ANNOTATORS}:
            abort(404)
        return jsonify(progress_for(pack, ann_id))

    @app.get("/api/items")
    def api_items():
        ann_id = request.args.get("ann") or ""
        if ann_id not in {a["id"] for a in ANNOTATORS}:
            return jsonify({"error": "invalid ann"}), 400
        split = request.args.get("split") or None
        only_todo = request.args.get("todo") in ("1", "true", "yes")
        items = list_items(pack, split=split)
        root = _hc(pack) / "human" / ann_id
        done = set()
        if root.is_dir():
            done = {p.stem for p in root.glob("*.json") if not p.name.startswith("_")}
        out = []
        for it in items:
            row = {
                "key": it["key"],
                "sample_id": it["sample_id"],
                "model_id": it["model_id"],
                "split": it["split"],
                "intent": it.get("intent") or "unspecified",
                "n_expected": it.get("n_expected"),
                "cat": it.get("cat"),
                "prompt": it.get("prompt_zh") or it.get("prompt") or "",
                "prompt_zh": it.get("prompt_zh") or "",
                "prompt_en": it.get("prompt_en") or it.get("prompt") or "",
                "img_url": "/media/" + it["img_relpath"],
                "done": it["key"] in done,
            }
            if only_todo and row["done"]:
                continue
            out.append(row)
        return jsonify({"annotator_id": ann_id, "n": len(out), "items": out})

    @app.get("/api/item/<ann_id>/<path:key>")
    def api_item(ann_id: str, key: str):
        if ann_id not in {a["id"] for a in ANNOTATORS}:
            abort(404)
        items = {it["key"]: it for it in list_items(pack)}
        if key not in items:
            abort(404)
        it = items[key]
        saved = None
        p = ann_path(pack, ann_id, key)
        if p.is_file():
            saved = json.loads(p.read_text(encoding="utf-8"))
        return jsonify(
            {
                "item": {
                    "key": it["key"],
                    "sample_id": it["sample_id"],
                    "model_id": it["model_id"],
                    "split": it["split"],
                    "intent": it.get("intent") or "unspecified",
                    "n_expected": it.get("n_expected"),
                    "cat": it.get("cat"),
                    "prompt": it.get("prompt_zh") or it.get("prompt") or "",
                    "prompt_zh": it.get("prompt_zh") or "",
                    "prompt_en": it.get("prompt_en") or it.get("prompt") or "",
                    "img_url": "/media/" + it["img_relpath"],
                },
                "saved": saved,
            }
        )

    @app.post("/api/save")
    def api_save():
        body = request.get_json(force=True, silent=True) or {}
        ann_id = body.get("annotator_id") or ""
        if ann_id not in {a["id"] for a in ANNOTATORS}:
            return jsonify({"ok": False, "error": "invalid annotator_id"}), 400
        sid = body.get("sample_id")
        mid = body.get("model_id")
        if not sid or not mid:
            return jsonify({"ok": False, "error": "need sample_id, model_id"}), 400
        key = body.get("key") or pair_key(sid, mid)
        intent = (body.get("intent_shown") or body.get("intent") or "unspecified").strip()
        inter_in = body.get("inter") or {}
        anat_in = body.get("anat") or {}
        overall_in = body.get("overall") or {}
        try:
            inter = {
                k: normalize_inter_item(k, inter_in.get(k), intent=intent)
                for k in INTER_ITEMS
            }
            inter = apply_inter_dependencies(inter, intent)
            anat = {
                k: normalize_code(anat_in.get(k), allow_u=True, allow_null=True, max_int=1)
                for k in ANAT_ITEMS
            }
            overall = {k: normalize_overall(overall_in.get(k)) for k in OVERALL_ITEMS}
        except ValueError as e:
            return jsonify({"ok": False, "error": str(e)}), 400

        missing = []
        for k in OVERALL_ITEMS:
            if overall.get(k) is None:
                missing.append(k)
        for k in ("I0", "I1"):
            if inter.get(k) is None:
                missing.append(k)
        if intent == "required":
            if inter.get("Ic") is None:
                missing.append("Ic")
            elif int(inter["Ic"]) >= 1 and inter.get("Ir") is None:
                missing.append("Ir")
        if intent == "forbidden" and inter.get("I3") is None:
            missing.append("I3")
        for k in ANAT_ITEMS:
            if anat.get(k) is None:
                missing.append(k)
        if missing:
            return jsonify({"ok": False, "error": f"Incomplete: {', '.join(missing)}"}), 400

        scores = construct_scores(inter, anat, intent=intent)
        rec = {
            "sample_id": sid,
            "model_id": mid,
            "key": key,
            "annotator_id": ann_id,
            "protocol": PROTOCOL_ID,
            "scheme": SCHEME_VERSION,
            "intent_shown": intent,
            "overall": overall,
            "Q_inter": overall["Q_inter"],
            "Q_anat": overall["Q_anat"],
            "inter": inter,
            "anat": anat,
            **scores,
            "Inter_pass": inter_pass_from_items(inter, intent),
            "Anat_pass": anat_pass_from_items(anat),
            "seconds": body.get("seconds"),
            "notes": (body.get("notes") or "")[:500],
            "split": body.get("split"),
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "source": "annot_frontend",
        }
        out = ann_path(pack, ann_id, key)
        atomic_write_json(out, rec)
        return jsonify({"ok": True, "path": str(out), "progress": progress_for(pack, ann_id)})

    return app


def main() -> None:
    pack = pack_root(os.environ.get("PACK") or "")
    port = int(os.environ.get("PORT") or "8080")
    host = os.environ.get("HOST") or "0.0.0.0"
    app = create_app(pack)
    print(f"annot frontend {PROTOCOL_ID} pack={pack} http://{host}:{port}/", flush=True)
    app.run(host=host, port=port, threaded=True)


if __name__ == "__main__":
    main()
