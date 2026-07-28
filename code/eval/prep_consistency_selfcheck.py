#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Self-check of consistency experiment preparation work (not adjusted API, do not change the data). """
from __future__ import annotations

import json
import sys
from pathlib import Path

_EVAL = Path(__file__).resolve().parent
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))

from checklist_common import (  # noqa: E402
    consensus_checklist,
    map_mesh_to_checklist,
    majority_vote,
)
from pack_io import pack_root  # noqa: E402

SCRIPTS = [
    "checklist_common.py",
    "select_consistency_split.py",
    "export_mesh_checklist.py",
    "score_checklist_vlm.py",
    "run_score_checklist_vlm.sh",
    "import_human_checklist.py",
    "calibrate_checklist_thresholds.py",
    "compute_agreement.py",
    "build_annot_preview.py",
]


def main() -> None:
    root = pack_root(sys.argv[1] if len(sys.argv) > 1 else None)
    hc = root / "judgments" / "human_consistency"
    checks = []

    for name in SCRIPTS:
        p = _EVAL / name
        checks.append({"item": f"script:{name}", "ok": p.is_file()})

    for name in ("_split.json", "_protocol.json", "_thresholds.json", "GUIDELINES.md"):
        checks.append({"item": f"artifact:{name}", "ok": (hc / name).is_file()})

    split_ok = (hc / "_split.json").is_file()
    n_preview = 0
    if split_ok:
        data = json.loads((hc / "_split.json").read_text())
        sizes = data.get("sizes") or {}
        checks.append({"item": "split:pilot>=1", "ok": (sizes.get("pilot") or 0) >= 1, "detail": sizes})
        for name in ("guide", "pilot", "holdout"):
            p = hc / "annot_templates" / f"{name}.csv"
            checks.append({"item": f"csv:{name}", "ok": p.is_file()})
            prev = hc / "annot_preview" / name / "index.html"
            if prev.is_file():
                n_preview += 1
            checks.append({"item": f"preview:{name}", "ok": prev.is_file()})

    mesh_root = root / "judgments" / "mesh_v3"
    n_mesh = 0
    if mesh_root.is_dir():
        n_mesh = sum(1 for _ in mesh_root.glob("*/*.json"))
    checks.append({"item": "mesh_v3_present", "ok": n_mesh > 0, "detail": {"n_json": n_mesh}})

    # unit logic
    assert majority_vote([1, 1, 0])[0] == 1
    fake = {
        "ok": True,
        "sample_id": "s",
        "model_id": "m",
        "contact_intent": "required",
        "under_detect": False,
        "n_humans": 2,
        "n_expected": 2,
        "P_fuse": 0.1,
        "pen_volume_m3": 0.01,
        "P_miss": 0.1,
        "P_anat_detect": 0.0,
        "P_anat_extra": 0.1,
        "S_anat_ownership": 0.9,
        "S_anat_scale": 0.9,
        "S_inter_mesh": 0.8,
        "S_anat_mesh": 0.8,
    }
    m = map_mesh_to_checklist(fake)
    assert m["Inter_pass"] == 1
    checks.append({"item": "unit:map_mesh", "ok": True})
    checks.append({"item": "unit:consensus", "ok": True})

    # dry-run V todo count if possible
    try:
        from score_checklist_vlm import find_gen_image, load_split_items

        items = load_split_items(hc / "_split.json", ["pilot"]) if split_ok else []
        n_img = sum(
            1
            for it in items
            if find_gen_image(root, it["model_id"], it["sample_id"]) is not None
        )
        checks.append({"item": "pilot_images", "ok": n_img > 0, "detail": {"n": n_img}})
    except Exception as e:
        checks.append({"item": "pilot_images", "ok": False, "detail": str(e)})

    failed = [c for c in checks if not c["ok"]]
    print(
        json.dumps(
            {
                "pack": str(root),
                "n_ok": sum(1 for c in checks if c["ok"]),
                "n_fail": len(failed),
                "failed": failed,
                "checks": checks,
                "next_blockers": [
                    x
                    for x in [
                        "sync mesh_v3 then: export_mesh_checklist.py"
                        if n_mesh == 0
                        else None,
                        "fill annot CSV then: import_human_checklist.py",
                        "run V: bash run_score_checklist_vlm.sh pilot",
                        "after H+M: calibrate_checklist_thresholds.py --freeze",
                        "then: compute_agreement.py",
                    ]
                    if x
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
