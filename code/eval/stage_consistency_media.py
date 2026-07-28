#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""according to _split.json Copy the generated image to human_consistency/media/<model>/。"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

_EVAL = Path(__file__).resolve().parent
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))

from pack_io import pack_root  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack", default="")
    args = ap.parse_args()
    pack = pack_root(args.pack) if args.pack else pack_root()
    hc = pack / "judgments" / "human_consistency"
    split = json.loads((hc / "_split.json").read_text(encoding="utf-8"))
    n_ok = n_miss = 0
    for name, rows in (split.get("splits") or {}).items():
        for r in rows:
            mid, sid = r["model_id"], r["sample_id"]
            src = None
            for rel in (
                r.get("img_relpath"),
                f"outputs/{mid}/{sid}.png",
                f"outputs/{mid}/{sid}.jpg",
                f"outputs/{mid}/{sid}.jpeg",
                f"outputs/{mid}/{sid}.webp",
            ):
                if not rel:
                    continue
                p = pack / rel
                if p.is_file():
                    src = p
                    break
            if src is None:
                n_miss += 1
                continue
            dst = hc / "media" / mid / f"{sid}{src.suffix.lower()}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.is_file() or dst.stat().st_size != src.stat().st_size:
                shutil.copy2(src, dst)
            # renew split Inner relative paths are convenient for the front end
            r["img_relpath"] = f"judgments/human_consistency/media/{mid}/{dst.name}"
            n_ok += 1
    (hc / "_split.json").write_text(
        json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"staged": n_ok, "missing": n_miss, "media": str(hc / "media")}))


if __name__ == "__main__":
    main()
