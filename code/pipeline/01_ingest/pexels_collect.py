#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 1: CC0 Material station video collection —— Pexels(Data source no.②layer).

exist SG Development machine runs (GPU host Overseas network black hole,Pexels Direct connection has been verified and available):
  search   according to configs/cc0_search_keywords.yaml Keyword-by-keyword tone Pexels search API,
           Output Candidate List candidates.json(Includes attribution information, No download)
  download Download by shortlist mp4 arrive <out>/<interaction_type>/<id>.mp4,
           Resume running from breakpoint (already exists and is skipped if it is not empty), support --types/--max-per-type Moving in batches

key: export MPIE_PEXELS_KEY=...(live ~/.mpie_env, Storage is strictly prohibited)
speed limit: Pexels Free file 200 req/h、2 Ten thousand/moon; Search per keyword by default 2 Page(80/Page)Run through all types
      about 90 requests, Within one hour quota; 429 automatically waits for retry.
license: Pexels License(Free for commercial use, No resale as is); Each video record photographer Belong,
      release benchmark pixels still have to pass docs/03_licensing Clause-by-clause verification.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests
import yaml

API = "https://api.pexels.com/videos/search"


def _get(url, key, params=None, stream=False):
    for attempt in range(5):
        r = requests.get(url, headers={"Authorization": key}, params=params,
                         stream=stream, timeout=60)
        if r.status_code == 429:  # speed limit: Wait a minute and try again
            wait = int(r.headers.get("Retry-After", 60))
            print(f"  429 rate-limited, sleep {wait}s", flush=True)
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r
    raise RuntimeError(f"giving up after retries: {url}")


def pick_file(video_files):
    """Choose the download file: ≤1080p The highest definition mp4(Balance point between storage and subsequent frame extraction quality)。"""
    mp4s = [f for f in video_files
            if f.get("file_type") == "video/mp4" and f.get("height")]
    ok = [f for f in mp4s if f["height"] <= 1080]
    if ok:
        return max(ok, key=lambda f: f["height"])
    return min(mp4s, key=lambda f: f["height"]) if mp4s else None


def cmd_search(args, key):
    kw_map = yaml.safe_load(Path(args.keywords).read_text())
    seen, cands = set(), []
    for itype, kws in kw_map.items():
        n_before = len(cands)
        for kw in kws:
            for page in range(1, args.pages + 1):
                r = _get(API, key, params={"query": kw, "per_page": 80,
                                           "page": page})
                videos = r.json().get("videos", [])
                for v in videos:
                    if v["id"] in seen:
                        continue
                    seen.add(v["id"])
                    if not (args.min_dur <= v.get("duration", 0) <= args.max_dur):
                        continue
                    if v.get("height", 0) < args.min_height:
                        continue
                    vf = pick_file(v.get("video_files", []))
                    if not vf:
                        continue
                    cands.append({
                        "id": v["id"], "interaction_type": itype, "keyword": kw,
                        "duration": v["duration"], "width": v["width"],
                        "height": v["height"], "page_url": v["url"],
                        "photographer": v.get("user", {}).get("name", ""),
                        "photographer_url": v.get("user", {}).get("url", ""),
                        "dl_url": vf["link"], "dl_height": vf["height"],
                        "license": "Pexels License", "source": "pexels",
                    })
                time.sleep(1.2)
                if len(videos) < 80:  # This keyword has reached the end, Don't turn to the next page
                    break
        print(f"{itype:22s} +{len(cands) - n_before}", flush=True)
    out = Path(args.manifest)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cands, ensure_ascii=False, indent=1))
    print(f"total {len(cands)} candidates -> {out}")


def cmd_download(args, key):
    cands = json.loads(Path(args.manifest).read_text())
    types = set(args.types.split(",")) if args.types else None
    per_type = {}
    n_ok = n_skip = 0
    for c in cands:
        it = c["interaction_type"]
        if types and it not in types:
            continue
        if per_type.get(it, 0) >= args.max_per_type:
            continue
        d = Path(args.out) / it
        d.mkdir(parents=True, exist_ok=True)
        fp = d / f"pexels_{c['id']}.mp4"
        if fp.exists() and fp.stat().st_size > 0:
            per_type[it] = per_type.get(it, 0) + 1
            n_skip += 1
            continue
        try:
            r = _get(c["dl_url"], key, stream=True)
            tmp = fp.with_suffix(".part")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
            clen = int(r.headers.get("Content-Length", 0))
            if clen and tmp.stat().st_size != clen:  # Cut and discard, Next round again
                tmp.unlink()
                print(f"  truncated, dropped: {c['id']}", flush=True)
                continue
            tmp.rename(fp)
            per_type[it] = per_type.get(it, 0) + 1
            n_ok += 1
            print(f"  {it}/{fp.name} ({fp.stat().st_size >> 20}MB) "
                  f"[{per_type[it]}/{args.max_per_type}]", flush=True)
        except Exception as e:  # Single failure will not interrupt the batch
            print(f"  FAIL {c['id']}: {e}", flush=True)
    print(f"done: {n_ok} downloaded, {n_skip} already present")
    print(json.dumps(per_type, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search")
    s.add_argument("--keywords", default=str(Path(__file__).resolve()
                   .parents[3] / "configs/cc0_search_keywords.yaml"))
    s.add_argument("--manifest", required=True)
    s.add_argument("--pages", type=int, default=2)
    s.add_argument("--min-dur", type=int, default=4)
    s.add_argument("--max-dur", type=int, default=60)
    s.add_argument("--min-height", type=int, default=720)
    d = sub.add_parser("download")
    d.add_argument("--manifest", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--types", default="", help="Comma separated, null=all")
    d.add_argument("--max-per-type", type=int, default=40)
    args = ap.parse_args()
    key = os.environ.get("MPIE_PEXELS_KEY", "")
    if not key:
        sys.exit("MPIE_PEXELS_KEY not set (source ~/.mpie_env)")
    (cmd_search if args.cmd == "search" else cmd_download)(args, key)


if __name__ == "__main__":
    main()
