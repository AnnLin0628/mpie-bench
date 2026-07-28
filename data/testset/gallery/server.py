#!/usr/bin/env python3
"""MPI E testset image gallery — waterfall viewer on :8080."""

from __future__ import print_function

import json
import mimetypes
import os
import re
import socket
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

try:
    from urllib.parse import parse_qs, unquote, urlparse
except ImportError:
    from urlparse import parse_qs, unquote, urlparse  # type: ignore

try:
    from socketserver import ThreadingMixIn
except ImportError:
    from SocketServer import ThreadingMixIn  # type: ignore


ROOT = Path(__file__).resolve().parent
OUTPUTS = Path(os.environ.get("MPI_OUTPUTS", str(ROOT.parent / "outputs"))).resolve()
PORT = int(os.environ.get("PORT", "8080"))
IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}

NAT_RE = re.compile(r"(\d+)")


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def natural_key(name):
    parts = NAT_RE.split(name)
    key = []
    for t in parts:
        if t.isdigit():
            key.append(int(t))
        else:
            key.append(t.lower())
    return key


def list_models():
    models = []
    if not OUTPUTS.is_dir():
        return models
    for d in sorted(OUTPUTS.iterdir()):
        if not d.is_dir() or d.name.startswith(".") or d.name.startswith("_"):
            continue
        count = 0
        for p in d.iterdir():
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                count += 1
        if count == 0:
            continue
        models.append({"name": d.name, "count": count})
    return models


def list_images(model, category=None):
    model_dir = (OUTPUTS / model).resolve()
    if not str(model_dir).startswith(str(OUTPUTS)) or not model_dir.is_dir():
        return []
    items = []
    for p in model_dir.iterdir():
        if not p.is_file() or p.suffix.lower() not in IMG_EXTS:
            continue
        if "__" in p.name:
            cat = p.name.split("__", 1)[0]
        else:
            cat = "other"
        if category and cat != category:
            continue
        items.append(
            {
                "name": p.name,
                "category": cat,
                "url": "/img/{0}/{1}".format(model, p.name),
                "mtime": p.stat().st_mtime,
            }
        )
    items.sort(key=lambda x: natural_key(x["name"]))
    return items


def list_categories(model):
    cats = {}
    for img in list_images(model):
        c = img["category"]
        cats[c] = cats.get(c, 0) + 1
    return [{"name": k, "count": v} for k, v in sorted(cats.items())]


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        # directory= only exists in 3.7+; chdir fallback for 3.6
        try:
            SimpleHTTPRequestHandler.__init__(self, *args, directory=str(ROOT), **kwargs)
        except TypeError:
            SimpleHTTPRequestHandler.__init__(self, *args, **kwargs)

    def log_message(self, fmt, *args):
        msg = fmt % args
        if msg.startswith('"GET /img/') or msg.startswith('"GET /favicon'):
            return
        SimpleHTTPRequestHandler.log_message(self, fmt, *args)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        qs = parse_qs(parsed.query)

        if path == "/api/models":
            return self._json(list_models())

        if path == "/api/categories":
            model = qs.get("model", [""])[0]
            if not model:
                return self._json({"error": "model required"}, 400)
            return self._json(list_categories(model))

        if path == "/api/images":
            model = qs.get("model", [""])[0]
            if not model:
                return self._json({"error": "model required"}, 400)
            category = qs.get("category", [None])[0] or None
            try:
                page = max(1, int(qs.get("page", ["1"])[0]))
                page_size = min(200, max(1, int(qs.get("page_size", ["60"])[0])))
            except ValueError:
                return self._json({"error": "bad page"}, 400)

            all_imgs = list_images(model, category)
            total = len(all_imgs)
            start = (page - 1) * page_size
            chunk = all_imgs[start : start + page_size]
            return self._json(
                {
                    "model": model,
                    "category": category,
                    "total": total,
                    "page": page,
                    "page_size": page_size,
                    "has_more": start + page_size < total,
                    "images": chunk,
                }
            )

        if path.startswith("/img/"):
            return self._serve_image(path[len("/img/") :])

        if path in ("/", "/index.html"):
            return self._serve_file(ROOT / "index.html", "text/html; charset=utf-8")

        return SimpleHTTPRequestHandler.do_GET(self)

    def _json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, fp, content_type):
        if not fp.is_file():
            self.send_error(404)
            return
        data = fp.read_bytes() if hasattr(fp, "read_bytes") else open(str(fp), "rb").read()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_image(self, rel):
        parts = rel.split("/", 1)
        if len(parts) != 2:
            self.send_error(400)
            return
        model, name = parts
        if "/" in name or "\\" in name or name in (".", "..") or model in (".", ".."):
            self.send_error(400)
            return
        fp = (OUTPUTS / model / name).resolve()
        if not str(fp).startswith(str(OUTPUTS)) or not fp.is_file():
            self.send_error(404)
            return
        ctype = mimetypes.guess_type(str(fp))[0] or "application/octet-stream"
        data = fp.read_bytes() if hasattr(fp, "read_bytes") else open(str(fp), "rb").read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "public, max-age=3600")
        self.end_headers()
        self.wfile.write(data)


def main():
    # pathlib.Path.read_bytes exists since 3.5; iterdir ok
    os.chdir(str(ROOT))
    print("Outputs : {0}".format(OUTPUTS))
    print("Gallery : http://0.0.0.0:{0}".format(PORT))
    for m in list_models():
        print("  - {0}: {1} images".format(m["name"], m["count"]))
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
        server.server_close()


if __name__ == "__main__":
    main()
