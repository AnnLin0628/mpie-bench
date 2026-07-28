#!/usr/bin/env python3
"""Closed source graph client（URL / Key They are always injected by environment variables, and the warehouse does not have built-in keys or intranet addresses):

- gpt-image-2 → OpenAI Compatible with gateway /images/edits
- gemini-3-pro-image → Gemini Native v1beta(Can be rolled back OpenAI chat）
- seedream-5-pro → Manufacturer image API(volcano ARK / BytePlus SEA wait,URL/Key/model can be covered)
"""
from __future__ import annotations

import base64
import os
import re
import time
from pathlib import Path

import requests

# The public warehouse does not hardcode the intranet gateway; please use environment variables to inject it. URL / Key / model。
DEFAULT_GPT_GATEWAY = ""
DEFAULT_GEMINI_V1BETA = ""
DEFAULT_SEEDREAM_URL = ""
DEFAULT_SEEDREAM_MODEL = "doubao-seedream-5-0-pro"
DEFAULT_SEEDREAM = "https://ark.ap-southeast.bytepluses.com/api/v3/images/generations"
_TRANSIENT_HINTS = (
    "no available channel",
    "model_not_found",
    "overloaded",
    "503",
    "502",
    "429",
    "empty picture",
    "inlinedata",
    "timed out",
    "timeout",
    "read timed out",
)


def _normalize_gateway_base(base: str) -> str:
    base = base.strip().rstrip("/")
    for suffix in (
        "/chat/completions",
        "/images/generations",
        "/images/edits",
        "/models",
    ):
        if base.endswith(suffix):
            base = base[: -len(suffix)].rstrip("/")
    return base


def gateway_base() -> str:
    """OpenAI Compatible gateway root path, e.g. https://example.com/v1。"""
    base = (
        os.environ.get("AI_GATEWAY_URL")
        or os.environ.get("MPIE_GATEWAY_URL")
        or DEFAULT_GPT_GATEWAY
    )
    base = _normalize_gateway_base(base)
    if not base:
        raise RuntimeError(
            "Missing gateway URL:please export AI_GATEWAY_URL=https://<host>/v1"
        )
    return base


def gpt_gateway_base() -> str:
    """gpt-image-2 Dedicated gateway (GPT_IMAGE_GATEWAY_URL, otherwise fall back gateway_base）。"""
    base = (
        os.environ.get("GPT_IMAGE_GATEWAY_URL")
        or os.environ.get("AI_GATEWAY_INTERNAL_URL")
        or DEFAULT_GPT_GATEWAY
    )
    return _normalize_gateway_base(base) or gateway_base()


def gemini_v1beta_base() -> str:
    """Gemini Native protocol root path (requires export GEMINI_V1BETA_URL）。"""
    base = (
        os.environ.get("GEMINI_V1BETA_URL")
        or os.environ.get("AI_GATEWAY_INTERNAL_V1BETA_URL")
        or DEFAULT_GEMINI_V1BETA
    )
    base = _normalize_gateway_base(base)
    if not base:
        raise RuntimeError(
            "Lack Gemini v1beta URL:please export GEMINI_V1BETA_URL=https://<host>/v1beta"
        )
    return base


def gemini_model() -> str:
    return os.environ.get("GEMINI_IMAGE_MODEL") or "gemini-3-pro-image"


def gateway_key() -> str:
    key = (
        os.environ.get("GPT_IMAGE_KEY")
        or os.environ.get("AI_GATEWAY_KEY")
        or os.environ.get("AI_GATEWAY_KEY")
        or os.environ.get("ARK_API_KEY")
    )
    if not key:
        raise RuntimeError(
            "Missing gateway Key:please export AI_GATEWAY_KEY / GPT_IMAGE_KEY / ARK_API_KEY"
        )
    return key


def _headers_json(key: str | None = None) -> dict:
    return {
        "Authorization": f"Bearer {key or gateway_key()}",
        "Content-Type": "application/json",
    }


def seedream_url() -> str:
    url = (
        os.environ.get("SEEDREAM_URL")
        or os.environ.get("SEEDREAM_URL")
        or os.environ.get("SEEDREAM5_LITE_URL")
        or DEFAULT_SEEDREAM_URL
        or DEFAULT_SEEDREAM
    )
    if not url:
        raise RuntimeError(
            "Lack Seedream URL:please export SEEDREAM_URL or SEEDREAM_URL"
        )
    return url


def seedream_key() -> str:
    """according to endpoint select key:volcano CN→ARK_API_KEY;other→SEEDREAM_KEY / AI_GATEWAY_KEY wait. """
    explicit = os.environ.get("SEEDREAM_KEY")
    if explicit:
        return explicit
    url = seedream_url()
    if "volces.com" in url or "cn-beijing" in url:
        key = (
            os.environ.get("ARK_API_KEY")
            or os.environ.get("AI_GATEWAY_KEY")
            or os.environ.get("AI_GATEWAY_KEY")
        )
    else:
        key = (
            os.environ.get("SEEDREAM_KEY")
            or os.environ.get("SEEDREAM5_LITE_KEY")
            or os.environ.get("ARK_API_KEY")
            or os.environ.get("AI_GATEWAY_KEY")
            or os.environ.get("AI_GATEWAY_KEY")
        )
    if not key:
        raise RuntimeError(
            "Lack Seedream Key:please export SEEDREAM_KEY / AI_GATEWAY_KEY / ARK_API_KEY / SEEDREAM_KEY"
        )
    return key


def seedream_model() -> str:
    return (
        os.environ.get("SEEDREAM_MODEL")
        or os.environ.get("SEEDREAM_MODEL")
        or os.environ.get("SEEDREAM5_MODEL")
        or DEFAULT_SEEDREAM_MODEL
    )


# Compatible with old function names
def seedream_sea_key() -> str:
    return seedream_key()


def seedream_sea_url() -> str:
    return seedream_url()


def seedream_sea_model() -> str:
    return seedream_model()


def _decode_image_payload(item: dict) -> bytes:
    if item.get("b64_json"):
        return base64.b64decode(item["b64_json"])
    url = item.get("url")
    if url:
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        return r.content
    raise RuntimeError(f"no image in response item: {str(item)[:200]}")


def _is_transient(exc: BaseException) -> bool:
    s = str(exc).lower()
    return any(h in s for h in _TRANSIENT_HINTS)


def _with_retries(fn, *, retries: int = 3, backoff: float = 4.0):
    last = None
    for i in range(retries + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last = e
            if i >= retries or not _is_transient(e):
                raise
            time.sleep(backoff * (i + 1))
    raise last  # pragma: no cover


def gpt_image2_edit(
    prompt: str,
    ref_paths: list[Path],
    *,
    size: str = "1024x1024",
    quality: str = "low",
    timeout: int = 300,
) -> tuple[bytes, dict]:
    """POST /images/edits，multipart image[]（≤8 local reference map). """
    if not ref_paths:
        raise ValueError("gpt-image-2 edit Requires at least 1 Reference picture")

    def _once() -> tuple[bytes, dict]:
        files = []
        for i, p in enumerate(ref_paths[:8]):
            raw = Path(p).read_bytes()
            mime = "image/jpeg" if raw[:3] == b"\xff\xd8\xff" else "image/png"
            files.append(("image[]", (f"ref{i}.jpg", raw, mime)))
        data = {
            "model": "gpt-image-2",
            "prompt": prompt[:4000],
            "size": size,
            "quality": quality,
            "n": "1",
        }
        url = f"{gpt_gateway_base()}/images/edits"
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {gateway_key()}"},
            data=data,
            files=files,
            timeout=timeout,
        )
        meta = {
            "api_model": "gpt-image-2",
            "endpoint": url,
            "http_status": resp.status_code,
            "size": size,
            "quality": quality,
            "n_refs_sent": len(files),
        }
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:400]}")
        body = resp.json()
        if body.get("error"):
            raise RuntimeError(f"upstream error: {str(body['error'])[:400]}")
        img = _decode_image_payload(body["data"][0])
        usage = body.get("usage") or {}
        if usage:
            meta["usage"] = usage
        return img, meta

    return _with_retries(_once)


_DATA_URI_RE = re.compile(
    r"data:image/[A-Za-z0-9.+-]+;base64,([A-Za-z0-9+/=]+)"
)


def _extract_gemini_v1beta_image(body: dict) -> bytes | None:
    for cand in body.get("candidates") or []:
        content = (cand.get("content") or {}) if isinstance(cand, dict) else {}
        for part in content.get("parts") or []:
            if not isinstance(part, dict):
                continue
            inline = part.get("inlineData") or part.get("inline_data") or {}
            data = inline.get("data")
            if data:
                return base64.b64decode(data)
            text = part.get("text") or ""
            m = _DATA_URI_RE.search(text)
            if m:
                return base64.b64decode(m.group(1))
    return None


def gemini_3_pro_image(
    prompt: str,
    ref_paths: list[Path],
    *,
    timeout: int = 300,
    max_tokens: int = 4096,
) -> tuple[bytes, dict]:
    """Default Singapore intranet Gemini Native v1beta；GEMINI_PROTOCOL=chat rollback OpenAI chat。"""
    protocol = (os.environ.get("GEMINI_PROTOCOL") or "v1beta").strip().lower()
    if protocol in ("chat", "openai"):
        return _gemini_via_chat(
            prompt, ref_paths, timeout=timeout, max_tokens=max_tokens
        )
    return _gemini_via_v1beta(prompt, ref_paths, timeout=timeout)


def _gemini_via_v1beta(
    prompt: str,
    ref_paths: list[Path],
    *,
    timeout: int = 300,
) -> tuple[bytes, dict]:
    model = gemini_model()

    def _once() -> tuple[bytes, dict]:
        parts: list[dict] = []
        for p in ref_paths[:8]:
            raw = Path(p).read_bytes()
            mime = "image/jpeg" if raw[:3] == b"\xff\xd8\xff" else "image/png"
            parts.append(
                {
                    "inline_data": {
                        "mime_type": mime,
                        "data": base64.b64encode(raw).decode("ascii"),
                    }
                }
            )
        parts.append({"text": prompt})
        url = f"{gemini_v1beta_base()}/models/{model}:generateContent"
        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]},
        }
        resp = requests.post(
            url, headers=_headers_json(), json=payload, timeout=timeout
        )
        meta = {
            "api_model": model,
            "alias": "nano-banana-pro",
            "endpoint": url,
            "protocol": "v1beta",
            "http_status": resp.status_code,
            "n_refs_sent": min(len(ref_paths), 8),
        }
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:400]}")
        body = resp.json()
        if body.get("error"):
            raise RuntimeError(f"upstream error: {str(body['error'])[:400]}")
        img_bytes = _extract_gemini_v1beta_image(body)
        if not img_bytes:
            raise RuntimeError(
                "gemini v1beta Returns an empty image(candidates none inlineData); can retry"
            )
        usage = body.get("usageMetadata") or body.get("usage") or {}
        if usage:
            meta["usage"] = usage
        return img_bytes, meta

    return _with_retries(_once)


def _gemini_via_chat(
    prompt: str,
    ref_paths: list[Path],
    *,
    timeout: int = 300,
    max_tokens: int = 4096,
) -> tuple[bytes, dict]:
    """POST /chat/completions; Pictured with markdown data URI embedded in message.content。"""
    model = gemini_model()

    def _once() -> tuple[bytes, dict]:
        content: list[dict] = []
        for p in ref_paths[:8]:
            raw = Path(p).read_bytes()
            mime = "image/jpeg" if raw[:3] == b"\xff\xd8\xff" else "image/png"
            b64 = base64.b64encode(raw).decode("ascii")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                }
            )
        content.append({"type": "text", "text": prompt})
        chat_base = (
            os.environ.get("GEMINI_CHAT_GATEWAY_URL")
            or os.environ.get("AI_GATEWAY_INTERNAL_URL")
            or gpt_gateway_base()
        )
        url = f"{_normalize_gateway_base(chat_base)}/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "modalities": ["text", "image"],
            "max_tokens": max_tokens,
        }
        resp = requests.post(
            url, headers=_headers_json(), json=payload, timeout=timeout
        )
        meta = {
            "api_model": model,
            "alias": "nano-banana-pro",
            "endpoint": url,
            "protocol": "chat",
            "http_status": resp.status_code,
            "n_refs_sent": min(len(ref_paths), 8),
        }
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:400]}")
        body = resp.json()
        if body.get("error"):
            raise RuntimeError(f"upstream error: {str(body['error'])[:400]}")
        choice = (body.get("choices") or [{}])[0].get("message") or {}
        img_bytes = _extract_gemini_image(choice)
        if not img_bytes:
            raise RuntimeError(
                "gemini-3-pro-image Returns an empty image(content none data URI); can retry"
            )
        usage = body.get("usage") or {}
        if usage:
            meta["usage"] = usage
        return img_bytes, meta

    return _with_retries(_once)


def _extract_gemini_image(message: dict) -> bytes | None:
    # form1：message.images[0].image_url.url
    for img in message.get("images") or []:
        if isinstance(img, dict):
            u = (img.get("image_url") or {}).get("url") or ""
            m = _DATA_URI_RE.search(u)
            if m:
                return base64.b64decode(m.group(1))
            if u.startswith("http"):
                r = requests.get(u, timeout=120)
                r.raise_for_status()
                return r.content

    content = message.get("content")
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "image_url":
                u = (part.get("image_url") or {}).get("url") or ""
                m = _DATA_URI_RE.search(u)
                if m:
                    return base64.b64decode(m.group(1))
            if part.get("type") == "text":
                m = _DATA_URI_RE.search(part.get("text") or "")
                if m:
                    return base64.b64decode(m.group(1))
    elif isinstance(content, str) and content.strip():
        m = _DATA_URI_RE.search(content)
        if m:
            return base64.b64decode(m.group(1))
    return None


def _path_to_data_uri(p: Path) -> str:
    raw = Path(p).read_bytes()
    mime = "image/jpeg" if raw[:3] == b"\xff\xd8\xff" else "image/png"
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def seedream_5_pro(
    prompt: str,
    ref_paths: list[Path],
    *,
    size: str = "2K",
    timeout: int = 180,
    max_refs: int = 4,
) -> tuple[bytes, dict]:
    """Seedream Pictures give rise to pictures. Default Beijing intranet Pro;Available SEEDREAM_URL/KEY/MODEL cover. """

    def _once() -> tuple[bytes, dict]:
        model = seedream_model()
        payload: dict = {
            "model": model,
            "prompt": prompt[:4000],
            "size": size,
            "response_format": "url",
            "output_format": "png",
            "watermark": False,
        }
        # Pro pass sequential_image_generation meeting 400；lite Need to disable
        if "pro" not in model.lower():
            payload["sequential_image_generation"] = "disabled"
            payload["stream"] = False
        if ref_paths:
            uris = [_path_to_data_uri(p) for p in ref_paths[:max_refs]]
            payload["image"] = uris if len(uris) > 1 else uris[0]
        url = seedream_url()
        key = seedream_key()
        resp = requests.post(
            url,
            headers=_headers_json(key),
            json=payload,
            timeout=timeout,
        )
        region = (
            "cn-beijing"
            if ("cn-beijing" in url or "volces.com" in url)
            else ("ap-southeast" if "ap-southeast" in url else "custom")
        )
        meta = {
            "api_model": model,
            "endpoint": url,
            "http_status": resp.status_code,
            "size": size,
            "n_refs_sent": min(len(ref_paths), max_refs) if ref_paths else 0,
            "region": region,
        }
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:400]}")
        body = resp.json()
        if body.get("error"):
            raise RuntimeError(f"upstream error: {str(body['error'])[:400]}")
        item = (body.get("data") or [{}])[0]
        img = _decode_image_payload(item)
        if body.get("usage"):
            meta["usage"] = body["usage"]
        if item.get("url"):
            meta["result_url"] = str(item["url"])[:200]
        return img, meta

    return _with_retries(_once)
