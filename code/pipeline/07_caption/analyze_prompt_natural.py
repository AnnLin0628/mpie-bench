#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test-set prompt natural distribution (no board cat dependency).

Primary axis from layer text: action / venue / activity.
Board cat appendix only — CHI3D/Harmony4D mapped by true action into natural buckets.

Usage:
  ~/miniconda3/bin/python3.11 analyze_prompt_natural.py
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

HOME = Path.home()
PROMPT_DIR = HOME / "mpie_bench/data/manifests/prompts_full"
DEFAULT_OUT = HOME / "mpie_bench/data/manifests/prompt_distribution"

PALETTE = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
    "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
    "#86BCB6", "#D37295", "#8CD17D", "#B6992D", "#499894",
    "#D4A6C8", "#FABFD2", "#A0CBE8",
]

STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "with", "from",
    "by", "for", "as", "is", "are", "their", "her", "his", "him", "she", "he",
    "they", "them", "who", "while", "both", "each", "other", "one", "two",
    "left", "right", "side", "frame", "woman", "man", "person", "people",
    "wearing", "looking", "standing", "against", "background", "camera",
    "slightly", "closely", "together", "around", "over", "under", "into",
    "toward", "towards", "across", "near", "between", "through", "onto",
    "young", "haired", "blonde", "bearded", "girl", "boy", "women", "men",
    "shirt", "dress", "black", "white", "red", "blue", "stands", "sits",
}


def setup_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "savefig.dpi": 300,
        "savefig.bbox": "tight", "pdf.fonttype": 42,
    })


def clean(text: str) -> str:
    t = (text or "").lower()
    t = re.sub(r"\([^)]*\)", " ", t)
    t = re.sub(r"[^a-z\s\-]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def load_rows(prompt_dir: Path):
    rows = []
    for f in sorted(prompt_dir.glob("prompts_*.json")):
        for item in json.loads(f.read_text()):
            vlm = item.get("vlm") or {}
            if "error" in vlm:
                continue
            summary = vlm.get("scene_summary") or ""
            board = item["cat"]
            source = {"chi3d": "CHI3D", "harmony4d": "Harmony4D"}.get(board, "CC0 / other")
            for t in vlm.get("targets") or []:
                layers = t.get("layers") or {}
                subjects = layers.get("subjects") or []
                byst = layers.get("bystanders")
                byst_s = "" if byst is None else str(byst).strip()
                has_byst = bool(byst_s) and byst_s.lower() not in ("none", "null", "n/a", "-")
                inter = (layers.get("interaction") or "").strip()
                setting = (layers.get("setting") or "").strip()
                camera = (layers.get("camera") or "").strip()
                rows.append({
                    "board_cat": board, "source": source,
                    "anchor": item["anchor"], "target_id": t.get("target"),
                    "interaction": inter, "setting": setting, "camera": camera,
                    "scene_summary": summary,
                    "n_subjects": len(subjects) if isinstance(subjects, list) else 0,
                    "has_bystander": has_byst,
                    "action_text": clean(inter),
                    "activity_text": clean(summary + " " + inter),
                })
    return rows


# Order matters: more specific phrases first
# 14 paper interaction types (natural labels from layers, not board cat)
ACTION_RULES = [
    ("handshake", ("handshake", "shaking hands", "shake hands", "shake right hands",
                   "shake left hands", "shakes hands", "shaking right hands",
                   "shaking left hands", "are shaking")),
    ("high-five", ("high-five", "high five", "highfive", "high-fiving", "slap hands",
                   "palm against palm")),
    ("hand-hold", ("holding hands", "hold hands", "hand in hand", "interlocked", "clasping hands",
                   "holds her hand", "holds his hand", "holding her hand", "holding his hand",
                   "joined hands", "holding the child's hands", "holds the toddler",
                   "holding onto the", "tug-of-war", "tug of war")),
    ("piggyback", ("piggyback", "piggy-back", "on his back", "on her back", "on the back",
                   "carried on the back", "rides on", "sits on the shoulders", "sit on the shoulders",
                   "on the shoulders of", "on their shoulders", "shoulder ride", "handstand")),
    ("carry / lift", ("carrying", "carries", "lifting", "lifted", "lifts ", "bridal carry",
                      "in his arms", "in her arms", "hoist", "hoisting", "picked up", "cradling",
                      "raises her off", "raises him off", "holds her up", "holds him up",
                      "hold r2 together", "holds r2 together", "support her weight",
                      "support his weight", "holding her ankles", "holding his ankles",
                      "helps support", "help support")),
    ("hug / embrace", ("hug", "embrace", "embracing", "cuddling", "cuddle", "wrapped in each other",
                       "arms wrapped around")),
    ("kiss", ("kiss", "kissing")),
    ("arm-around / shoulder", (
        "arm around", "arms around", "around the shoulder", "around her shoulder", "around his shoulder",
        "leaning her head", "leaning his head", "resting her head", "resting his head",
        "head on her shoulder", "head on his shoulder", "chin on", "temple",
        "hands on her shoulders", "hands on his shoulders", "resting both hands on",
        "around the waist", "around her waist", "around his waist", "wrapping her hands",
        "wrapping his hands", "hands gently around", "arm linked", "arms linked",
        "linked through", "arm-in-arm", "arm in arm",
        "arm on", "leans his arm", "leans her arm", "hand gently on", "hand on the right shoulder",
        "hand on the left shoulder", "hand on her shoulder", "hand on his shoulder",
        "resting her right hand", "resting his right hand", "leans back against",
    )),
    ("dance hold", ("dance", "ballroom", "waltz", "partner hold", "leading her", "leading him",
                    "dancing", "dance pose", "dip her", "dip him", "ballerina")),
    ("grapple / wrestle", ("grappl", "wrestl", "clinch", "takedown", "pinning", "pinned",
                           "grabbing", "grab ", "locks ", "holds down", "body lock",
                           "restrain", "restraining")),
    ("strike / combat", ("punch", "kick", "strike", "fighting", "combat", "hit ", "hitting",
                         "pushing", "push ", "blocking a", "throws a", "boxers face",
                         "throws a punch", "swinging at")),
    ("reach / touch", ("touch", "touches", "touching", "extends", "reaching",
                       "places her hand", "places his hand", "places both hands",
                       "hand on the", "hand on her", "hand on his", "resting a hand",
                       "lays a hand", "feeding", "cradles the face", "apply lipstick",
                       "presses her hands", "presses his hands", "points to", "claps")),
    ("close lean / pose", (
        "back-to-back", "back to back", "leaning against each other", "lean against each other",
        "side by side", "side-by-side", "shoulder to shoulder", "shoulder-to-shoulder",
        "heads close", "forehead", "cheek to cheek", "stand close", "standing close",
        "sitting closely", "sitting close", "stands a short distance", "stand a short distance",
        "close together",
    )),
    ("talk / face-to-face", ("talking", "convers", "posing", "facing each other", "eye contact",
                             "speaking", "looking at each other", "laughing together",
                             "explains a point", "reviewing documents", "looking towards each other",
                             "smiling at")),
]

# Fallback subclass when rules miss (under C, avoid huge complex outer ring)
COMPLEX_SUBTYPES = [
    ("shared prop / co-hold", "C2", (
        "holding a", "holds a", "holding the", "holds the", "stack of", "boxes",
        "certificate", "frisbee", "rope", "phone", "rabbit", "flowers", "balloon",
    )),
    ("group / multi-person", "C2", (
        "three ", "four ", "five ", "group", "friends stand", "children stand",
        "bases hold", "flyer",
    )),
    ("co-activity / play", "C1", (
        "playing", "throw", "catch", "watch", "anticipating", "pinata", "disc",
    )),
    ("proximity / pose", "C0", (
        "stand", "sit", "looking", "smiling", "pose",
    )),
]

# Four contact tiers: none → hand → torso/point-line → high (weight-bearing ∪ combat)
# Former C3/C4 merged: hard to rank by area; unified high-contact tier.
# Primary = C0–C3; action types under density (not board cat)
ACTION_TO_C = {
    "talk / face-to-face": "C0",
    "close lean / pose": "C0",
    "proximity / pose": "C0",
    "handshake": "C1",
    "high-five": "C1",
    "reach / touch": "C1",
    "co-activity / play": "C1",
    "hand-hold": "C2",
    "arm-around / shoulder": "C2",
    "dance hold": "C2",
    "shared prop / co-hold": "C2",
    "group / multi-person": "C2",
    "hug / embrace": "C3",
    "kiss": "C3",
    "piggyback": "C3",
    "carry / lift": "C3",
    "strike / combat": "C3",
    "grapple / wrestle": "C3",
}

C_ORDER = ["C0", "C1", "C2", "C3"]
C_LABELS = {
    "C0": "C0 none / proximity",
    "C1": "C1 hand-level",
    "C2": "C2 torso / point-line",
    "C3": "C3 high contact (intimate / combat)",
}
# Base colors per tier; inner ring shades same hue
C_HUES = {
    "C0": "#6B7280",  # slate
    "C1": "#59A14F",  # green
    "C2": "#2A9D8F",  # teal
    "C3": "#F28E2B",  # amber (high-contact highlight)
}

# fallback: keyword scoring for density when rules miss
C_FALLBACK_KW = [
    ("C3", (
        "grappl", "wrestl", "pinning", "takedown", "clinch", "punch", "kick", "combat",
        "carry", "piggyback", "lift", "hug", "embrace", "on the shoulders", "cradl",
    )),
    ("C2", ("holding hands", "arm around", "dance", "arm linked", "around the waist")),
    ("C1", ("handshake", "high-five", "high five", "touch", "reaching", "places")),
    ("C0", ("talking", "facing", "side by side", "side-by-side", "posing", "convers")),
]

def tag_action_rule(text: str) -> str:
    t = (text or "").lower()
    for label, kws in ACTION_RULES:
        if any(k in t for k in kws):
            return label
    return "other / complex"


def _hex_to_rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def _rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(
        max(0, min(255, int(rgb[0] * 255))),
        max(0, min(255, int(rgb[1] * 255))),
        max(0, min(255, int(rgb[2] * 255))),
    )


def shade_family(base_hex: str, n: int, light=0.78, dark=0.42):
    """Generate n shade variants of one hue (outer ring segments)."""
    if n <= 0:
        return []
    if n == 1:
        return [base_hex]
    br, bg, bb = _hex_to_rgb(base_hex)
    out = []
    for i in range(n):
        t = i / (n - 1)  # 0=light … 1=dark
        mix = light + (dark - light) * t
        # interpolate toward white/black, keep hue
        if mix >= 0.55:
            w = (mix - 0.55) / 0.45  # toward white
            r, g, b = br + (1 - br) * w * 0.55, bg + (1 - bg) * w * 0.55, bb + (1 - bb) * w * 0.55
        else:
            d = (0.55 - mix) / 0.55  # toward black
            r, g, b = br * (1 - 0.55 * d), bg * (1 - 0.55 * d), bb * (1 - 0.55 * d)
        # nudge toward base to avoid gray
        r, g, b = 0.35 * br + 0.65 * r, 0.35 * bg + 0.65 * g, 0.35 * bb + 0.65 * b
        out.append(_rgb_to_hex((r, g, b)))
    return out


def assign_interaction_and_c(text: str, action_rule: str) -> tuple[str, str]:
    """Return (interaction_type, contact_density C0–C3)."""
    if action_rule in ACTION_TO_C:
        return action_rule, ACTION_TO_C[action_rule]
    t = (text or "").lower()
    for label, kws in ACTION_RULES:
        if any(k in t for k in kws):
            return label, ACTION_TO_C[label]
    for label, c, kws in COMPLEX_SUBTYPES:
        if any(k in t for k in kws):
            return label, c
    for c, kws in C_FALLBACK_KW:
        if any(k in t for k in kws):
            return "complex / mixed", c
    return "complex / mixed", "C2"


def tag_venue(text: str) -> str:
    t = (text or "").lower()
    rules = [
        ("Motion-capture studio", ("motion capture", "mocap", "capture studio")),
        ("Studio / plain backdrop", ("studio", "backdrop", "cyclorama", "seamless", "plain white", "solid color", "solid white")),
        ("Home / indoor living", ("living room", "bedroom", "kitchen", "apartment", "home", "house", "sofa", "couch")),
        ("Street / urban", ("street", "sidewalk", "city", "urban", "alley", "crosswalk")),
        ("Park / nature", ("park", "garden", "forest", "beach", "mountain", "trail", "lawn", "field")),
        ("Sports / gym / arena", ("gym", "dojo", "arena", "stadium", "court", "ring", "mat", "wrestling", "karate")),
        ("Stage / ballroom", ("stage", "ballroom", "dance floor", "theater", "theatre", "club")),
        ("Office / public indoor", ("office", "lobby", "hallway", "corridor", "classroom", "restaurant", "bar")),
    ]
    for label, kws in rules:
        if any(k in t for k in kws):
            return label
    return "Other / unspecified"


def tag_io(text: str) -> str:
    t = (text or "").lower()
    out_kw = ("outdoor", "outside", "street", "park", "beach", "field", "garden", "sky")
    in_kw = ("indoor", "inside", "studio", "room", "gym", "office", "hallway", "apartment")
    o, i = any(k in t for k in out_kw), any(k in t for k in in_kw)
    if o and not i:
        return "Outdoor"
    if i and not o:
        return "Indoor"
    if o and i:
        return "Outdoor" if ("outdoor" in t or "outside" in t) else "Indoor"
    return "Unspecified"


def tag_shot(text: str) -> str:
    t = (text or "").lower()
    if any(k in t for k in ("extreme close", "close-up", "close up", "closeup")):
        return "Close-up"
    if any(k in t for k in ("full body", "full-body", "full shot", "entire body")):
        return "Full body"
    if any(k in t for k in ("wide shot", "wide-angle", "long shot", "establishing")):
        return "Wide"
    if "medium" in t or "mid-shot" in t or "waist" in t:
        return "Medium"
    return "Unspecified"


def annotate_rules(rows):
    for r in rows:
        r["action_rule"] = tag_action_rule(r["interaction"])
        itype, clev = assign_interaction_and_c(r["interaction"], r["action_rule"])
        r["interaction_type"] = itype
        r["contact_c"] = clev
        r["venue"] = tag_venue(r["setting"])
        r["io"] = tag_io(r["setting"])
        r["shot"] = tag_shot(r["camera"])
        n = r["n_subjects"]
        r["n_subj_bin"] = "4+" if n >= 4 else (str(n) if n > 0 else "0")
        r["bystander_bin"] = "With bystanders" if r["has_bystander"] else "No bystanders"
    return rows


def cluster_texts(texts, k, seed=42):
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=3, stop_words="english")
    X = vec.fit_transform(texts if any(texts) else ["empty"])
    k = min(k, max(2, X.shape[0] // 30))
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = km.fit_predict(X)
    terms = np.array(vec.get_feature_names_out())
    return labels, terms, km.cluster_centers_


def name_action_cluster(member_rule_counts: Counter, top_terms: list) -> str:
    if member_rule_counts:
        maj, n = member_rule_counts.most_common(1)[0]
        total = sum(member_rule_counts.values())
        if maj != "other / complex" and n >= max(3, 0.35 * total):
            return maj
    keep = [t for t in top_terms if t.replace(" ", "") not in STOP and len(t) > 2][:3]
    return " · ".join(keep) if keep else "misc"


def run_action_clusters(rows, k=14):
    texts = [r["action_text"] or "empty" for r in rows]
    labels, terms, centers = cluster_texts(texts, k=k)
    buckets = defaultdict(list)
    for i, lab in enumerate(labels):
        buckets[int(lab)].append(i)
        rows[i]["action_cluster_id"] = int(lab)
    cluster_meta, id_to_name = [], {}
    for cid, idxs in buckets.items():
        top = terms[centers[cid].argsort()[::-1][:8]].tolist()
        rule_c = Counter(rows[j]["action_rule"] for j in idxs)
        name = name_action_cluster(rule_c, top)
        base, suffix = name, 2
        while name in id_to_name.values():
            name = f"{base} ({suffix})"
            suffix += 1
        id_to_name[cid] = name
        cluster_meta.append({
            "cluster_id": cid, "name": name, "size": len(idxs),
            "top_terms": ", ".join(top[:6]),
            "rule_mix": dict(rule_c.most_common(5)),
            "source_mix": dict(Counter(rows[j]["source"] for j in idxs)),
            "example": rows[idxs[0]]["interaction"][:200],
        })
    for r in rows:
        r["action_natural"] = id_to_name[r["action_cluster_id"]]
        # primary axis: rule hit → rule; else TF-IDF cluster name
        if r["action_rule"] != "other / complex":
            r["action_primary"] = r["action_rule"]
        else:
            r["action_primary"] = r["action_natural"]
    cluster_meta.sort(key=lambda x: -x["size"])
    return cluster_meta


def run_activity_clusters(rows, k=10):
    texts = [r["activity_text"] or "empty" for r in rows]
    labels, terms, centers = cluster_texts(texts, k=k)
    buckets = defaultdict(list)
    for i, lab in enumerate(labels):
        buckets[int(lab)].append(i)
        rows[i]["activity_cluster_id"] = int(lab)
    meta, id_to_name = [], {}
    for cid, idxs in buckets.items():
        top = terms[centers[cid].argsort()[::-1][:6]].tolist()
        rule_c = Counter(rows[j]["action_rule"] for j in idxs)
        venue_c = Counter(rows[j]["venue"] for j in idxs)
        a = rule_c.most_common(1)[0][0]
        v = venue_c.most_common(1)[0][0]
        name = f"{a} @ {v.split('/')[0].strip()}"
        base, s = name, 2
        while name in id_to_name.values():
            name = f"{base} ({s})"
            s += 1
        id_to_name[cid] = name
        meta.append({
            "cluster_id": cid, "name": name, "size": len(idxs),
            "top_terms": ", ".join(top),
            "source_mix": dict(Counter(rows[j]["source"] for j in idxs)),
            "example": (rows[idxs[0]]["scene_summary"] or rows[idxs[0]]["interaction"])[:200],
        })
    for r in rows:
        r["activity_natural"] = id_to_name[r["activity_cluster_id"]]
    meta.sort(key=lambda x: -x["size"])
    return meta


def save_fig(fig, out_dir, stem):
    fig.savefig(str(Path(out_dir) / f"{stem}.png"))
    fig.savefig(str(Path(out_dir) / f"{stem}.pdf"))
    plt.close(fig)


def plot_donut(counts_ordered, title, center_n, out_dir, stem):
    labels = [n for n, _ in counts_ordered]
    sizes = [c for _, c in counts_ordered]
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(labels))]
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    wedges, _ = ax.pie(sizes, colors=colors, startangle=90,
                       wedgeprops=dict(width=0.42, edgecolor="white", linewidth=1.1))
    ax.text(0, 0.06, f"N = {center_n:,}", ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(0, -0.12, "target prompts", ha="center", va="center", fontsize=9, color="#555")
    ax.legend(wedges, [f"{lab} ({s})" for lab, s in zip(labels, sizes)],
              loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)
    ax.set_title(title, pad=12)
    save_fig(fig, out_dir, stem)


def plot_donut_c_actions(rows, out_dir, stem="fig_donut_c0c3_actions"):
    """Inner ring C0–C3 colors; outer ring action shades (natural clusters)."""
    # per-C action counts (desc for stable legend)
    by_c = {c: Counter() for c in C_ORDER}
    for r in rows:
        by_c[r["contact_c"]][r["interaction_type"]] += 1

    inner_sizes, inner_colors, inner_labels = [], [], []
    outer_sizes, outer_colors, outer_labels = [], [], []
    legend_outer = []  # (color, label, count, c)

    for c in C_ORDER:
        items = by_c[c].most_common()
        total = sum(v for _, v in items)
        if total == 0:
            continue
        inner_sizes.append(total)
        inner_colors.append(C_HUES[c])
        inner_labels.append(f"{c} ({total})")
        shades = shade_family(C_HUES[c], len(items))
        for (act, cnt), col in zip(items, shades):
            outer_sizes.append(cnt)
            outer_colors.append(col)
            outer_labels.append(act)
            legend_outer.append((col, act, cnt, c))

    fig, ax = plt.subplots(figsize=(9.2, 7.0))
    # outer ring: action breakdown
    ax.pie(
        outer_sizes, colors=outer_colors, startangle=90,
        radius=1.0,
        wedgeprops=dict(width=0.38, edgecolor="white", linewidth=0.9),
    )
    # inner ring: C0–C3
    wedges_in, _ = ax.pie(
        inner_sizes, colors=inner_colors, startangle=90,
        radius=0.62,
        wedgeprops=dict(width=0.28, edgecolor="white", linewidth=1.2),
    )
    ax.text(0, 0.04, f"N = {len(rows):,}", ha="center", va="center",
            fontsize=13, fontweight="bold")
    ax.text(0, -0.10, "C0–C3 × actions", ha="center", va="center",
            fontsize=8.5, color="#555")

    # left: C legend; right: actions grouped by C
    from matplotlib.patches import Patch
    leg_c = [Patch(facecolor=C_HUES[c], edgecolor="white",
                   label=f"{C_LABELS[c]}  ({sum(by_c[c].values())})")
             for c in C_ORDER if sum(by_c[c].values())]
    leg1 = ax.legend(handles=leg_c, loc="upper left", bbox_to_anchor=(-0.28, 1.02),
                     frameon=False, fontsize=8.5, title="Contact density", title_fontsize=9)
    ax.add_artist(leg1)

    # outer legend: actions per C (truncate long names)
    outer_handles, outer_labs = [], []
    for col, act, cnt, c in legend_outer:
        if cnt < max(8, int(0.008 * len(rows))):  # skip tiny slices in legend
            continue
        outer_handles.append(Patch(facecolor=col, edgecolor="white"))
        outer_labs.append(f"[{c}] {act}  ({cnt})")
    ax.legend(outer_handles, outer_labs, loc="center left", bbox_to_anchor=(1.02, 0.5),
              frameon=False, fontsize=7.5, title="Actions within each C", title_fontsize=9)

    ax.set_title("Contact density (C0–C3) with natural interaction types", pad=14)
    save_fig(fig, out_dir, stem)

    # also save inner-ring-only figure for paper crop
    fig2, ax2 = plt.subplots(figsize=(6.4, 5.6))
    c_present = [c for c in C_ORDER if sum(by_c[c].values())]
    w2, _ = ax2.pie(
        inner_sizes, colors=inner_colors, startangle=90,
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=1.3),
    )
    ax2.text(0, 0.06, f"N = {len(rows):,}", ha="center", va="center",
             fontsize=14, fontweight="bold")
    ax2.text(0, -0.12, "target prompts", ha="center", va="center", fontsize=9, color="#555")
    ax2.legend(
        w2,
        [f"{C_LABELS[c]} ({sum(by_c[c].values())})" for c in c_present],
        loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=9,
    )
    ax2.set_title("Contact-density levels (C0–C3)", pad=12)
    save_fig(fig2, out_dir, "fig_donut_c0c3")

    return {
        "by_contact_c": {c: sum(by_c[c].values()) for c in C_ORDER},
        "by_c_action": {c: dict(by_c[c].most_common()) for c in C_ORDER},
        "taxonomy": {
            "interaction_types": list(ACTION_TO_C.keys()) + ["complex / mixed"],
            "action_to_c": dict(ACTION_TO_C),
            "c_labels": C_LABELS,
            "note": "Types from layer.interaction natural tags; board_cat not used",
        },
    }


def _barh(ax, labels, values, color, title):
    y = np.arange(len(labels))
    ax.barh(y, values, color=color, height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_title(title)
    xmax = max(values) if values else 1
    for i, v in enumerate(values):
        ax.text(v + xmax * 0.01, i, str(v), va="center", fontsize=8)
    ax.set_xlim(0, xmax * 1.18)


def plot_rule_action_and_venue(rows, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
    items = Counter(r["action_rule"] for r in rows).most_common()
    _barh(axes[0], [k for k, _ in items], [v for _, v in items], "#4E79A7",
          "Action (rule tags on interaction text)")
    items = Counter(r["venue"] for r in rows).most_common()
    _barh(axes[1], [k for k, _ in items], [v for _, v in items], "#76B7B2",
          "Venue (rule tags on setting text)")
    fig.suptitle("Natural tags — independent of board category", y=1.02)
    fig.tight_layout()
    save_fig(fig, out_dir, "fig_natural_action_venue")


def plot_facets(rows, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.8))
    specs = [
        (axes[0, 0], "io", ["Indoor", "Outdoor", "Unspecified"], "#4E79A7", "Indoor / outdoor"),
        (axes[0, 1], "shot", ["Close-up", "Medium", "Full body", "Wide", "Unspecified"], "#59A14F", "Shot scale"),
        (axes[1, 0], "n_subj_bin", ["1", "2", "3", "4+", "0"], "#B07AA1", "Main subjects"),
        (axes[1, 1], "bystander_bin", ["No bystanders", "With bystanders"], "#F28E2B", "Bystanders"),
    ]
    for ax, key, order, color, title in specs:
        c = Counter(r[key] for r in rows)
        labels = [k for k in order if c.get(k, 0)] + [k for k in sorted(c) if k not in order]
        _barh(ax, labels, [c[k] for k in labels], color, title)
    fig.suptitle("Coverage axes from layer fields", y=1.01)
    fig.tight_layout()
    save_fig(fig, out_dir, "fig_natural_facets")


def plot_heatmap_action_venue(rows, out_dir):
    actions = [a for a, _ in Counter(r["action_rule"] for r in rows).most_common()]
    venues = [v for v, _ in Counter(r["venue"] for r in rows).most_common()]
    mat = np.zeros((len(actions), len(venues)))
    ai, vi = {a: i for i, a in enumerate(actions)}, {v: i for i, v in enumerate(venues)}
    for r in rows:
        mat[ai[r["action_rule"]], vi[r["venue"]]] += 1
    fig, ax = plt.subplots(figsize=(max(7, 0.9 * len(venues) + 2), max(4.5, 0.35 * len(actions) + 1.5)))
    im = ax.imshow(mat, aspect="auto", cmap="Blues")
    ax.set_xticks(range(len(venues)))
    ax.set_xticklabels(venues, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(actions)))
    ax.set_yticklabels(actions, fontsize=8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i, j] > 0:
                ax.text(j, i, f"{int(mat[i, j])}", ha="center", va="center", fontsize=6,
                        color="white" if mat[i, j] > mat.max() * 0.55 else "#111")
    ax.set_title("Natural action × venue (counts)")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    save_fig(fig, out_dir, "fig_heatmap_action_venue")


def plot_source_within_action(rows, out_dir):
    actions = [a for a, _ in Counter(r["action_rule"] for r in rows).most_common()]
    sources = ["CC0 / other", "CHI3D", "Harmony4D"]
    colors = {"CC0 / other": "#BAB0AC", "CHI3D": "#4E79A7", "Harmony4D": "#F28E2B"}
    data = {s: [] for s in sources}
    for a in actions:
        c = Counter(r["source"] for r in rows if r["action_rule"] == a)
        for s in sources:
            data[s].append(c.get(s, 0))
    fig, ax = plt.subplots(figsize=(8.5, max(4.0, 0.35 * len(actions) + 1)))
    y, left = np.arange(len(actions)), np.zeros(len(actions))
    for s in sources:
        vals = np.array(data[s], dtype=float)
        ax.barh(y, vals, left=left, color=colors[s], height=0.65, label=s)
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels(actions)
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.set_title("Source composition inside each natural action")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    save_fig(fig, out_dir, "fig_source_within_action")


def plot_activity_bars(activity_meta, out_dir):
    items = [(m["name"], m["size"]) for m in activity_meta]
    fig, ax = plt.subplots(figsize=(8.0, max(4.0, 0.35 * len(items) + 1)))
    _barh(ax, [n for n, _ in items], [s for _, s in items], "#E15759",
          "Activity clusters (scene_summary + interaction)")
    fig.tight_layout()
    save_fig(fig, out_dir, "fig_activity_clusters")


def write_csv(rows, path):
    fields = [
        "source", "board_cat", "anchor", "target_id",
        "contact_c", "interaction_type",
        "action_primary", "action_rule", "action_natural", "activity_natural",
        "venue", "io", "shot", "n_subjects", "has_bystander",
        "interaction", "setting",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_summary(rows, action_meta, activity_meta, path, c_stats=None):
    def dist(key):
        return dict(Counter(r[key] for r in rows).most_common())

    payload = {
        "N": len(rows),
        "n_scenes": len({(r["board_cat"], r["anchor"]) for r in rows}),
        "method": (
            "C0–C3 contact density × natural interaction types from layers; "
            "board_cat is appendix-only"
        ),
        "by_source": dist("source"),
        "by_contact_c": {c: sum(1 for r in rows if r["contact_c"] == c) for c in C_ORDER},
        "by_interaction_type": dist("interaction_type"),
        "by_action_rule": dist("action_rule"),
        "by_action_natural": dist("action_natural"),
        "by_action_primary": dist("action_primary"),
        "by_activity_natural": dist("activity_natural"),
        "by_venue": dist("venue"),
        "by_io": dist("io"),
        "by_shot": dist("shot"),
        "by_n_subjects": dist("n_subj_bin"),
        "by_bystander": dist("bystander_bin"),
        "by_board_cat_appendix": dist("board_cat"),
        "action_clusters_tfidf": action_meta,
        "activity_clusters_tfidf": activity_meta,
    }
    if c_stats:
        payload["by_c_action"] = c_stats.get("by_c_action")
        payload["taxonomy"] = c_stats.get("taxonomy")
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def write_html(summary, fig_stems, out_dir):
    parts = [f"""<!DOCTYPE html><html lang=en><head><meta charset=utf-8>
<title>MPIE natural prompt distribution (N={summary['N']})</title>
<style>
body{{font-family:-apple-system,Helvetica,Arial,sans-serif;margin:24px;color:#1f2430;background:#fafafa}}
h1{{font-size:22px;margin:0 0 6px}} h2{{font-size:16px;margin:28px 0 10px;border-bottom:1px solid #e5e7eb;padding-bottom:6px}}
.meta{{color:#555;margin-bottom:16px;max-width:820px;line-height:1.45}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
.card{{background:#fff;border:1px solid #e8eaed;border-radius:8px;padding:12px}}
.card img{{width:100%;height:auto}}
table{{border-collapse:collapse;width:100%;font-size:13px;background:#fff}}
th,td{{border:1px solid #e5e7eb;padding:6px 8px;text-align:left;vertical-align:top}}
th{{background:#f3f4f6}}
.note{{background:#fff8e6;border:1px solid #f0e0b0;padding:10px 12px;border-radius:6px;margin:12px 0;max-width:820px}}
@media (max-width:900px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body>
<h1>MPIE-Bench natural prompt distribution</h1>
<p class=meta>N = <b>{summary['N']}</b> target prompts &nbsp;|&nbsp; {summary['n_scenes']} scenes<br>
Hero axis: <b>C0–C3 contact density</b> × natural <b>interaction types</b> from
<code>layers.interaction</code> (not CC0 board cat). CHI3D / H4D fold into the same buckets.</p>
<div class=note>Inner ring = C0–C3 (4 hue families). Outer ring = action subtypes in matching shades.
Replaces the old 14-class board-<code>cat</code> story.</div>
"""]
    parts.append("<h2>Figures</h2><div class=grid>")
    for stem, cap in fig_stems:
        if (Path(out_dir) / f"{stem}.png").exists():
            parts.append(
                f'<div class=card><img src="{stem}.png" alt="{stem}">'
                f'<div style="margin-top:6px;color:#555;font-size:12px">{cap}</div></div>'
            )
    parts.append("</div>")

    def table(title, d):
        parts.append(f"<h2>{title}</h2><table><tr><th>Label</th><th>Count</th><th>%</th></tr>")
        n = summary["N"] or 1
        for k, v in d.items():
            parts.append(f"<tr><td>{k}</td><td>{v}</td><td>{100 * v / n:.1f}%</td></tr>")
        parts.append("</table>")

    table("By data source", summary["by_source"])
    table("Contact density C0–C3", summary.get("by_contact_c") or {})
    table("Interaction type (natural, ~14)", summary.get("by_interaction_type") or {})
    if summary.get("by_c_action"):
        parts.append("<h2>Actions within each C</h2>")
        for c in C_ORDER:
            d = summary["by_c_action"].get(c) or {}
            if d:
                table(C_LABELS[c], d)
    table("Primary action (rule + TF-IDF fallback)", summary["by_action_primary"])
    table("Natural action (rule tags only)", summary["by_action_rule"])
    table("Natural action (TF-IDF cluster names)", summary["by_action_natural"])
    table("Venue (from setting text)", summary["by_venue"])
    table("Activity clusters", summary["by_activity_natural"])
    table("Indoor / outdoor", summary["by_io"])
    table("Shot scale", summary["by_shot"])

    parts.append("<h2>TF-IDF action clusters (detail)</h2><table>")
    parts.append("<tr><th>Name</th><th>Size</th><th>Source mix</th><th>Top terms</th><th>Example</th></tr>")
    for c in summary["action_clusters_tfidf"]:
        src = ", ".join(f"{k}:{v}" for k, v in c["source_mix"].items())
        ex = (c.get("example") or "").replace("<", "&lt;")[:160]
        parts.append(
            f"<tr><td>{c['name']}</td><td>{c['size']}</td><td>{src}</td>"
            f"<td>{c['top_terms']}</td><td>{ex}</td></tr>"
        )
    parts.append("</table>")
    parts.append("<h2>Appendix — board category (not used for main distribution)</h2>")
    table("Board cat (CC0 taxonomy / mocap folders)", summary["by_board_cat_appendix"])
    parts.append("<p class=meta>CSV: <code>targets_natural.csv</code> · JSON: <code>summary.json</code></p></body></html>")
    (Path(out_dir) / "index.html").write_text("".join(parts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-dir", default=str(PROMPT_DIR))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--k-action", type=int, default=14)
    ap.add_argument("--k-activity", type=int, default=10)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    setup_style()

    print("Loading…")
    rows = load_rows(Path(args.prompt_dir))
    print(f"N={len(rows)}")
    annotate_rules(rows)
    print("Clustering actions…")
    action_meta = run_action_clusters(rows, k=args.k_action)
    print("Clustering activities…")
    activity_meta = run_activity_clusters(rows, k=args.k_activity)

    print("Plotting…")
    fig_stems = []
    # main figure: C0–C3 × action shades
    c_stats = plot_donut_c_actions(rows, out)
    fig_stems.append(("fig_donut_c0c3_actions",
                      "HERO: C0–C3 (inner hue) × natural actions (outer shades)"))
    fig_stems.append(("fig_donut_c0c3", "C0–C3 contact-density only"))
    plot_donut(Counter(r["interaction_type"] for r in rows).most_common(),
               "Natural interaction types (not board cat)", len(rows), out, "fig_donut_category")
    fig_stems.append(("fig_donut_category", "Flat view of ~14 natural interaction types"))
    plot_donut(Counter(r["action_natural"] for r in rows).most_common(),
               "TF-IDF action clusters (auto-named)", len(rows), out, "fig_donut_tfidf_action")
    fig_stems.append(("fig_donut_tfidf_action", "TF-IDF action clusters"))
    plot_rule_action_and_venue(rows, out)
    fig_stems.append(("fig_natural_action_venue", "Action + venue rule tags"))
    plot_facets(rows, out)
    fig_stems.append(("fig_natural_facets", "IO / shot / subjects / bystanders"))
    # heatmap / source use interaction_type
    for r in rows:
        r["action_rule_plot"] = r["action_rule"]
        r["action_rule"] = r["interaction_type"]
    plot_heatmap_action_venue(rows, out)
    fig_stems.append(("fig_heatmap_action_venue", "Interaction type × venue heatmap"))
    plot_source_within_action(rows, out)
    fig_stems.append(("fig_source_within_action", "CHI3D / H4D / CC0 inside each type"))
    for r in rows:
        r["action_rule"] = r["action_rule_plot"]
    plot_activity_bars(activity_meta, out)
    fig_stems.append(("fig_activity_clusters", "Activity clusters"))

    write_csv(rows, out / "targets_natural.csv")
    write_csv(rows, out / "targets_tagged.csv")
    summary = write_summary(rows, action_meta, activity_meta, out / "summary.json", c_stats=c_stats)
    write_html(summary, fig_stems, out)
    print("---")
    print("source", summary["by_source"])
    print("contact_c", summary["by_contact_c"])
    print("interaction_type top", list(summary["by_interaction_type"].items())[:14])
    print("venue top", list(summary["by_venue"].items())[:8])
    print("Wrote", out)


if __name__ == "__main__":
    main()
