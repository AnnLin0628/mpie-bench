#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test-set prompt distribution analysis (paper figures).

Rule labels + light TF-IDF clustering on prompts_full layers:
  - interaction type donut
  - setting / shot / person count / bystander / contact density bar charts
  - interaction × setting heat map
  - interaction word cloud (supplementary)
  - summary HTML + CSV + PDF/PNG

Usage:
  python analyze_prompt_distribution.py
  python analyze_prompt_distribution.py --out $MPIE_ROOT/data/manifests/prompt_distribution
"""
from __future__ import print_function, division

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

# board category → contact density (aligned with testset_split.md)
CAT_DENSITY = {
    "face_to_face_talk": "C1",
    "handshake": "C1",
    "high_five": "C1",
    "hand_hold": "C2",
    "arm_around_shoulder": "C2",
    "dance": "C2",
    "hug": "C3",
    "piggyback": "C3",
    "carry_lift": "C3",
    "dance_lift": "C3",
    "wrestle_grapple": "C3",
    "fight_combat": "C3",
    "restrain_pin": "C3",
    "other_multi_person": "C3",
    "chi3d": "mixed",
    "harmony4d": "mixed",
    "ava": "mixed",
    "panoptic": "C3",
}

CAT_DISPLAY = {
    "arm_around_shoulder": "Arm around shoulder",
    "carry_lift": "Carry / lift",
    "chi3d": "CHI3D (mixed)",
    "dance": "Dance",
    "dance_lift": "Dance lift",
    "face_to_face_talk": "Face-to-face talk",
    "fight_combat": "Fight / combat",
    "hand_hold": "Hand hold",
    "handshake": "Handshake",
    "harmony4d": "Harmony4D (mixed)",
    "high_five": "High five",
    "hug": "Hug",
    "other_multi_person": "Other multi-person",
    "piggyback": "Piggyback",
    "wrestle_grapple": "Wrestle / grapple",
}

# ColorBrewer Set3-inspired, colorblind-friendlier reorder
PALETTE = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
    "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
    "#86BCB6", "#D37295", "#8CD17D", "#B6992D", "#499894",
    "#D4A6C8", "#FABFD2", "#B6992D",
]

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "with", "from",
    "by", "for", "as", "is", "are", "their", "her", "his", "him", "she", "he",
    "they", "them", "who", "while", "both", "each", "other", "one", "two",
    "left", "right", "side", "frame", "woman", "man", "person", "people",
    "wearing", "looking", "standing", "against", "background", "camera",
    "slightly", "closely", "together", "around", "over", "under", "into",
    "toward", "towards", "across", "near", "between", "through", "onto",
}


def setup_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def load_rows(prompt_dir):
    rows = []
    for f in sorted(Path(prompt_dir).glob("prompts_*.json")):
        for item in json.loads(f.read_text()):
            vlm = item.get("vlm") or {}
            if "error" in vlm:
                continue
            summary = vlm.get("scene_summary") or ""
            for t in vlm.get("targets") or []:
                layers = t.get("layers") or {}
                subjects = layers.get("subjects") or []
                byst = layers.get("bystanders")
                byst_s = "" if byst is None else str(byst).strip()
                has_byst = bool(byst_s) and byst_s.lower() not in ("none", "null", "n/a", "-")
                rows.append({
                    "cat": item["cat"],
                    "anchor": item["anchor"],
                    "target_id": t.get("target"),
                    "interaction": (layers.get("interaction") or "").strip(),
                    "setting": (layers.get("setting") or "").strip(),
                    "camera": (layers.get("camera") or "").strip(),
                    "lighting": (layers.get("lighting_color") or "").strip(),
                    "style": (layers.get("style") or "").strip(),
                    "prompt": (t.get("prompt") or "").strip(),
                    "scene_summary": summary,
                    "n_subjects": len(subjects) if isinstance(subjects, list) else 0,
                    "has_bystander": has_byst,
                    "confidence": t.get("confidence"),
                    "flag_underage": t.get("flag_underage"),
                })
    return rows


def tag_indoor_outdoor(text):
    t = (text or "").lower()
    outdoor_kw = (
        "outdoor", "outside", "street", "park", "beach", "field", "garden",
        "sidewalk", "plaza", "courtyard", "rooftop", "yard", "lawn", "trail",
        "mountain", "forest", "snow", "rain", "sky", "stadium", "court",
        "playground", "pier", "dock", "boardwalk",
    )
    indoor_kw = (
        "indoor", "inside", "studio", "room", "hallway", "corridor", "gym",
        "office", "kitchen", "bedroom", "living room", "classroom", "stage",
        "arena", "dojo", "club", "bar", "restaurant", "lobby", "warehouse",
        "apartment", "home", "house", "basement", "bathroom",
    )
    out_hit = any(k in t for k in outdoor_kw)
    in_hit = any(k in t for k in indoor_kw)
    if out_hit and not in_hit:
        return "Outdoor"
    if in_hit and not out_hit:
        return "Indoor"
    if out_hit and in_hit:
        # both mentioned: prefer outdoor if "outdoor" literal, else indoor
        if "outdoor" in t or "outside" in t:
            return "Outdoor"
        return "Indoor"
    return "Unspecified"


def tag_venue(text):
    t = (text or "").lower()
    rules = [
        ("Studio / plain backdrop", ("studio", "backdrop", "cyclorama", "seamless", "plain white", "solid color")),
        ("Home / indoor living", ("living room", "bedroom", "kitchen", "apartment", "home", "house", "sofa")),
        ("Street / urban", ("street", "sidewalk", "city", "urban", "alley", "crosswalk")),
        ("Park / nature", ("park", "garden", "forest", "beach", "mountain", "trail", "lawn", "field")),
        ("Sports / gym / arena", ("gym", "dojo", "arena", "stadium", "court", "ring", "mat", "wrestling")),
        ("Stage / performance", ("stage", "ballroom", "dance floor", "theater", "theatre", "club")),
        ("Office / public indoor", ("office", "lobby", "hallway", "corridor", "classroom", "restaurant", "bar")),
    ]
    for label, kws in rules:
        if any(k in t for k in kws):
            return label
    return "Other / unspecified"


def tag_shot(text):
    t = (text or "").lower()
    # order matters: more specific first
    if any(k in t for k in ("extreme close", "close-up", "close up", "closeup", "tight shot")):
        return "Close-up"
    if any(k in t for k in ("medium close", "medium-close", "bust shot", "head and shoulders")):
        return "Medium close"
    if any(k in t for k in ("full body", "full-body", "full shot", "entire body", "head to toe")):
        return "Full body"
    if any(k in t for k in ("wide shot", "wide-angle", "long shot", "establishing", "wide angle")):
        return "Wide"
    if "medium" in t or "mid-shot" in t or "mid shot" in t or "waist" in t:
        return "Medium"
    return "Unspecified"


def tag_contact(text, cat):
    """Fine contact phrase from interaction text; fallback to category."""
    t = (text or "").lower()
    rules = [
        ("handshake", ("handshake", "shaking hands", "shake hands")),
        ("high-five", ("high-five", "high five", "highfive")),
        ("hand-hold", ("holding hands", "hold hands", "hand in hand", "interlocked fingers", "clasping hands")),
        ("arm-around", ("arm around", "arms around", "arm over the shoulder", "around the shoulder")),
        ("hug / embrace", ("hug", "embrace", "embracing", "cuddling")),
        ("kiss", ("kiss", "kissing")),
        ("piggyback", ("piggyback", "on the back", "carried on")),
        ("carry / lift", ("carrying", "carries", "lifting", "lifted", "bridal carry", "in arms")),
        ("dance hold", ("dance", "ballroom", "waltz", "partner hold", "leading")),
        ("grapple / wrestle", ("grappl", "wrestl", "clinch", "takedown", "pinning", "pinned", "grab", "grabbing")),
        ("strike / combat", ("punch", "kick", "strike", "fighting", "combat", "hit ", "hitting", "push", "pushing")),
        ("reach / touch", ("touch", "touches", "touching", "extends", "reaching", "places her hand", "places his hand", "hand on the")),
        ("talk / pose", ("talking", "convers", "posing", "facing each other", "eye contact")),
        ("group / multi", ("group", "three people", "four people", "crowd", "circle")),
    ]
    for label, kws in rules:
        if any(k in t for k in kws):
            return label
    # fallback from cat
    fallback = {
        "handshake": "handshake",
        "high_five": "high-five",
        "hand_hold": "hand-hold",
        "arm_around_shoulder": "arm-around",
        "hug": "hug / embrace",
        "piggyback": "piggyback",
        "carry_lift": "carry / lift",
        "dance_lift": "carry / lift",
        "dance": "dance hold",
        "wrestle_grapple": "grapple / wrestle",
        "fight_combat": "strike / combat",
        "face_to_face_talk": "talk / pose",
        "restrain_pin": "grapple / wrestle",
        "other_multi_person": "group / multi",
    }
    return fallback.get(cat, "other")


def annotate(rows):
    for r in rows:
        r["density"] = CAT_DENSITY.get(r["cat"], "other")
        r["io"] = tag_indoor_outdoor(r["setting"])
        r["venue"] = tag_venue(r["setting"])
        r["shot"] = tag_shot(r["camera"])
        r["contact"] = tag_contact(r["interaction"], r["cat"])
        if r["n_subjects"] <= 0:
            r["n_subj_bin"] = "0"
        elif r["n_subjects"] == 1:
            r["n_subj_bin"] = "1"
        elif r["n_subjects"] == 2:
            r["n_subj_bin"] = "2"
        elif r["n_subjects"] == 3:
            r["n_subj_bin"] = "3"
        else:
            r["n_subj_bin"] = "4+"
        r["bystander_bin"] = "With bystanders" if r["has_bystander"] else "No bystanders"
    return rows


def save_fig(fig, out_dir, stem):
    out_dir = Path(out_dir)
    png = out_dir / (stem + ".png")
    pdf = out_dir / (stem + ".pdf")
    fig.savefig(str(png))
    fig.savefig(str(pdf))
    plt.close(fig)
    return png, pdf


def plot_donut_cat(rows, out_dir):
    counts = Counter(r["cat"] for r in rows)
    # sort by count desc
    items = sorted(counts.items(), key=lambda x: -x[1])
    labels = [CAT_DISPLAY.get(c, c) for c, _ in items]
    sizes = [n for _, n in items]
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(items))]
    n = sum(sizes)

    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    wedges, _ = ax.pie(
        sizes, colors=colors, startangle=90,
        wedgeprops=dict(width=0.42, edgecolor="white", linewidth=1.2),
    )
    ax.text(0, 0.06, "N = {:,}".format(n), ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(0, -0.12, "target prompts", ha="center", va="center", fontsize=9, color="#555555")
    ax.legend(
        wedges, ["{} ({})".format(lab, s) for lab, s in zip(labels, sizes)],
        loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8,
    )
    ax.set_title("Interaction category distribution", pad=12)
    return save_fig(fig, out_dir, "fig_donut_category")


def _barh(ax, labels, values, color="#4E79A7", title="", xlabel="Count"):
    y = np.arange(len(labels))
    ax.barh(y, values, color=color, height=0.62, edgecolor="none")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    xmax = max(values) if values else 1
    for i, v in enumerate(values):
        ax.text(v + xmax * 0.01, i, str(v), va="center", fontsize=8, color="#333333")
    ax.set_xlim(0, xmax * 1.18)


def plot_facet_bars(rows, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.8))
    specs = [
        (axes[0, 0], "io", ["Indoor", "Outdoor", "Unspecified"], "#4E79A7", "Indoor / outdoor"),
        (axes[0, 1], "shot", ["Close-up", "Medium close", "Medium", "Full body", "Wide", "Unspecified"],
         "#59A14F", "Shot scale"),
        (axes[1, 0], "density", ["C1", "C2", "C3", "mixed"], "#E15759", "Contact-density tier"),
        (axes[1, 1], "n_subj_bin", ["1", "2", "3", "4+", "0"], "#B07AA1", "Main subjects per target"),
    ]
    for ax, key, order, color, title in specs:
        c = Counter(r[key] for r in rows)
        labels = [k for k in order if c.get(k, 0) > 0] + [k for k in sorted(c) if k not in order]
        values = [c[k] for k in labels]
        _barh(ax, labels, values, color=color, title=title)
    fig.suptitle("Multi-axis coverage (rule tags on layer fields)", y=1.01, fontsize=12)
    fig.tight_layout()
    return save_fig(fig, out_dir, "fig_bars_facets")


def plot_venue_bystander(rows, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    c_v = Counter(r["venue"] for r in rows)
    items = sorted(c_v.items(), key=lambda x: -x[1])
    _barh(axes[0], [k for k, _ in items], [v for _, v in items], color="#76B7B2", title="Venue type (from setting)")
    c_b = Counter(r["bystander_bin"] for r in rows)
    order = ["No bystanders", "With bystanders"]
    _barh(axes[1], order, [c_b[k] for k in order], color="#F28E2B", title="Bystanders")
    fig.tight_layout()
    return save_fig(fig, out_dir, "fig_bars_venue_bystander")


def plot_heatmap_cat_io(rows, out_dir):
    cats = [c for c, _ in Counter(r["cat"] for r in rows).most_common()]
    ios = ["Indoor", "Outdoor", "Unspecified"]
    mat = np.zeros((len(cats), len(ios)), dtype=float)
    idx = {c: i for i, c in enumerate(cats)}
    jdx = {o: i for i, o in enumerate(ios)}
    for r in rows:
        mat[idx[r["cat"]], jdx[r["io"]]] += 1
    # row-normalize for readability
    row_sum = mat.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    mat_n = mat / row_sum

    fig, ax = plt.subplots(figsize=(6.5, max(4.5, 0.32 * len(cats) + 1.5)))
    im = ax.imshow(mat_n, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(ios)))
    ax.set_xticklabels(ios)
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels([CAT_DISPLAY.get(c, c) for c in cats], fontsize=8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, "{:.0f}".format(mat[i, j]), ha="center", va="center",
                    fontsize=7, color="#111111" if mat_n[i, j] < 0.55 else "white")
    ax.set_title("Category × indoor/outdoor (counts; color = row share)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row share")
    fig.tight_layout()
    return save_fig(fig, out_dir, "fig_heatmap_cat_io")


def plot_contact_bars(rows, out_dir):
    c = Counter(r["contact"] for r in rows)
    items = sorted(c.items(), key=lambda x: -x[1])
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    _barh(ax, [k for k, _ in items], [v for _, v in items], color="#4E79A7",
          title="Fine-grained contact / action (from interaction text)")
    fig.tight_layout()
    return save_fig(fig, out_dir, "fig_bars_contact")


def clean_interaction_text(text):
    t = (text or "").lower()
    t = re.sub(r"\([^)]*\)", " ", t)  # drop (R1) etc.
    t = re.sub(r"[^a-z\s\-]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def plot_top_terms(rows, out_dir):
    """Top interaction unigrams (paper-safe substitute when wordcloud/Pillow mismatch)."""
    texts = [clean_interaction_text(r["interaction"]) for r in rows if r["interaction"]]
    words = []
    for t in texts:
        words.extend(w for w in t.split() if len(w) > 2 and w not in STOPWORDS)
    freq = Counter(words)
    for w in ("woman", "man", "person", "their", "with"):
        freq.pop(w, None)
    if not freq:
        return None
    top = freq.most_common(30)
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    _barh(ax, [w for w, _ in top], [c for _, c in top], color="#4E79A7",
          title="Top interaction terms (cleaned layer text)")
    fig.tight_layout()
    return save_fig(fig, out_dir, "fig_wordcloud_interaction")


def cluster_interaction(rows, k=12):
    texts = [clean_interaction_text(r["interaction"]) or "empty" for r in rows]
    vec = TfidfVectorizer(max_features=4000, ngram_range=(1, 2), min_df=3, stop_words="english")
    X = vec.fit_transform(texts)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    terms = np.array(vec.get_feature_names())
    clusters = []
    for i in range(k):
        center = km.cluster_centers_[i]
        top = terms[center.argsort()[::-1][:6]]
        idxs = [j for j, lab in enumerate(labels) if lab == i]
        clusters.append({
            "cluster_id": i,
            "size": len(idxs),
            "top_terms": ", ".join(top.tolist()),
            "example": rows[idxs[0]]["interaction"][:180] if idxs else "",
            "top_cats": ", ".join(
                "{}:{}".format(CAT_DISPLAY.get(c, c), n)
                for c, n in Counter(rows[j]["cat"] for j in idxs).most_common(3)
            ),
        })
    clusters.sort(key=lambda x: -x["size"])
    for r, lab in zip(rows, labels):
        r["interaction_cluster"] = int(lab)
    return clusters


def write_csv(rows, path):
    fields = [
        "cat", "anchor", "target_id", "density", "io", "venue", "shot", "contact",
        "n_subjects", "n_subj_bin", "has_bystander", "bystander_bin",
        "interaction_cluster", "confidence", "flag_underage",
        "interaction", "setting", "camera",
    ]
    with open(str(path), "w") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_summary_json(rows, clusters, path):
    def dist(key):
        c = Counter(r[key] for r in rows)
        return dict(sorted(c.items(), key=lambda x: -x[1]))

    payload = {
        "N": len(rows),
        "n_scenes": len({(r["cat"], r["anchor"]) for r in rows}),
        "by_cat": dist("cat"),
        "by_density": dist("density"),
        "by_io": dist("io"),
        "by_venue": dist("venue"),
        "by_shot": dist("shot"),
        "by_contact": dist("contact"),
        "by_n_subjects": dist("n_subj_bin"),
        "by_bystander": dist("bystander_bin"),
        "flag_underage_true": sum(1 for r in rows if r.get("flag_underage") is True),
        "interaction_clusters": clusters,
    }
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def write_html(summary, fig_stems, out_dir):
    parts = ["""<!DOCTYPE html><html lang=en><head><meta charset=utf-8>
<title>MPIE-Bench prompt distribution (N={N})</title>
<style>
body{{font-family:-apple-system,"Helvetica Neue",Arial,sans-serif;margin:24px;color:#1f2430;background:#fafafa}}
h1{{font-size:22px;margin:0 0 6px}}
h2{{font-size:16px;margin:28px 0 10px;border-bottom:1px solid #e5e7eb;padding-bottom:6px}}
.meta{{color:#555;margin-bottom:18px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
.card{{background:#fff;border:1px solid #e8eaed;border-radius:8px;padding:12px}}
.card img{{width:100%;height:auto;display:block}}
table{{border-collapse:collapse;width:100%;font-size:13px;background:#fff}}
th,td{{border:1px solid #e5e7eb;padding:6px 8px;text-align:left}}
th{{background:#f3f4f6}}
code{{background:#f3f4f6;padding:1px 4px;border-radius:3px}}
@media (max-width:900px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body>
<h1>MPIE-Bench test-set prompt distribution</h1>
<p class=meta>N = <b>{N}</b> target prompts &nbsp;|&nbsp; {n_scenes} scenes &nbsp;|&nbsp;
rule tags on <code>layers.*</code> + TF-IDF/KMeans on cleaned <code>interaction</code></p>
""".format(N=summary["N"], n_scenes=summary["n_scenes"])]

    # figures
    parts.append("<h2>Paper figures</h2><div class=grid>")
    for stem, cap in fig_stems:
        p = Path(out_dir) / (stem + ".png")
        if p.exists():
            parts.append(
                '<div class=card><img src="{}.png" alt="{}"><div style="margin-top:6px;color:#555;font-size:12px">{}</div></div>'.format(
                    stem, stem, cap
                )
            )
    parts.append("</div>")

    def table(title, d, rename=None):
        parts.append("<h2>{}</h2><table><tr><th>Label</th><th>Count</th><th>%</th></tr>".format(title))
        n = summary["N"] or 1
        for k, v in d.items():
            lab = (rename or {}).get(k, k)
            parts.append("<tr><td>{}</td><td>{}</td><td>{:.1f}%</td></tr>".format(lab, v, 100.0 * v / n))
        parts.append("</table>")

    table("By board category", summary["by_cat"], CAT_DISPLAY)
    table("By contact-density tier", summary["by_density"])
    table("Indoor / outdoor", summary["by_io"])
    table("Venue", summary["by_venue"])
    table("Shot scale", summary["by_shot"])
    table("Fine contact / action", summary["by_contact"])
    table("Main subjects", summary["by_n_subjects"])
    table("Bystanders", summary["by_bystander"])

    parts.append("<h2>Interaction TF-IDF clusters (k={})</h2>".format(len(summary["interaction_clusters"])))
    parts.append("<table><tr><th>#</th><th>Size</th><th>Top terms</th><th>Top cats</th><th>Example</th></tr>")
    for c in summary["interaction_clusters"]:
        parts.append(
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                c["cluster_id"], c["size"], c["top_terms"], c["top_cats"],
                (c["example"] or "").replace("<", "&lt;")[:160],
            )
        )
    parts.append("</table>")
    parts.append("<p class=meta>Artifacts: PDF/PNG next to this HTML; CSV <code>targets_tagged.csv</code>; JSON <code>summary.json</code>.</p>")
    parts.append("</body></html>")
    (Path(out_dir) / "index.html").write_text("".join(parts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-dir", default=str(PROMPT_DIR))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--k", type=int, default=12, help="TF-IDF KMeans clusters for interaction")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    print("Loading prompts from", args.prompt_dir)
    rows = load_rows(args.prompt_dir)
    print("Loaded", len(rows), "target prompts")
    if not rows:
        raise SystemExit("no prompts found")

    annotate(rows)
    print("Clustering interaction text (TF-IDF + KMeans k={})...".format(args.k))
    clusters = cluster_interaction(rows, k=args.k)

    print("Plotting figures...")
    fig_stems = []
    plot_donut_cat(rows, out_dir)
    fig_stems.append(("fig_donut_category", "Interaction category donut"))
    plot_facet_bars(rows, out_dir)
    fig_stems.append(("fig_bars_facets", "Indoor/outdoor · shot · density · #subjects"))
    plot_venue_bystander(rows, out_dir)
    fig_stems.append(("fig_bars_venue_bystander", "Venue type · bystanders"))
    plot_heatmap_cat_io(rows, out_dir)
    fig_stems.append(("fig_heatmap_cat_io", "Category × indoor/outdoor heatmap"))
    plot_contact_bars(rows, out_dir)
    fig_stems.append(("fig_bars_contact", "Fine-grained contact / action"))
    plot_top_terms(rows, out_dir)
    fig_stems.append(("fig_wordcloud_interaction", "Top interaction terms (aux)"))

    write_csv(rows, out_dir / "targets_tagged.csv")
    summary = write_summary_json(rows, clusters, out_dir / "summary.json")
    write_html(summary, fig_stems, out_dir)

    # quick console summary
    print("---")
    print("N =", summary["N"], "| scenes =", summary["n_scenes"])
    print("Indoor/Outdoor:", summary["by_io"])
    print("Shot:", summary["by_shot"])
    print("Density:", summary["by_density"])
    print("Subjects:", summary["by_n_subjects"])
    print("Bystanders:", summary["by_bystander"])
    print("Wrote ->", out_dir)
    print("HTML ->", out_dir / "index.html")


if __name__ == "__main__":
    main()
