#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Interaction density weight calibration: use Hi4D Vertex-level true contact annotation is fitted with four weights.

process(Execute after data is obtained):
  1. right Hi4D Four features are calculated for each frame(bbox_IoU / PoseProximity / FlowMag / OcclusionRatio)
     —— Direct reuse interaction_density.py characteristic function of, Read here where it falls into the feature cache
  2. y = Hi4D Official label "Whether there is vertex contact in this frame"(binary)
  3. Logistic regression fitting → The normalization coefficient is w1-w4
  4. CHI3D of 8 Class interaction annotation is reserved for verification(newspaper AUC), Not involved in fitting

enter: --features features.parquet (List: iou,prox,flow,occ,contact_gt)
output: Calibration weights + train/verify AUC, Glue directly in interaction_density.py --weights
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

FEATS = ["iou", "prox", "flow", "occ"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="Hi4D feature+touchGT parquet")
    ap.add_argument("--holdout", default="", help="CHI3D set aside parquet(Optional)")
    args = ap.parse_args()

    df = pd.read_parquet(args.features)
    X, y = df[FEATS].values, df["contact_gt"].astype(int).values
    clf = LogisticRegression(class_weight="balanced", max_iter=1000).fit(X, y)
    w = np.clip(clf.coef_[0], 0, None)          # Negative coefficients are truncated to0(monotonic prior)
    w = w / w.sum() if w.sum() > 0 else np.ones(4) / 4
    print("Fit weight (w1_iou, w2_prox, w3_flow, w4_occ):")
    print(",".join(f"{x:.3f}" for x in w))
    print(f"train AUC = {roc_auc_score(y, clf.decision_function(X)):.4f}  (n={len(y)})")

    if args.holdout:
        h = pd.read_parquet(args.holdout)
        s = h[FEATS].values @ w
        print(f"CHI3D holdout AUC = {roc_auc_score(h['contact_gt'].astype(int), s):.4f}  (n={len(h)})")


if __name__ == "__main__":
    main()
