# src/evaluate.py
import numpy as np
import pandas as pd
from typing import List
import os

def compute_topk_from_similarities(similarities_list, class_names: List[str], true_classes: List[str], topk=(1,5)):
    """
    similarities_list: list of numpy arrays (C,) in the same order as image_paths
    true_classes: list of true class names (same length)
    """
    rows = []
    for sims, true in zip(similarities_list, true_classes):
        # top indices (desc)
        idxs = np.argsort(-sims)
        top1_idx = idxs[0]
        top1 = class_names[top1_idx]
        topk_idx = idxs[:max(topk)]
        topk_names = [class_names[i] for i in topk_idx]
        rows.append({
            "true_class": true,
            "top1": top1,
            "top1_score": float(sims[top1_idx]),
            "top5": "|".join(topk_names[:5])
        })
    df = pd.DataFrame(rows)
    df["top1_hit"] = df["top1"] == df["true_class"]
    df["top5_hit"] = df.apply(lambda r: r["true_class"] in r["top5"].split("|"), axis=1)
    results = {
        "top1": df["top1_hit"].mean(),
        "top5": df["top5_hit"].mean(),
        "df": df
    }
    return results

def save_predictions_csv(df, out_path):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_csv(out_path, index=False)
