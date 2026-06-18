#!/usr/bin/env python3
"""Export quantum kernel matrices as 2D UMAP/MDS embeddings for browser visualization.

Usage:
    python export_kernel_umap.py --results-dir path/to/results/folder
    python export_kernel_umap.py --batch path/to/parent/folder
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

from sklearn.manifold import MDS


def load_checkpoint(ckpt_dir):
    """Load kernel matrix and labels from a checkpoint directory."""
    ckpt_path = Path(ckpt_dir)

    kernel = None
    for name in ["kernel_matrix.npy", "K.npy", "kernel.npy"]:
        p = ckpt_path / name
        if p.exists():
            kernel = np.load(p)
            break

    if kernel is None:
        npz_files = list(ckpt_path.glob("*.npz"))
        for npz in npz_files:
            data = np.load(npz)
            for key in data.files:
                if "kernel" in key.lower() or key == "K":
                    kernel = data[key]
                    break
            if kernel is not None:
                break

    if kernel is None:
        return None, None, {}

    labels = None
    for name in ["labels.npy", "y.npy", "y_train.npy"]:
        p = ckpt_path / name
        if p.exists():
            labels = np.load(p)
            break

    extras = {}
    for name in ["metrics.json", "info.json"]:
        p = ckpt_path / name
        if p.exists():
            with open(p) as f:
                extras = json.load(f)
            break

    return kernel, labels, extras


def embed_2d(kernel_matrix):
    """Convert kernel matrix to 2D embedding via UMAP or MDS."""
    D = 1.0 - kernel_matrix
    np.fill_diagonal(D, 0.0)
    D = np.clip(D, 0, None)
    D = (D + D.T) / 2.0

    if HAS_UMAP:
        reducer = UMAP(n_components=2, metric="precomputed", random_state=42)
        coords = reducer.fit_transform(D)
    else:
        reducer = MDS(n_components=2, dissimilarity="precomputed", random_state=42, normalized_stress="auto")
        coords = reducer.fit_transform(D)

    return coords


def compute_h_score(image):
    """Compute bas_global_stripe_heterogeneity = min(mean_pairwise_row_l1, mean_pairwise_col_l1)."""
    if image is None:
        return 0.0
    img = image.astype(float)
    n_rows, n_cols = img.shape[:2]

    if len(img.shape) == 3:
        img = img.mean(axis=2)

    row_means = img.mean(axis=1)
    col_means = img.mean(axis=0)

    if len(row_means) > 1:
        row_diffs = np.abs(np.subtract.outer(row_means, row_means))
        mean_row_l1 = row_diffs.sum() / (len(row_means) * (len(row_means) - 1))
    else:
        mean_row_l1 = 0.0

    if len(col_means) > 1:
        col_diffs = np.abs(np.subtract.outer(col_means, col_means))
        mean_col_l1 = col_diffs.sum() / (len(col_means) * (len(col_means) - 1))
    else:
        mean_col_l1 = 0.0

    return float(min(mean_row_l1, mean_col_l1))


def parse_folder_name(folder_name):
    """Extract metadata from the results folder name."""
    meta = {}
    layers_match = re.search(r"(\d+)\s*layers?", folder_name, re.IGNORECASE)
    if layers_match:
        meta["num_layers"] = int(layers_match.group(1))

    samples_match = re.search(r"(\d+)\s*samples?", folder_name, re.IGNORECASE)
    if samples_match:
        meta["num_samples"] = int(samples_match.group(1))

    size_match = re.search(r"(\d+)x(\d+)", folder_name)
    if size_match:
        meta["image_size"] = f"{size_match.group(1)}x{size_match.group(2)}"

    for cohort in ["EE", "EM", "MM"]:
        if cohort in folder_name:
            meta["cohort"] = cohort
            break

    for noise in ["Binary", "Gaussian", "None"]:
        if noise.lower() in folder_name.lower():
            meta["noise_type"] = noise
            break

    return meta


def process_results_dir(results_dir):
    """Process a single results directory and produce kernel_umap.json."""
    results_path = Path(results_dir)
    if not results_path.is_dir():
        print(f"Error: {results_dir} is not a directory", file=sys.stderr)
        return False

    folder_name = results_path.name
    metadata = parse_folder_name(folder_name)
    metadata["results_dir"] = str(results_path)

    meta_files = list(results_path.glob("metadata*.json")) + list(results_path.glob("config*.json"))
    if meta_files:
        with open(meta_files[0]) as f:
            file_meta = json.load(f)
            for k, v in file_meta.items():
                if k not in metadata:
                    metadata[k] = v

    images = None
    for name in ["images.npy", "X.npy", "X_train.npy"]:
        p = results_path / name
        if p.exists():
            images = np.load(p)
            break

    h_scores = None
    if images is not None:
        h_scores = np.array([compute_h_score(img) for img in images])
        h_max = h_scores.max()
        if h_max > 0:
            h_scores = h_scores / h_max

    ckpt_dirs = sorted(
        [d for d in results_path.iterdir() if d.is_dir() and any(d.glob("*.npy")) or any(d.glob("*.npz"))],
        key=lambda d: d.name,
    )

    if not ckpt_dirs:
        print(f"No checkpoint directories found in {results_dir}", file=sys.stderr)
        return False

    shared_labels = None
    for name in ["labels.npy", "y.npy", "y_train.npy"]:
        p = results_path / name
        if p.exists():
            shared_labels = np.load(p)
            break

    epochs_data = []
    for i, ckpt_dir in enumerate(ckpt_dirs):
        kernel, labels, extras = load_checkpoint(ckpt_dir)
        if kernel is None:
            continue

        if labels is None:
            labels = shared_labels

        if labels is None:
            print(f"Warning: no labels found for {ckpt_dir.name}, using zeros", file=sys.stderr)
            labels = np.zeros(kernel.shape[0], dtype=int)

        if "num_samples" not in metadata:
            metadata["num_samples"] = int(kernel.shape[0])

        coords = embed_2d(kernel)

        epoch_num = i
        step_match = re.search(r"(\d+)", ckpt_dir.name)
        if step_match:
            epoch_num = int(step_match.group(1))

        points = []
        for j in range(coords.shape[0]):
            pt = {
                "id": j,
                "x": round(float(coords[j, 0]), 4),
                "y": round(float(coords[j, 1]), 4),
                "label": int(labels[j]),
            }
            if h_scores is not None and j < len(h_scores):
                pt["h_score"] = round(float(h_scores[j]), 4)
            else:
                pt["h_score"] = 0.0
            points.append(pt)

        epoch_entry = {
            "epoch": epoch_num,
            "step": epoch_num,
            "cost": extras.get("cost", extras.get("loss", 0.0)),
            "accuracy": extras.get("accuracy", extras.get("acc", 0.0)),
            "points": points,
        }
        epochs_data.append(epoch_entry)
        print(f"  Processed checkpoint {ckpt_dir.name} ({len(points)} points)")

    if not epochs_data:
        print(f"No valid checkpoints in {results_dir}", file=sys.stderr)
        return False

    output = {"metadata": metadata, "epochs": epochs_data}
    out_path = results_path / "kernel_umap.json"
    with open(out_path, "w") as f:
        json.dump(output, f)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {out_path} ({size_mb:.1f} MB, {len(epochs_data)} epochs)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Export quantum kernel matrices as UMAP/MDS embeddings")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--results-dir", help="Path to a single results folder")
    group.add_argument("--batch", help="Path to parent folder; processes all result folders within")
    args = parser.parse_args()

    if not HAS_UMAP:
        print("Note: umap-learn not installed, falling back to sklearn MDS", file=sys.stderr)

    if args.results_dir:
        success = process_results_dir(args.results_dir)
        sys.exit(0 if success else 1)
    else:
        parent = Path(args.batch)
        if not parent.is_dir():
            print(f"Error: {args.batch} is not a directory", file=sys.stderr)
            sys.exit(1)

        count = 0
        for child in sorted(parent.iterdir()):
            if child.is_dir() and not child.name.startswith("."):
                print(f"\nProcessing: {child.name}")
                if process_results_dir(child):
                    count += 1

        print(f"\nDone. Processed {count} result folders.")


if __name__ == "__main__":
    main()
