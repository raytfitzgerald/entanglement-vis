#!/usr/bin/env python3
"""Export quantum kernel embeddings by re-running circuits at 3 parameter snapshots.

Re-runs the quantum circuit at 'initial', 'halfway', and 'end' parameter snapshots
from run_*.json on all training images from dataset_train.npz, computes kernel
matrices K[i,j] = |<psi(xi)|psi(xj)>|^2 averaged across runs, then applies
UMAP/MDS for 2D visualization.

Requires kernel_compression_study.py to be importable (place it in the same
directory as this script, or on PYTHONPATH).

Usage:
    python export_kernel_umap.py --results-dir path/to/results/folder
    python export_kernel_umap.py --batch path/to/parent/folder
"""

import argparse
import glob
import json
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

SNAPSHOT_NAMES = ["initial", "halfway", "end"]
SNAPSHOT_STEPS = {"initial": 0, "halfway": 100, "end": 200}

_kcs = None


def import_simulation():
    """Import kernel_compression_study from PYTHONPATH or this script's directory."""
    global _kcs
    if _kcs is not None:
        return _kcs

    script_dir = str(Path(__file__).resolve().parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    try:
        import kernel_compression_study as kcs
        _kcs = kcs
        return kcs
    except ImportError:
        return None


def compute_quantum_states(kcs, params_snapshot, X, config):
    """Run the quantum circuit forward pass on all images, returning state vectors.

    params_snapshot is a dict with keys 'embedding' and 'ansatz'.
    Tries several API patterns from kernel_compression_study.
    Returns array of shape (N, state_dim) with complex amplitudes.
    """
    embedding_params = np.array(params_snapshot.get("embedding", []))
    ansatz_params = np.array(params_snapshot.get("ansatz", []))

    if hasattr(kcs, "get_quantum_states"):
        return np.array(kcs.get_quantum_states(
            embedding_params, ansatz_params, X, config
        ))
    if hasattr(kcs, "compute_states"):
        return np.array(kcs.compute_states(
            embedding_params, ansatz_params, X, config
        ))
    if hasattr(kcs, "circuit_forward"):
        return np.array([
            kcs.circuit_forward(embedding_params, ansatz_params, x, config) for x in X
        ])
    if hasattr(kcs, "get_state_vector"):
        return np.array([
            kcs.get_state_vector(embedding_params, ansatz_params, x, config) for x in X
        ])

    all_params = np.concatenate([embedding_params, ansatz_params]) if len(embedding_params) else ansatz_params
    if hasattr(kcs, "get_quantum_states"):
        return np.array(kcs.get_quantum_states(all_params, X, config))
    if hasattr(kcs, "circuit_forward"):
        return np.array([kcs.circuit_forward(all_params, x, config) for x in X])

    raise AttributeError(
        "kernel_compression_study has no recognized state-vector function. "
        "Expected one of: get_quantum_states, compute_states, circuit_forward, get_state_vector"
    )


def compute_kernel_matrix(states):
    """Compute K[i,j] = |<psi(xi)|psi(xj)>|^2 from an array of state vectors."""
    inner = states @ states.conj().T
    return np.abs(inner) ** 2


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
    """Compute min(mean_pairwise_row_l1, mean_pairwise_col_l1) on a 2D image."""
    if image is None:
        return 0.0
    img = image.astype(float)
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

    # Match both ASCII 'x' and unicode '×'
    size_match = re.search(r"(\d+)[x×](\d+)", folder_name)
    if size_match:
        meta["image_size"] = f"{size_match.group(1)}x{size_match.group(2)}"

    kernel_match = re.search(r"(\d+)[x×](\d+)[x×](\d+)\s*kernels?", folder_name, re.IGNORECASE)
    if kernel_match:
        meta["kernel_size"] = f"{kernel_match.group(1)}x{kernel_match.group(2)}x{kernel_match.group(3)}"

    noise_match = re.search(r"noise[=_]?([\d.]+)", folder_name, re.IGNORECASE)
    if noise_match:
        meta["noise"] = float(noise_match.group(1))

    for cohort in ["EE", "EM", "MM"]:
        if cohort in folder_name:
            meta["cohort"] = cohort
            break

    label_match = re.search(r"^[\d_]+_(.*?):", folder_name)
    if label_match:
        meta["label"] = label_match.group(1).strip()

    return meta


def process_results_dir(results_dir):
    """Process a single results directory and produce kernel_umap.json."""
    results_path = Path(results_dir)
    if not results_path.is_dir():
        print(f"Error: {results_dir} is not a directory", file=sys.stderr)
        return False

    # Load dataset
    dataset_path = results_path / "dataset_train.npz"
    if not dataset_path.exists():
        print(f"Error: dataset_train.npz not found in {results_dir}", file=sys.stderr)
        return False

    dataset = np.load(dataset_path)
    X = dataset["X"]  # shape (num_samples, image_size, image_size)
    Y = dataset["Y"]  # shape (num_samples,) with values {0, 1}
    num_samples = len(Y)
    image_size = int(dataset["image_size"]) if "image_size" in dataset else X.shape[1]
    print(f"  Loaded dataset: {num_samples} samples, image size {image_size}x{image_size}")

    # Compute h_scores from X
    h_scores_raw = np.array([compute_h_score(img) for img in X])
    h_max = h_scores_raw.max()
    h_scores = h_scores_raw / h_max if h_max > 0 else h_scores_raw

    # Find run files
    run_files = sorted(glob.glob(str(results_path / "run_*.json")))
    # Filter out params_snapshots companion files
    run_files = [f for f in run_files if "params_snapshots" not in f]
    if not run_files:
        print(f"Error: no run_*.json files found in {results_dir}", file=sys.stderr)
        return False

    num_runs = len(run_files)
    print(f"  Found {num_runs} run files")

    runs = []
    for rf in run_files:
        with open(rf) as f:
            runs.append(json.load(f))

    # Get config from first run
    config = runs[0].get("config", {})

    # Try to import simulation code for kernel re-computation
    kcs = import_simulation()
    can_recompute = kcs is not None

    if not can_recompute:
        print("  Warning: kernel_compression_study.py not found.", file=sys.stderr)
        print("  Place it next to this script or on PYTHONPATH to enable kernel re-computation.", file=sys.stderr)
        print("  Falling back to identity kernel (points will lack meaningful spatial structure).", file=sys.stderr)

    # Compute kernel matrices at each snapshot
    snapshots_data = []
    for snap_name in SNAPSHOT_NAMES:
        print(f"  Computing snapshot '{snap_name}' (step {SNAPSHOT_STEPS[snap_name]})...")

        if can_recompute:
            K_accum = np.zeros((num_samples, num_samples))
            valid_runs = 0

            for run_idx, run in enumerate(runs):
                params_snapshots = run.get("params_snapshots", {})
                if snap_name not in params_snapshots:
                    print(f"    Warning: run {run_idx+1} missing '{snap_name}' snapshot", file=sys.stderr)
                    continue

                snap_params = params_snapshots[snap_name]
                # snap_params is {"embedding": [...], "ansatz": [...]}
                try:
                    states = compute_quantum_states(kcs, snap_params, X, config)
                    K = compute_kernel_matrix(states)
                    K_accum += K
                    valid_runs += 1
                except Exception as e:
                    print(f"    Error in run {run_idx+1}: {e}", file=sys.stderr)
                    continue

            if valid_runs == 0:
                print(f"    No valid runs for '{snap_name}', skipping", file=sys.stderr)
                continue

            K_avg = K_accum / valid_runs
        else:
            # Fallback: use identity-like kernel (visualization will be random-ish)
            K_avg = np.eye(num_samples) * 0.5 + 0.5 * np.random.RandomState(
                SNAPSHOT_STEPS[snap_name]
            ).rand(num_samples, num_samples)
            K_avg = (K_avg + K_avg.T) / 2
            np.fill_diagonal(K_avg, 1.0)

        coords = embed_2d(K_avg)

        points = []
        for j in range(num_samples):
            points.append({
                "id": j,
                "x": round(float(coords[j, 0]), 4),
                "y": round(float(coords[j, 1]), 4),
                "label": int(Y[j]),
                "h_score": round(float(h_scores[j]), 4),
            })

        snapshots_data.append({
            "name": snap_name,
            "step": SNAPSHOT_STEPS[snap_name],
            "points": points,
        })
        print(f"    Done ({len(points)} points" + (f", {valid_runs} runs averaged" if can_recompute else "") + ")")

    if not snapshots_data:
        print(f"Error: no valid snapshots produced", file=sys.stderr)
        return False

    # Aggregate full training curves across runs (200 steps)
    all_costs = []
    all_accs = []
    all_entropy_bas = []
    all_entropy_non_bas = []

    for run in runs:
        if "costs" in run:
            all_costs.append(run["costs"])
        if "accuracies" in run:
            all_accs.append(run["accuracies"])
        if "entanglement_entropy_history_bas" in run:
            all_entropy_bas.append(run["entanglement_entropy_history_bas"])
        if "entanglement_entropy_history_not_bas" in run:
            all_entropy_non_bas.append(run["entanglement_entropy_history_not_bas"])

    def avg_lists(lists):
        if not lists:
            return []
        max_len = max(len(l) for l in lists)
        result = []
        for i in range(max_len):
            vals = [l[i] for l in lists if i < len(l)]
            result.append(round(sum(vals) / len(vals), 6))
        return result

    metrics = {
        "steps": list(range(len(avg_lists(all_costs)) if all_costs else 0)),
        "costs": avg_lists(all_costs),
        "accuracies": avg_lists(all_accs),
        "entropy_bas": avg_lists(all_entropy_bas),
        "entropy_non_bas": avg_lists(all_entropy_non_bas),
    }

    # Reference states: keys are "bas" and "not_bas", each has "real", "imag", "probabilities"
    reference_states = None
    for run in runs:
        if "reference_states" in run:
            ref = run["reference_states"]
            reference_states = {}
            for key in ["bas", "not_bas"]:
                if key in ref:
                    state = ref[key]
                    reference_states[key] = {
                        "real": [round(float(v), 6) for v in state.get("real", [])],
                        "imag": [round(float(v), 6) for v in state.get("imag", [])],
                    }
                    if "probabilities" in state:
                        reference_states[key]["probabilities"] = [
                            round(float(v), 6) for v in state["probabilities"]
                        ]
            if reference_states:
                break

    # Build metadata
    folder_name = results_path.name
    metadata = parse_folder_name(folder_name)
    metadata["results_dir"] = str(results_path)
    metadata["num_samples"] = num_samples
    metadata["num_runs"] = num_runs
    metadata["image_size"] = f"{image_size}x{image_size}"

    # Merge config from run files
    for key in ["num_layers", "kernel_size", "noise", "use_cnot", "use_u2",
                 "entanglement", "encoding_3x3"]:
        if key in config and key not in metadata:
            metadata[key] = config[key]

    # Merge any top-level metadata/config files
    meta_files = list(results_path.glob("metadata*.json")) + list(results_path.glob("config*.json"))
    for mf in meta_files:
        with open(mf) as f:
            file_meta = json.load(f)
            for k, v in file_meta.items():
                if k not in metadata and k != "study_config":
                    metadata[k] = v

    # Write output
    output = {
        "metadata": metadata,
        "snapshots": snapshots_data,
        "metrics": metrics,
    }
    if reference_states:
        output["reference_states"] = reference_states

    out_path = results_path / "kernel_umap.json"
    with open(out_path, "w") as f:
        json.dump(output, f)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"  Wrote {out_path} ({size_mb:.1f} MB, {len(snapshots_data)} snapshots)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Export quantum kernel embeddings from parameter snapshots")
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
            if child.is_dir() and not child.name.startswith((".","_")):
                print(f"\nProcessing: {child.name}")
                if process_results_dir(child):
                    count += 1

        print(f"\nDone. Processed {count} result folders.")


if __name__ == "__main__":
    main()
