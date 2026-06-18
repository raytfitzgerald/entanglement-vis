#!/usr/bin/env python3
"""Export quantum kernel embeddings by re-running circuits at 3 parameter snapshots.

Loads run_*.json files from a results directory, re-runs the quantum circuit at
the 'initial', 'halfway', and 'end' parameter snapshots on all training images,
computes kernel matrices K[i,j] = |<psi(xi)|psi(xj)>|^2, averages across runs,
then applies UMAP/MDS for 2D visualization.

Requires kernel_compression_study.py to be importable (either on PYTHONPATH or
in the results directory).

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


def import_simulation(results_dir):
    """Import kernel_compression_study from the results directory or PYTHONPATH."""
    results_path = Path(results_dir).resolve()
    candidates = [results_path, results_path.parent, Path.cwd()]
    for p in candidates:
        if (p / "kernel_compression_study.py").exists():
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
            break

    try:
        import kernel_compression_study as kcs
        return kcs
    except ImportError:
        print("Error: cannot import kernel_compression_study.py", file=sys.stderr)
        print("Place it in the results directory, its parent, or on PYTHONPATH.", file=sys.stderr)
        sys.exit(1)


def compute_quantum_states(kcs, params, X):
    """Run the quantum circuit forward pass on all images, returning state vectors.

    Tries several common API patterns from kernel_compression_study.
    Returns array of shape (N, state_dim) with complex amplitudes.
    """
    if hasattr(kcs, "get_quantum_states"):
        return np.array(kcs.get_quantum_states(params, X))
    if hasattr(kcs, "circuit_forward"):
        return np.array([kcs.circuit_forward(params, x) for x in X])
    if hasattr(kcs, "get_state_vector"):
        return np.array([kcs.get_state_vector(params, x) for x in X])
    if hasattr(kcs, "kernel_circuit"):
        return np.array([kcs.kernel_circuit(params, x) for x in X])
    raise AttributeError(
        "kernel_compression_study has no recognized state-vector function. "
        "Expected one of: get_quantum_states, circuit_forward, get_state_vector, kernel_circuit"
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
    """Compute min(mean_pairwise_row_l1, mean_pairwise_col_l1)."""
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

    # Load dataset
    dataset_path = results_path / "dataset_train.npz"
    if not dataset_path.exists():
        print(f"Error: dataset_train.npz not found in {results_dir}", file=sys.stderr)
        return False

    dataset = np.load(dataset_path)
    X = dataset["X"]
    Y = dataset["Y"]
    num_samples = len(Y)
    print(f"  Loaded dataset: {num_samples} samples, images shape {X.shape[1:]}")

    # Compute h_scores from X
    h_scores = np.array([compute_h_score(img) for img in X])
    h_max = h_scores.max()
    if h_max > 0:
        h_scores = h_scores / h_max

    # Find run files
    run_files = sorted(glob.glob(str(results_path / "run_*.json")))
    if not run_files:
        print(f"Error: no run_*.json files found in {results_dir}", file=sys.stderr)
        return False

    num_runs = len(run_files)
    print(f"  Found {num_runs} run files")

    # Load all runs
    runs = []
    for rf in run_files:
        with open(rf) as f:
            runs.append(json.load(f))

    # Import simulation code and compute kernel matrices
    kcs = import_simulation(results_dir)

    snapshots_data = []
    for snap_name in SNAPSHOT_NAMES:
        print(f"  Computing kernel for snapshot '{snap_name}'...")
        K_accum = np.zeros((num_samples, num_samples))
        valid_runs = 0

        for run_idx, run in enumerate(runs):
            params_snapshots = run.get("params_snapshots", {})
            if snap_name not in params_snapshots:
                print(f"    Warning: run {run_idx+1} missing '{snap_name}' snapshot, skipping", file=sys.stderr)
                continue

            params = np.array(params_snapshots[snap_name])
            states = compute_quantum_states(kcs, params, X)
            K = compute_kernel_matrix(states)
            K_accum += K
            valid_runs += 1

        if valid_runs == 0:
            print(f"    Error: no valid runs for snapshot '{snap_name}'", file=sys.stderr)
            continue

        K_avg = K_accum / valid_runs
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
        print(f"    Done ({valid_runs} runs averaged, {len(points)} points)")

    if not snapshots_data:
        print(f"Error: no valid snapshots produced for {results_dir}", file=sys.stderr)
        return False

    # Aggregate metrics across runs (full 200-step curves)
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

    # Reference states (from first run that has them)
    reference_states = None
    for run in runs:
        if "reference_states" in run:
            ref = run["reference_states"]
            reference_states = {}
            for key in ["bas", "non_bas", "BAS", "non_BAS"]:
                if key in ref:
                    state = ref[key]
                    if isinstance(state, dict):
                        reference_states[key.lower().replace("-", "_")] = {
                            "real": [round(float(v), 6) for v in state.get("real", state.get("re", []))],
                            "imag": [round(float(v), 6) for v in state.get("imag", state.get("im", []))],
                        }
                    elif isinstance(state, list):
                        arr = np.array(state)
                        if np.iscomplex(arr).any():
                            reference_states[key.lower().replace("-", "_")] = {
                                "real": [round(float(v), 6) for v in arr.real],
                                "imag": [round(float(v), 6) for v in arr.imag],
                            }
                        else:
                            reference_states[key.lower().replace("-", "_")] = {
                                "real": [round(float(v), 6) for v in arr],
                                "imag": [0.0] * len(arr),
                            }
            if reference_states:
                break

    # Build metadata
    folder_name = results_path.name
    metadata = parse_folder_name(folder_name)
    metadata["results_dir"] = str(results_path)
    metadata["num_samples"] = num_samples
    metadata["num_runs"] = num_runs

    meta_files = list(results_path.glob("metadata*.json")) + list(results_path.glob("config*.json"))
    if meta_files:
        with open(meta_files[0]) as f:
            file_meta = json.load(f)
            for k, v in file_meta.items():
                if k not in metadata:
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
            if child.is_dir() and not child.name.startswith("."):
                print(f"\nProcessing: {child.name}")
                if process_results_dir(child):
                    count += 1

        print(f"\nDone. Processed {count} result folders.")


if __name__ == "__main__":
    main()
