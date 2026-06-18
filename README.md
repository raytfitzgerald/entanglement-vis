# Quantum Entanglement Geometry Visualizer

An interactive 3D visualization of how quantum entanglement looks geometrically in the state space of a two-qubit system.

## What This Shows

A two-qubit quantum state with real amplitudes lives on **S³** (a 3-sphere in 4D). This tool projects that space into 3D using stereographic projection, revealing the geometric structure of entanglement:

- **The blue surface** is the **Segre variety** — the set of all product (separable) states. Any state on this surface has zero entanglement.
- **Entangled states exist off this surface.** The further a state is from the surface, the more entangled it is.
- **The red path** is the **S³ geodesic** — the shortest possible path between two states, which may pass through entangled territory.
- **The orange path** is the **separable geodesic** — the shortest path that stays on the product state surface (only shown when both endpoints are separable).

### Key Insight

Entanglement provides a geometric shortcut. For example, going from |00⟩ to |11⟩:
- The shortest separable path (staying on the surface) has length **π/√2 ≈ 2.22**
- The shortest path through entangled space (via the Bell state) has length **π/2 ≈ 1.57**
- That's a **29% shorter** route — but it requires passing through maximum entanglement

### What the Bloch Spheres Show

Each qubit has a Bloch vector representing its individual state. When qubits are entangled, their individual Bloch vectors **shrink** — at maximum entanglement (Bell state), the vectors collapse to zero length. This illustrates how entanglement moves information from local qubits to global correlations.

## Live Demo

**[Open the visualizer](https://raytfitzgerald.github.io/entanglement-vis/)**

## Run Locally

No build step, no dependencies to install. It's a single HTML file that loads Three.js from a CDN.

### Option 1: Python (built-in on macOS/Linux)

```bash
git clone https://github.com/raytfitzgerald/entanglement-vis.git
cd entanglement-vis
python3 -m http.server 8090
```

Then open [http://localhost:8090](http://localhost:8090)

### Option 2: Node.js

```bash
git clone https://github.com/raytfitzgerald/entanglement-vis.git
cd entanglement-vis
npx serve .
```

### Option 3: Just open the file

Most browsers will work if you just open `index.html` directly, but some may block the Three.js CDN import due to CORS. A local server is more reliable.

### Access from other devices on your network

Start the server bound to all interfaces:

```bash
python3 -m http.server 8090 --bind 0.0.0.0
```

Then open `http://<your-local-ip>:8090` from any device on the same Wi-Fi.

## Controls

| Control | Description |
|---------|-------------|
| **Start / End** dropdowns | Choose preset states (|00⟩, |01⟩, |10⟩, |11⟩, |+0⟩, |0+⟩, |++⟩, Bell Φ⁺) or Random |
| **Random Both** | Pick two uniformly random states on S³ |
| **Path** dropdown | Switch between S³ geodesic, separable geodesic, or both |
| **Play / Pause** | Animate the state moving along the selected path |
| **Reset** | Return to the start state |
| **Speed** slider | Control animation speed |
| **Click + drag** | Rotate the 3D view |
| **Scroll** | Zoom in/out |

## What You're Seeing

| Visual Element | Meaning |
|----------------|---------|
| Blue translucent surface | All product (separable) states — the Segre variety |
| Red tube | S³ geodesic — shortest path in full state space |
| Orange tube | Separable geodesic — shortest path staying on the surface |
| Glowing sphere | Current state position |
| Dashed white line | Distance from current state to nearest product state (= entanglement) |
| Green dot (START) | Starting state |
| Pink dot (END) | Ending state |
| Bloch spheres | Individual qubit states — vector length shows purity |
| Concurrence graph | Entanglement measure (0 = separable, 1 = maximally entangled) along the path |
| Distance panel | Arc lengths on S³ comparing the two routes |

## The Math

For those interested in the details:

- **State space**: A 2-qubit state with real amplitudes is |ψ⟩ = a|00⟩ + b|01⟩ + c|10⟩ + d|11⟩ where a² + b² + c² + d² = 1. This is S³ ⊂ R⁴.
- **Stereographic projection**: S³ → R³ from the south pole (−1,0,0,0): (a,b,c,d) ↦ (b,c,d)/(1+a)
- **Product states**: States where the coefficient matrix [[a,b],[c,d]] has rank 1, i.e., ad − bc = 0. These form the Segre embedding of S¹ × S¹ → S³.
- **Concurrence**: C = 2|ad − bc|, ranging from 0 (separable) to 1 (maximally entangled).
- **S³ geodesic**: Great circle arc (SLERP) between two points on S³. Length = arccos(|u·v|).
- **Separable geodesic**: The product state surface is a flat torus (metric ds² = dθ₁² + dθ₂²), so geodesics are straight lines in the (θ₁, θ₂) parameter space.
- **Bloch vectors**: Computed from the reduced density matrix of each qubit. Length = 1 for pure (product) states, < 1 for mixed (entangled) states.

## Tab 2: Kernel Space

The second tab visualizes how quantum kernel states cluster during training. It shows a 2D scatter plot of UMAP-embedded kernel matrix entries, animated across training epochs.

### Two-Step Workflow

**Step 1: Export your data (Python)**

```bash
pip install numpy scikit-learn umap-learn  # umap-learn is optional, falls back to MDS
python scripts/export_kernel_umap.py --results-dir path/to/results/folder
```

This reads quantum kernel matrices from checkpoint directories and exports a `kernel_umap.json` file. To process all result folders under a parent directory at once:

```bash
python scripts/export_kernel_umap.py --batch path/to/parent/folder
```

**Step 2: Load in the browser**

Open the site, click the **Kernel Space** tab, and either drag-and-drop your `kernel_umap.json` onto the drop zone or click to browse.

### Kernel Space Controls

| Control | Description |
|---------|-------------|
| **Load data** | Drag-and-drop or file picker for `kernel_umap.json` |
| **Study selector** | Switch between multiple loaded studies |
| **Epoch slider** | Scrub through training epochs with step/cost/accuracy readouts |
| **Play / Pause** | Auto-advance through epochs (configurable 1–10 fps) |
| **Color mode** | Color points by label (BAS=orange, non-BAS=blue) or by h_score (viridis colormap) |
| **H-score threshold** | Show a threshold indicator on the scatter plot |
| **Overlay mode** | Plot two studies in the same scatter space (Procrustes alignment) |

### Metrics

- **Cluster distance**: Euclidean distance between BAS and non-BAS centroids in UMAP space
- **Silhouette score**: Cluster quality measure (−1 to 1), shown as a colored badge
- **Epoch strip**: Bottom line chart showing accuracy and cost vs epoch; click to jump

### Expected JSON Format

```json
{
  "metadata": {
    "num_samples": 200,
    "num_layers": 5,
    "image_size": "10x10"
  },
  "epochs": [
    {
      "epoch": 0,
      "step": 0,
      "cost": 0.693,
      "accuracy": 0.51,
      "points": [
        {"id": 0, "x": 1.23, "y": -0.45, "label": 1, "h_score": 0.0},
        ...
      ]
    }
  ]
}
```

## License

MIT
