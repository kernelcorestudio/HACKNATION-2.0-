# CARN (Cascading Residual Network) — Working & Architecture

## 1. Formulas & Mathematical Equations

CARN ka main architectural innovation **Multi-Level Cascading Mechanism** hai, jo dono Local (block ke andar) aur Global (pure network ke dauran) residual features ko aapas me interconnect karta hai.

### 1.1 Global Cascading Mechanism
Standard ResNets me har block agle block ko sequentially feed karta hai: $\mathbf{F}_k = f(\mathbf{F}_{k-1})$. Isme pehle wale blocks ke low-level details aakhiri layer tak aate-aate dilute ho jaate hain.

CARN me sabhi intermediate blocks ke features ko ek sath aage le jaya jata hai:
$$\mathbf{F}_1 = \mathcal{B}_1(\mathbf{F}_0)$$
$$\mathbf{F}_2 = \mathcal{B}_2(\mathbf{F}_1)$$
$$\mathbf{F}_3 = \mathcal{B}_3(\mathbf{F}_2)$$

Fir global feature representation sabhi states ko concatenate karke $1\times 1$ compression conv ke zariye banti hai:
$$\mathbf{F}_{\text{global}} = \mathbf{W}_{\text{compress}} * \left[ \mathbf{F}_0, \mathbf{F}_1, \mathbf{F}_2, \mathbf{F}_3 \right] + \mathbf{F}_0$$

- **$[\cdot]$:** Channel dimension ke along concatenation (e.g. $64 \times 4 = 256$ channels).
- **$\mathbf{W}_{\text{compress}} \in \mathbb{R}^{64 \times 256 \times 1 \times 1}$:** $1 \times 1$ bottleneck convolution jo channels ko wapas $64$ kar deta hai.
- **$\mathbf{F}_0$ Skip:** Long identity skip connection jo gradient vanishing ko rokta hai.

---

### 1.2 Local Cascading Block (CB) Formulation
Global cascade ki tarah hi, har ek **Cascading Block (CB)** ke andar teen **Local Residual Blocks (LRBs)** aapas me cascade hote hain:

$$\mathbf{B}_1 = f_{\text{LRB}}(\mathbf{F}_{\text{in}})$$
$$\mathbf{B}_2 = f_{\text{LRB}}(\mathbf{B}_1)$$
$$\mathbf{B}_3 = f_{\text{LRB}}\left( \mathbf{W}_{1\times 1} * [\mathbf{B}_1, \mathbf{B}_2] \right)$$

Final block output:
$$\mathbf{F}_{\text{out}} = \mathbf{W}_{\text{exit}} * [\mathbf{B}_1, \mathbf{B}_2, \mathbf{B}_3] + \mathbf{F}_{\text{in}}$$

- **Intuition:** Is local skip highway ki wajah se model shallow textures (edges) aur deep semantics (land cover context) ko ek hi block me fuse kar leta hai.

---

### 1.3 Charbonnier Loss Function (WorldStrat Training)
CARN ko ESA WorldStrat dataset par train karte waqt Charbonnier penalty use ki gayi thi:

$$\mathcal{L}_{\text{CARN}} = \sqrt{\|\mathbf{I}_{\text{SR}} - \mathbf{I}_{\text{HR}}\|^2 + \epsilon^2}, \quad \epsilon = 10^{-3}$$

- **Advantage over MSE ($L_2$ Loss):** $L_2$ loss outliers (jaise bright rooftop reflections) par overfit hokar blurriness introduce karta hai. Charbonnier loss robust $L_1$ behavior deta hai jo smooth aur radiometrically stable boundaries banata hai.

---

### 1.4 Group Convolution & Efficiency
FLOPs aur memory footprint reduce karne ke liye standard convolutions ko $G=4$ groups me divide kiya ja sakta hai:

$$\mathbf{y}_g = \mathbf{W}_g * \mathbf{x}_g, \quad g \in \{1, 2, 3, 4\}$$
$$\text{Computational Cost Reduction} = \frac{1}{G} = 75\% \text{ reduction in FLOPs}$$

---

## 2. Structure & Model Architecture

Text-based architectural flow:

```
[Input Tensor: 3 x 64 x 64] (Sentinel-2 10m GSD)
       │
       ▼
[Entry Conv Head: Conv2d(3, 64, 3x3)] ──────────────┐
       │                                            │
       ▼ F_0                                        │
┌──────────────────────────────────────────────┐    │
│ Cascading Block 1 (Local Cascading Residual) │ ─┐ │
└──────────────────────────────────────────────┘  │ │
       │ F_1                                      │ │
┌──────────────────────────────────────────────┐  │ │
│ Cascading Block 2 (Local Cascading Residual) │ ─┼─┤
└──────────────────────────────────────────────┘  │ │
       │ F_2                                      │ │
┌──────────────────────────────────────────────┐  │ │
│ Cascading Block 3 (Local Cascading Residual) │ ─┼─┤
└──────────────────────────────────────────────┘  │ │
       │ F_3                                      │ │
       ▼                                          │ │
[Global Concatenation: Cat(F_0, F_1, F_2, F_3)] ◄─┴─┘
  (256 Channels)
       │
       ▼
[1x1 Channel Compression Conv2d(256, 64, 1x1)]
       │
       ▼ (+) Global Identity Add: F_compress + F_0
       │
[Upsampling Stage 1: Conv2d(64, 256) + PixelShuffle(2x)]
       │ (128 x 128)
[Upsampling Stage 2: Conv2d(64, 256) + PixelShuffle(2x)]
       │ (256 x 256)
       ▼
[Exit Reconstruction Conv2d(64, 3, 3x3)]
       │
       ▼
[Output Super-Resolved Raster: 3 x 256 x 256] (2.5m GSD)
```

### Module Specifications Table

| Component | Layer Type | Channels | Kernel / Stride | Purpose |
| :--- | :--- | :---: | :---: | :--- |
| **Input Conv** | `nn.Conv2d` | $3 \to 64$ | $3 \times 3$, pad 1 | Feature tokenization. |
| **Cascading Blocks (x3)**| `CascadingBlock` | $64 \to 64$ | $3 \times 3$ + $1 \times 1$ | Multi-level hierarchical representations. |
| ↳ *LRBs (x3 per CB)* | `LocalResidualBlock` | $64 \to 64$ | $3 \times 3$, ReLU | Local feature extraction. |
| ↳ *Local 1x1 Convs* | `nn.Conv2d` | $128 \to 64, 192 \to 64$ | $1 \times 1$ | Local cascade compression. |
| **Global Bottleneck** | `nn.Conv2d` | $256 \to 64$ | $1 \times 1$ | Compresses all intermediate block outputs. |
| **Upsamplers** | `PixelShuffle(2)` $\times 2$ | $64 \to 256 \to 64$ | Sub-pixel Conv | $4\times$ spatial resolution increase. |
| **Final Exit** | `nn.Conv2d` | $64 \to 3$ | $3 \times 3$, pad 1 | 2.5m RGB radiance reconstruction. |

---

## 3. Workflow: Training Se Prediction Tak

```
[Training Workflow (ESA WorldStrat Protocol)]
Paired Global Dataset (Sentinel-2 10m LR + SPOT-6/7 1.5m Reference)
  │
  ├─► Patch Extraction: 64x64 non-overlapping patches
  │
  ├─► Random Geometric Augmentations: Horizontal & Vertical Flips, 90° Rotations
  │
  ├─► Forward Pass: CARN infers 256x256 super-resolved patch
  │
  ├─► Loss Optimization: Charbonnier Robust Penalty (eps = 1e-3)
  │
  ├─► Optimizer: Adam (lr = 1e-4, halved every 20 epochs)
  │
  └─► Saved Checkpoint: weights/evoland_carn_x4_worldstrat.pt (0.98M params, 3.75 MB)

[Prediction / Deployment Workflow]
Offline Field Survey / Edge Tablet
  │
  ├─► Step 1: Ingest Sentinel-2 L2A optical tile
  │
  ├─► Step 2: Normalize reflectance array to range [0.0, 1.0]
  │
  ├─► Step 3: Run single-pass deterministic inference through CARN engine (~420 ms)
  │
  ├─► Step 4: Rescale to radiometric DN (or export as GeoTIFF with original CRS)
  │
  └─► Step 5: Render 2.5m map on tablet display (< 195 MB RAM used)
```

---

## 4. Hyperparameters & Unka Effect

| Hyperparameter | Value | Description | Tune Karne Par Effect |
| :--- | :---: | :--- | :--- |
| **`num_features`** | `64` | Base channel count. | 64 channels sub-1M parameter budget aur feature richness ka sweet spot hai. 128 karne par RAM double ho jayegi. |
| **`num_blocks`** | `3` | Cascading Blocks ki sankhya. | 3 blocks rakhne se model deep aur lightweight dono rehta hai. 5 blocks karne se latency ~700ms ho jati hai. |
| **`scale_factor`** | `4` | Target upscaling factor. | 10m Sentinel-2 ko 2.5m GSD me map karta hai. |
| **`epsilon` ($\epsilon$)** | `1e-3` | Charbonnier loss smoothing factor. | Square root operation ko zero gradient singularity se bachata hai. |
| **`learning_rate`** | `1e-4` | Training step size. | Stable convergence ke liye standard learning rate. |
