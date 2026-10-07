# SwinIR (Shifted Window Transformer) — Working & Architecture

## 1. Formulas & Mathematical Equations

SwinIR ka foundation **Shifted Window Multi-Head Self-Attention (SW-MSA)** aur **Residual Swin Transformer Blocks (RSTB)** par tika hua hai.

### 1.1 Shifted Window Mechanism
Consecutive layers me attention calculate karne ka tarika badalta hai:
- **Layer $l$ (Standard W-MSA):** Regular non-overlapping $M \times M$ ($8 \times 8$) windows me attention nikalta hai.
- **Layer $l+1$ (Shifted SW-MSA):** Windows ko $(\lfloor \frac{M}{2} \rfloor, \lfloor \frac{M}{2} \rfloor) = (4, 4)$ pixels se shift kar diya jata hai.

$$\hat{\mathbf{z}}^l = \text{W-MSA}(\text{LN}(\mathbf{z}^{l-1})) + \mathbf{z}^{l-1}$$
$$\mathbf{z}^l = \text{MLP}(\text{LN}(\hat{\mathbf{z}}^l)) + \hat{\mathbf{z}}^l$$
$$\hat{\mathbf{z}}^{l+1} = \text{SW-MSA}(\text{LN}(\mathbf{z}^l)) + \mathbf{z}^l$$
$$\mathbf{z}^{l+1} = \text{MLP}(\text{LN}(\hat{\mathbf{z}}^{l+1})) + \hat{\mathbf{z}}^{l+1}$$

- **Intuition:** Window shift karne se pichli layer ki do alag-alag windows ke border pixels ek nayi window ke andar aa jate hain, jisse cross-window communication possible ho jata hai.

---

### 1.2 Self-Attention with Continuous Relative Position Bias
Query ($\mathbf{Q}$), Key ($\mathbf{K}$), Value ($\mathbf{V}$) matrices ke beech attention formula:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Softmax}\left(\frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d}} + \mathbf{B}\right) \mathbf{V}$$

- **$d = \frac{C}{\text{num\_heads}}$:** Head dimension.
- **$\mathbf{B} \in \mathbb{R}^{M^2 \times M^2}$:** Learned Relative Position Bias jo coordinates $[-M+1, M-1]$ se sample hota hai. Yeh spatial 2D geometry aur distance preserve karta hai.

---

### 1.3 Residual Swin Transformer Block (RSTB)
Har RSTB ke andar multiple Swin Transformer Layers (STL) hote hain, aur end me ek convolutional layer inductive bias add karti hai:

$$\mathbf{F}_{i, 0} = \mathbf{F}_{i-1}$$
$$\mathbf{F}_{i, j} = \text{STL}_{i, j}(\mathbf{F}_{i, j-1}), \quad j = 1, \dots, L$$
$$\mathbf{F}_i = \mathbf{W}_i * \mathbf{F}_{i, L} + \mathbf{F}_{i, 0}$$

- **$\mathbf{W}_i$:** $3 \times 3$ convolutional layer jo transformer features ko spatial continuity deti hai.
- **Long Skip Connection:** Residual identity addition $\mathbf{F}_{i, 0}$ gradient propagation ko stable banata hai.

---

### 1.4 Charbonnier Training Loss Function
$$\mathcal{L}_{\text{SwinIR}} = \sqrt{\|\mathbf{I}_{\text{SR}} - \mathbf{I}_{\text{HR}}\|^2 + \epsilon^2}, \quad \epsilon = 10^{-3}$$

---

## 2. Structure & Model Architecture

Text-based architectural pipeline:

```
[Input Tensor: 3 x 64 x 64] (Sentinel-2 10m GSD)
       │
       ▼
[Shallow Feature Extraction Conv2d(3, 180, 3x3)] ─────────────┐
       │                                                      │
       ▼                                                      │
[Deep Feature Extraction: 6x RSTB Blocks]                     │
  ┌────────────────────────────────────────────────────────┐  │
  │ • RSTB 1                                               │  │
  │   - Swin Transformer Layer (W-MSA 8x8)                 │  │
  │   - Swin Transformer Layer (SW-MSA Shift 4)            │  │
  │   - Swin Transformer Layer (W-MSA 8x8)                 │  │
  │   - Swin Transformer Layer (SW-MSA Shift 4)            │  │
  │   - Conv2d(180, 180, 3x3) + Residual Skip (+)          │  │
  │ • RSTB 2 ... RSTB 6                                    │  │
  └────────────────────────────────────────────────────────┘  │
       │                                                      │
       ▼                                                      │
[Trunk Conv2d(180, 180, 3x3)]                                 │
       │                                                      │
       ▼ (+) Global Long Residual ◄───────────────────────────┘
       │
[High-Quality Reconstruction Module]
  • Conv2d(180, 720) + PixelShuffle(2x)  --> (180 x 128 x 128)
  • Conv2d(180, 720) + PixelShuffle(2x)  --> (180 x 256 x 256)
  • Conv2d(180, 3, 3x3)
       │
       ▼
[Super-Resolved Output Image: 3 x 256 x 256] (2.5m GSD)
```

### Module Specifications Table (SwinIR-M)

| Component | Layer Type | Channels | Window / Shift | Purpose |
| :--- | :--- | :---: | :---: | :--- |
| **Shallow Head** | `nn.Conv2d` | $3 \to 180$ | $3 \times 3$, pad 1 | Projects input into 180-dim token space. |
| **RSTB Blocks (x6)**| `ResidualSwinTransformerBlock` | $180 \to 180$ | $8 \times 8$ | Cascaded deep hierarchical transformer blocks. |
| ↳ *STL Layers* | `SwinTransformerBlock` | $180 \to 180$ | Shift = 0 / 4 | Alternates local window and shifted window attention. |
| ↳ *Block Conv* | `nn.Conv2d` | $180 \to 180$ | $3 \times 3$ | Injects translational invariance into tokens. |
| **Trunk Conv** | `nn.Conv2d` | $180 \to 180$ | $3 \times 3$ | Aggregates all deep transformer features. |
| **Upsampler** | `PixelShuffle(2)` $\times 2$ | $180 \to 3$ | Sub-pixel conv | $4\times$ spatial resolution upscaling. |

---

## 3. Workflow: Training Se Prediction Tak

```
[Training Workflow]
Paired Remote Sensing Data (Sentinel-2 10m LR + SPOT-6/7 1.5m Reference)
  │
  ├─► Patch Extraction: 64x64 input patches (target 256x256)
  │
  ├─► Forward Pass: SwinIR executes tokenization, 6 RSTBs, and 4x PixelShuffle
  │
  ├─► Loss Computation: Charbonnier Loss vs High-Res Ground Truth
  │
  ├─► Optimizer: Adam (lr = 2e-4, Cosine Decay schedule)
  │
  └─► Saved Checkpoint: weights/swinir_x4_satellite.pt (11.9M params, 67 MB)

[Prediction Workflow]
AOI Tile Selection
  │
  ├─► Step 1: Preprocess reflectance to [0, 1] range
  │
  ├─► Step 2: Ensure dimensions are divisible by 8 (pad if necessary)
  │
  ├─► Step 3: Run forward pass through SwinIR network (~1,650 ms on CPU)
  │
  └─► Step 4: Export 4x super-resolved 2.5m imagery
```

---

## 4. Hyperparameters & Unka Effect

| Hyperparameter | Value | Description | Tune Karne Par Effect |
| :--- | :---: | :--- | :--- |
| **`embed_dim`** | `180` | Latent token channel dimension. | **180:** High representational capacity; kam karke 64 karne se parameters 11.9M se ghat kar ~1.5M ho jayenge (SwinIR-Light). |
| **`window_size`** | `8` | Attention patch dimension ($8 \times 8$). | **8:** Satellite boundaries ke liye balanced context. |
| **`num_heads`** | `6` | Self-attention heads ($180 / 6 = 30$ dim per head). | Multi-angle edge capture enable karta hai. |
| **`depths`** | `[6, 6, 6, 6, 6, 6]` | 6 RSTB blocks jisme har ek me 6 transformer layers hain (total 36 attention layers). | Massive depth provides strong global context. |
| **`mlp_ratio`** | `2.0` | Feed-forward network expansion factor ($180 \to 360 \to 180$). | Non-linear capacity determine karta hai. |
