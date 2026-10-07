# SRM-Net (Residual Attention + MC-Dropout) — Working & Architecture

## 1. Formulas & Mathematical Equations

SRM-Net do core mathematical modules par based hai: **Squeeze-and-Excitation Channel Attention** aur **Test-Time Monte-Carlo Dropout for Bayesian Epistemic Uncertainty**.

### 1.1 Squeeze-and-Excitation Channel Attention (SE-CA)
Global spatial information ko 1D channel vector me condense kiya jata hai (Squeeze step):

$$z_c = \frac{1}{H \times W} \sum_{i=1}^H \sum_{j=1}^W x_c(i, j), \quad \mathbf{z} \in \mathbb{R}^C$$

Iske baad channel inter-dependencies learn karne ke liye two-layer MLP gating apply hoti hai (Excitation step):

$$\mathbf{s} = \sigma\left(\mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \cdot \mathbf{z})\right)$$

$$\tilde{\mathbf{X}} = \mathbf{X} \odot \mathbf{s}$$

- **$\mathbf{W}_1 \in \mathbb{R}^{\frac{C}{r} \times C}$:** Dimensionality reduction matrix ($r=16$).
- **$\mathbf{W}_2 \in \mathbb{R}^{C \times \frac{C}{r}}$:** Dimensionality expansion matrix.
- **$\sigma$:** Sigmoid function jo har channel ko $0$ se $1$ ke beech importance weight deti hai.

---

### 1.2 Bayesian Epistemic Uncertainty via Test-Time MC-Dropout
Standard inference me `model.eval()` karne par dropout layers off ho jati hain. Lekin SRM-Net me `force_dropout=True` rakha jata hai, jisse yeh ek **Deep Gaussian Process** ki tarah act karta hai.

Given input tile $\mathbf{x}$, model $T$ independent stochastic forward passes execute karta hai:
$$\{\hat{\mathbf{y}}_1, \hat{\mathbf{y}}_2, \dots, \hat{\mathbf{y}}_T\} = f_{\boldsymbol{\theta}, \mathbf{z}_t}(\mathbf{x}), \quad \mathbf{z}_t \sim \text{Bernoulli}(1 - p)$$

**1. Mean Super-Resolved Prediction:**
$$\boldsymbol{\mu}_{\text{SR}}(x, y) = \frac{1}{T} \sum_{t=1}^T \hat{\mathbf{y}}_t(x, y)$$

**2. Per-Pixel Epistemic Uncertainty (Variance):**
$$\boldsymbol{\sigma}^2(x, y) = \frac{1}{T} \sum_{t=1}^T \left(\hat{\mathbf{y}}_t(x, y) - \boldsymbol{\mu}_{\text{SR}}(x, y)\right)^2$$

- **Intuition:** Agar kisi pixel par sabhi $T$ passes ek jaisa output dete hain ($\sigma^2 \to 0$), iska matlab model ko 100% confidence hai. Agar stochastic dropout se pixel bar-bar badal raha hai ($\sigma^2 \text{ high}$), to wahan model guess kar raha hai (Hallucination Alert).

---

### 1.3 2D Spatial Bernoulli Dropout
Standard 1D dropout activations ko randomly drop karta hai. Remote sensing me spatial correlation maintain karne ke liye `Dropout2d` use hota hai jo pure feature channel mask ko drop karta hai:

$$\mathbf{M} \in \{0, 1\}^C, \quad M_c \sim \text{Bernoulli}(1 - p), \quad p = 0.20$$

$$\tilde{\mathbf{X}}_c = \frac{1}{1 - p} \cdot \mathbf{X}_c \cdot M_c$$

---

### 1.4 Fused Scientific Trust Confidence Score
Variance, cycle error aur spectral angle ko fuse karke ek 0% se 100% ka metric banta hai:

$$C(x, y) = \left[ 1.0 - \left( 0.45 \cdot \frac{\boldsymbol{\sigma}^2(x,y)}{\max \boldsymbol{\sigma}^2} + 0.35 \cdot \tilde{\mathcal{L}}_{\text{cycle}}(x,y) + 0.20 \cdot \tilde{\text{SAM}}(x,y) \right) \right] \times 100\%$$

---

## 2. Structure & Model Architecture

Text-based architectural pipeline:

```
[Input Tensor: 3 x 64 x 64] (Sentinel-2 10m GSD)
       │
       ├─────────────────────────────────────────┐ (Skip Shortcut)
       ▼                                         ▼
[Shallow Conv Head]                      [Bicubic Upsample 4x]
  (3 -> 64, 3x3)                                 │
       │                                         │
       ▼                                         │
[Trunk: 8x Residual Blocks with MC-Dropout]      │
  ┌────────────────────────────────────────┐     │
  │ • Conv2d(64, 64, 3x3) + ReLU           │     │
  │ • 2D Spatial Dropout (p=0.20, ACTIVE)  │     │
  │ • Conv2d(64, 64, 3x3)                  │     │
  │ • Squeeze-and-Excitation CA (r=16)     │     │
  │ • Local Residual Addition (+)          │     │
  └────────────────────────────────────────┘     │
       │                                         │
       ▼                                         │
[Trunk Fusion Conv] (64 -> 64)                   │
       │                                         │
       ▼ (+) Global Residual Addition            │
       │                                         │
[Upsampler Stage 1]                              │
  Conv(64 -> 256) + ReLU + PixelShuffle(2x)      │
       │                                         │
[Upsampler Stage 2]                              │
  Conv(64 -> 256) + ReLU + PixelShuffle(2x)      │
       │                                         │
       ▼                                         │
[Reconstruction Tail Conv] (64 -> 3, 3x3)        │
       │                                         │
       ▼ High-Frequency Residual Detail          │
       │                                         │
       └─────────────────► (+) ◄─────────────────┘
                            │
                            ▼
     [Radiometric Spectral Balance Constraint]
                            │
                            ▼
         [Output Tensor: 3 x 256 x 256] (2.5m GSD)
```

### Module Specifications Table

| Component | Layer Type | Channels | Kernel / Operation | Purpose |
| :--- | :--- | :---: | :---: | :--- |
| **Head** | `nn.Conv2d` | $3 \to 64$ | $3 \times 3$, pad 1 | Spatial feature projection. |
| **Trunk (8 Blocks)** | `ResidualBlockWithDropout` | $64 \to 64$ | $3 \times 3$ + SE-CA | Deep representation with active MC uncertainty. |
| **Spatial Dropout** | `nn.Dropout2d` | 64 | $p = 0.20$ | Bayesian Monte-Carlo sampling. |
| **Channel Attention**| `ChannelAttention` | $64 \to 4 \to 64$ | Adaptive Avg Pool + MLP | Radiometric channel recalibration. |
| **Upsampling (4x)** | `PixelShuffle(2)` $\times 2$ | $64 \to 256 \to 64$ | Dual Sub-pixel Conv | Spatial magnification from 10m to 2.5m. |
| **Tail** | `nn.Conv2d` | $64 \to 3$ | $3 \times 3$, pad 1 | Synthesizes high-frequency details. |

---

## 3. Workflow: Training Se Prediction Tak

```
[Training Phase]
Paired High-Res Dataset (Sentinel-2 10m + SPOT 1.5m Reference)
  │
  ├─► Random Patch Extraction (64x64) with Data Augmentations
  │
  ├─► Forward Pass: SRM-Net predicts 4x SR patch (256x256)
  │
  ├─► Loss Optimization: L1 Loss + Spectral Cosine Distance
  │
  ├─► Optimizer: Adam (lr=1e-4, weight_decay=1e-5)
  │
  └─► Checkpoint Export: weights/srmnet_x4_sentinel2.pt (0.93M params, 3.75 MB)

[Prediction & Real-Time Uncertainty Pipeline]
User selects AOI bounding box on Web Map
  │
  ├─► Step 1: Preprocess raw Sentinel-2 reflectance (normalize to [0, 1])
  │
  ├─► Step 2: Monte-Carlo Loop (T = 8 passes with force_dropout=True)
  │     For pass t = 1 to 8:
  │       y_t = model(x, force_dropout=True)
  │
  ├─► Step 3: Compute Mean Prediction: mean_sr = (1/8) * sum(y_t)
  │
  ├─► Step 4: Compute Variance Heatmap: variance = (1/8) * sum((y_t - mean_sr)^2)
  │
  ├─► Step 5: Compute Cycle-Consistency Loss: downsample(mean_sr) vs input_lr
  │
  ├─► Step 6: Fuse into Trust Confidence Map: Trust = 1.0 - (Variance + Cycle_Error)
  │
  └─► Step 7: Send High-Res Tile + Color-Coded Uncertainty Overlay to Web UI (3.7s total)
```

---

## 4. Hyperparameters & Unka Effect

| Hyperparameter | Value | Description | Tune Karne Par Effect |
| :--- | :---: | :--- | :--- |
| **`num_blocks`** | `8` | Residual Blocks ki ginti. | **Badhane par (e.g. 16):** Finer edge details aayenge par CPU latency 470ms se badh kar 900ms ho jayegi. **Kam karne par (e.g. 4):** Latency 250ms ho jayegi par blurry results aayenge. |
| **`dropout_rate` ($p$)** | `0.20` | Spatial dropout rate. | **0.20 optimal hai:** Agar $p > 0.35$ karein to output noisy ho jayega; agar $p < 0.05$ karein to uncertainty spread itna kam hoga ki hallucination pakad nahi aayegi. |
| **`num_mc_passes` ($T$)** | `8` | Monte-Carlo stochastic passes. | **8 passes:** Perfect balance between 3.7s UI speed aur statistical variance stability. $T=16$ karne par latency 7.4s ho jayegi. |
| **`reduction_ratio` ($r$)** | `16` | Channel Attention reduction. | Squeeze bottleneck; $r=16$ model size ko 0.93M params par compact rakhta hai. |
| **`learning_rate`** | `1e-4` | Adam optimizer initial step size. | Cosine learning rate decay ke sath smoothly converge hota hai. |
