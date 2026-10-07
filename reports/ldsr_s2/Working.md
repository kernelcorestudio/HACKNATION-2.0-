# LDSR-S2 (Latent Diffusion Super-Resolution) — Working & Architecture

## 1. Formulas & Mathematical Equations

LDSR-S2 do stage deep generative framework par kaam karta hai: **Spatial Autoencoder Latent Compression** aur **Conditional Denoising Diffusion Implicit Models (DDIM)**.

### 1.1 Spatial Autoencoder Encoding & Decoding
Input 4-band satellite image $\mathbf{x} \in \mathbb{R}^{H \times W \times 4}$ ko compressed latent manifold $\mathbf{z} \in \mathbb{R}^{h \times w \times c}$ me map kiya jata hai:

$$\mathbf{z}_0 = \mathcal{E}(\mathbf{x}), \quad \hat{\mathbf{x}} = \mathcal{D}(\hat{\mathbf{z}}_0)$$

- **$\mathcal{E}$:** Encoder jo spatial dimensionality ko reduce karta hai taaki diffusion process fast aur memory-efficient ho sake.
- **$\mathcal{D}$:** Decoder jo final denoised latent ko wapas high-resolution 2.5m satellite radiance me project karta hai.

---

### 1.2 Forward Diffusion Process (Closed-Form Noise Addition)
Training ke waqt latent $\mathbf{z}_0$ me discrete timesteps $t$ par calibrated Gaussian noise add ki jati hai:

$$q(\mathbf{z}_t | \mathbf{z}_0) = \mathcal{N}\left(\mathbf{z}_t; \, \sqrt{\bar{\alpha}_t}\mathbf{z}_0, \, (1 - \bar{\alpha}_t)\mathbf{I}\right)$$

$$\mathbf{z}_t = \sqrt{\bar{\alpha}_t}\mathbf{z}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$$

- **$\beta_t \in (0, 1)$:** Variance schedule.
- **$\alpha_t = 1 - \beta_t, \quad \bar{\alpha}_t = \prod_{s=1}^t \alpha_s$:** Cumulative noise scaling factors.

---

### 1.3 Denoising Score Matching Loss Function
UNet neural network $\boldsymbol{\epsilon}_\theta$ ko train kiya jata hai noisy latent $\mathbf{z}_t$ se exact added noise $\boldsymbol{\epsilon}$ predict karne ke liye, conditioning vector $\mathbf{y}$ (10m Sentinel-2 low-res input) ke sath:

$$\mathcal{L}_{\text{LDM}} = \mathbb{E}_{\mathbf{z}_0, \, \mathbf{y}, \, \boldsymbol{\epsilon}, \, t} \left[ \left\| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{z}_t, t, \tau_\theta(\mathbf{y})) \right\|_2^2 \right]$$

---

### 1.4 Reverse DDIM Sampling Formulation (15 Steps)
Inference ke waqt, pure Gaussian noise $\mathbf{z}_T \sim \mathcal{N}(0, \mathbf{I})$ se shuru karke 15 deterministic steps me noise subtract ki jati hai:

$$\mathbf{z}_{t-1} = \sqrt{\alpha_{t-1}} \left( \frac{\mathbf{z}_t - \sqrt{1 - \alpha_t} \boldsymbol{\epsilon}_\theta(\mathbf{z}_t, t)}{\sqrt{\alpha_t}} \right) + \sqrt{1 - \alpha_{t-1} - \sigma_t^2} \boldsymbol{\epsilon}_\theta(\mathbf{z}_t, t) + \sigma_t \boldsymbol{\eta}_t$$

- Deterministic sampling ke liye $\sigma_t = 0$ set kiya jata hai, jisse same seed par identical output milta hai.

---

## 2. Structure & Model Architecture

Text-based architectural pipeline:

```
[Sentinel-2 10m 4-Band Input: 4 x H x W]
       │
       ├─────────────────────────────────────────┐
       ▼                                         ▼
[First-Stage Autoencoder Encoder E]    [Conditioning Preprocessor tau_theta]
       │                                         │
       ▼ Latent z_0                              │
[Forward Noise Injection (t=15 steps)]           │
       │                                         │
       ▼ Noisy Latent z_t                        │
       │                                         │
       ├─────────────────┐                       │
       │                 ▼                       ▼
       │      [Cross-Attention Denoising UNet epsilon_theta]
       │                 │
       │                 ▼
       │      [Predicted Noise epsilon_hat]
       │                 │
       ▼                 ▼
       [DDIM Reverse Step Update: z_{t-1}] <── (Repeat 15 Steps)
       │
       ▼ Denoised Latent z_hat_0
[First-Stage Autoencoder Decoder D]
       │
       ▼
[Super-Resolved 2.5m 4-Band Output: 4 x 4H x 4W] (B02, B03, B04, B08)
```

### Module Specifications Table

| Component | Architecture Type | Input / Output | Operational Role |
| :--- | :--- | :---: | :--- |
| **Encoder $\mathcal{E}$** | Convolutional Downsampler | $4 \times H \times W \to 4 \times \frac{H}{4} \times \frac{W}{4}$ | High-dimensional satellite reflectance ko latent manifold me compress karta hai. |
| **Denoising UNet** | Multi-Res Cross-Attention UNet | Latent $\mathbf{z}_t$ + Time $t$ + LR $\mathbf{y}$ | 15 steps me Gaussian noise ko estimate karta hai. |
| **Cross-Attention**| Scaled Dot-Product Attention | Token dim = 512 | Low-res spatial guidance inject karta hai. |
| **Decoder $\mathcal{D}$** | Sub-Pixel Upsampler + ResBlocks | Latent $\to 4 \times 4H \times 4W$ | Latent codes se sharp 2.5m multi-spectral pixels synthesize karta hai. |

---

## 3. Workflow: Training Se Prediction Tak

```
[Training Workflow (ESA OpenSR Protocol)]
Multi-Spectral Paired Dataset (Sentinel-2 10m + Sub-2.5m Aerial Reference)
  │
  ├─► Step 1: Pre-train First Stage Autoencoder (L1 + Perceptual LPIPS Loss)
  │
  ├─► Step 2: Freeze Autoencoder, encode HR images into latent z_0
  │
  ├─► Step 3: Sample random timestep t in [1, 1000] and add Gaussian noise
  │
  ├─► Step 4: Train Denoising UNet to predict noise epsilon with AdamW (lr = 1e-4)
  │
  └─► Checkpoint Export: weights/opensr-ldsrs2_v1_0_0.ckpt (~1.13 GB)

[Prediction Workflow]
Target Sentinel-2 AOI
  │
  ├─► Step 1: Ingest 4-band reflectance stack (Blue, Green, Red, NIR)
  │
  ├─► Step 2: Extract conditioning features tau_theta(y_LR)
  │
  ├─► Step 3: Initialize pure Gaussian noise latent z_15 ~ N(0, I)
  │
  ├─► Step 4: Execute 15 DDIM reverse iterations:
  │     For step = 15 down to 1:
  │       epsilon = UNet(z_step, step, conditioning)
  │       z_{step-1} = DDIM_update(z_step, epsilon)
  │
  ├─► Step 5: Pass denoised latent z_0 through Decoder D
  │
  └─► Step 6: Export 4-band 2.5m GeoTIFF (Time elapsed: ~18.4 seconds)
```

---

## 4. Hyperparameters & Unka Effect

| Hyperparameter | Value | Description | Tune Karne Par Effect |
| :--- | :---: | :--- | :--- |
| **`ddim_steps`** | `15` | Reverse diffusion inference steps. | **15 steps:** Speed aur quality ka sweet spot (~18.4s). 50 steps karne se fine textures thodi behtar hongi par time 1 minute se upar chala jayega. 5 steps karne se blurry artifacts aayenge. |
| **`guidance_scale`** | `2.0` | Classifier-free conditioning strength. | Conditioning kitni strongly force hoti hai; > 4.0 karne par contrast oversaturate ho sakta hai. |
| **`latent_channels`** | `4` | Compressed latent representation channels. | Manifold capacity; 4 channels Sentinel-2 ke 4 bands ke sath aligned hai. |
| **`beta_schedule`** | `linear` | Noise variance schedule ($\beta_1 = 10^{-4}$ to $\beta_T = 0.02$). | Noise injection curvature control karta hai. |
