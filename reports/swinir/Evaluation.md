# SwinIR (Shifted Window Transformer) — Evaluation & Comparison

## 1. Evaluation Metrics (Model Ko Judge Karne Ke Paimane)

SwinIR ko high-capacity Vision Transformer baseline ke roop me SPOT-6/7 ground truth ke against benchmark kiya gaya:

### 1. Structural Similarity Index (SSIM) — *Higher is Better ($\uparrow$)*
- **Formula:**
  $$\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}$$
- **SwinIR Score:** **0.1582** (CNN baselines se behtar, par HAT se thoda peeche).

### 2. Peak Signal-to-Noise Ratio (PSNR) — *Higher is Better ($\uparrow$)*
- **SwinIR Score:** **23.45 dB** (High pixel-level accuracy).

### 3. Spectral Angle Mapper (SAM) — *Lower is Better ($\downarrow$)*
- **Formula:**
  $$\text{SAM} = \arccos\left(\frac{\mathbf{x} \cdot \mathbf{y}}{\|\mathbf{x}\|_2 \|\mathbf{y}\|_2}\right)$$
- **SwinIR Score:** **6.45°** (Good spectral angle, although HAT achieves 5.67° due to explicit Channel Attention Blocks).

### 4. Cycle Consistency MAE ($\mathcal{L}_{\text{cycle}}$) — *Lower is Better ($\downarrow$)*
- **SwinIR Score:** **0.0168** (Low error, consistent radiance conservation).

### 5. CPU Latency & Resource Footprint — *Lower is Better ($\downarrow$)*
- **Parameters:** **11.90 Million**
- **Inference Time (CPU):** **~1,650 ms**
- **RAM Footprint:** **~620 MB**

---

## 2. Comparison: SwinIR vs. HAT vs. SRM-Net vs. CARN

| Metric / Dimension | SwinIR (Transformer Baseline) | HAT (Production Engine) | SRM-Net (Uncertainty Engine) | CARN (ESA Baseline) |
| :--- | :---: | :---: | :---: | :---: |
| **Model Category** | Shifted Window Transformer | Hybrid Attention Transformer | Residual CNN + MC-Dropout | Cascading Residual CNN |
| **Parameters** | **11.90 M** | **1.38 M (-88% size)** | **0.93 M** | **0.98 M** |
| **Structural SSIM** | 0.1582 | **0.1667 (Highest)** | 0.1556 | 0.1420 |
| **Spectral SAM (deg)** | 6.45° | **5.67° (Lowest Error)** | 7.13° | 8.12° |
| **Cycle MAE Error** | 0.0168 | **0.0142** | 0.0198 | 0.0235 |
| **Active Uncertainty** | No (Deterministic) | Yes (Active Dropout) | **Yes (3.7s Fast MC-8)** | No |
| **CPU Latency** | ~1,650 ms | 807 ms | **472 ms (Fastest)** | 420 ms |
| **RAM Footprint** | ~620 MB | ~480 MB | ~210 MB | **~195 MB** |

---

## 3. Decision Guide: Kab SwinIR Use Karein aur Kab Doosra?

```
                        [Transformer Super-Resolution Architecture]
                                             │
               ┌─────────────────────────────┴─────────────────────────────┐
               ▼                                                           ▼
      [Architectural Research]                                    [Operational GIS System]
               │                                                           │
        USE SWINIR                                                       USE HAT
  (Comparative Transformer                                    (88% Smaller Model: 1.38M)
   Baseline in Publications)                                  (Higher SSIM: 0.1667 + CAB)
```

### 1. SwinIR Kab Use Karein?
- **Academic Research & Papers:** Jab literature review me standard Vision Transformer benchmark prove karna ho.
- **Deep Token Representation Experiments:** Jab 180-dim heavy token embeddings analyze karni hon.

### 2. HAT Kyu Behtar Hai Production Ke Liye?
- HAT SwinIR se **8.6x chhota** (1.38M vs 11.9M) hai, **2x tez** hai, aur **Channel Attention** hone ki wajah se satellite imagery ka natural spectral color balance behtar preserve karta hai.
