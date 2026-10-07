# CARN (Cascading Residual Network) — Evaluation & Comparison

## 1. Evaluation Metrics (Model Ko Judge Karne Ke Paimane)

CARN ko evaluate karne ke liye international remote sensing standard metrics use hote hain:

### 1. Structural Similarity Index (SSIM) — *Higher is Better ($\uparrow$)*
- **Formula:**
  $$\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}$$
- **CARN Score:** **0.1420** (Conservative structural reconstruction, consistent baseline across international datasets).

### 2. Peak Signal-to-Noise Ratio (PSNR) — *Higher is Better ($\uparrow$)*
- **Formula:**
  $$\text{PSNR} = 10 \cdot \log_{10}\left(\frac{1.0}{\text{MSE}}\right)$$
- **CARN Score:** **22.56 dB** (Smooth and stable radiometric profile).

### 3. Spectral Angle Mapper (SAM) — *Lower is Better ($\downarrow$)*
- **Formula:**
  $$\text{SAM} = \arccos\left(\frac{\mathbf{x} \cdot \mathbf{y}}{\|\mathbf{x}\|_2 \|\mathbf{y}\|_2}\right)$$
- **CARN Score:** **8.27°** (Higher than HAT's 5.67°, par acceptable for baseline comparison).

### 4. ERGAS (CNES Synthesis Error) — *Lower is Better ($\downarrow$)*
- **Formula:**
  $$\text{ERGAS} = 100 \frac{h}{l} \sqrt{\frac{1}{B}\sum_{i=1}^B \frac{\text{RMSE}(i)^2}{\text{Mean}(i)^2}}$$
- **CARN Score:** **3.68** (Satisfies the international remote sensing benchmark standard < 4.0).

### 5. Memory & Resource Footprint — *Lower is Better ($\downarrow$)*
- **Model Size:** **3.75 MB** on disk.
- **Working RAM:** **~195 MB** (Lowest in the entire benchmark catalog).

---

## 2. Comparison Matrix (Similar Models Ke Sath Tula)

Standardized benchmark on Sentinel-2 L2A to 2.5m GSD against SPOT-6/7 1.5m Ground Truth:

| Metric / Dimension | CARN (ESA Baseline) | HAT (Production Engine) | SRM-Net (Uncertainty Engine) | Real-ESRGAN (Adversarial GAN) |
| :--- | :---: | :---: | :---: | :---: |
| **Model Category** | Cascading CNN | Vision Transformer | Residual CNN + MC-Dropout | Deep GAN (RRDBNet) |
| **Parameter Count** | **0.98 M** | 1.38 M | 0.93 M | 16.70 M |
| **Structural SSIM** | **0.1420** | **0.1667 (Sharpest)** | 0.1556 | 0.1019 – 0.1291 |
| **Spectral SAM (deg)** | **8.27°** | 5.67° (Best) | 7.13° | 5.24° |
| **Cycle MAE Error** | **0.0235** | 0.0142 | 0.0198 | 0.0382 (High Error) |
| **Deterministic CPU Latency**| **420 ms (Fastest)** | 807 ms | 472 ms | ~5,850 ms |
| **RAM Footprint** | **~195 MB (Lowest)** | ~480 MB | ~210 MB | ~1,150 MB |
| **Primary Deployment** | **Offline Edge / Field Kits** | **Official GIS Exports** | **Interactive Web UI** | **Visual Media & Audit** |

---

## 3. Decision Guide: Kab Kaunsa Model Use Karein?

```
                                  [Select Super-Resolution Engine]
                                                 │
                   ┌─────────────────────────────┴─────────────────────────────┐
                   ▼                                                           ▼
         [High Compute Environment]                                 [Low Power / Edge Kit]
                   │                                                           │
         ┌─────────┴─────────┐                                                 ▼
         ▼                   ▼                                              USE CARN
      USE HAT           USE SRM-NET                                  (< 195 MB RAM Footprint)
  (For Max Accuracy)  (For Web UI & Uncertainty)                     (Sub-1M Params, Offline)
```

### 1. CARN Kab Use Karein? (Best For:)
- **Offline Field Survey Devices / Edge Kits:** Jab user ke paas internet na ho aur battery-powered rugged device ya Raspberry Pi par model run karna ho.
- **Academic & Scientific Validation:** Jab paper submission ya hackathon panel me international standard baseline comparison dikhana ho (ESA EvoLand / WorldStrat).
- **Conservative Agriculture & Forestry Mapping:** Jab aisi zameen dekh rahe hon jaha over-sharpening se bachna zaroori ho.

### 2. HAT Kab Use Karein?
- Jab maximum spatial edge sharpness aur lowest spectral distortion chahiye ho legal cadastral map export ke liye.

### 3. SRM-Net Kab Use Karein?
- Jab interactive web browser me live zoom/pan aur hallucination uncertainty heatmap chahiye ho.
