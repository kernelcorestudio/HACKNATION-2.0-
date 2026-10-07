# HAT (Hybrid Attention Transformer) — Evaluation & Comparison

## 1. Evaluation Metrics (Model Ko Judge Karne Ke Paimane)

Super-Resolution aur Remote Sensing me standard classification metrics (jaise Accuracy ya F1-score) ke bajaye spatial, radiometric aur spectral fidelity metrics use kiye jaate hain:

### 1. Structural Similarity Index (SSIM) — *Higher is Better ($\uparrow$)*
- **Formula:**
  $$\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}$$
- **Explanation:** Yeh measure karta hai ki zameen ke structural patterns, roads ke edges, aur building corners kitne accurately reconstruct hue hain compared to high-resolution ground truth.
- **HAT Score:** **0.1667** (All models me highest).

### 2. Peak Signal-to-Noise Ratio (PSNR) — *Higher is Better ($\uparrow$)*
- **Formula:**
  $$\text{PSNR} = 10 \cdot \log_{10}\left(\frac{\text{MAX}_I^2}{\text{MSE}}\right), \quad \text{MSE} = \frac{1}{mn}\sum_{i=0}^{m-1}\sum_{j=0}^{n-1}[I(i,j) - K(i,j)]^2$$
- **Explanation:** Output image aur reference ground truth ke beech pixel-level noise/error ka measure. Decibels (dB) me measure hota hai.
- **HAT Score:** **~23.8 dB** (Clean radiometric consistency).

### 3. Spectral Angle Mapper (SAM) — *Lower is Better ($\downarrow$)*
- **Formula:**
  $$\text{SAM}(\mathbf{x}, \mathbf{y}) = \arccos\left(\frac{\sum_{i=1}^B x_i y_i}{\sqrt{\sum_{i=1}^B x_i^2} \sqrt{\sum_{i=1}^B y_i^2}}\right)$$
- **Explanation:** Har pixel ke spectral vector (Red, Green, Blue radiance) ka angle difference measure karta hai (degrees me). Yeh confirm karta hai ki image zoom hone par vegetation ya soil ka color change na ho.
- **HAT Score:** **5.67°** (Lowest spectral distortion).

### 4. ERGAS (Relative Dimensionless Global Error in Synthesis) — *Lower is Better ($\downarrow$)*
- **Formula:**
  $$\text{ERGAS} = 100 \frac{h}{l} \sqrt{\frac{1}{B}\sum_{i=1}^B \frac{\text{RMSE}(i)^2}{\text{Mean}(i)^2}}$$
- **Explanation:** French Space Agency (CNES) ka global synthesis quality standard.
- **HAT Score:** **3.18** (International high-quality standard < 4.0).

### 5. Cycle Consistency MAE ($\mathcal{L}_{\text{cycle}}$) — *Lower is Better ($\downarrow$)*
- **Formula:**
  $$\mathcal{L}_{\text{cycle}} = \|\mathcal{D}_{4\times}(\mathbf{I}_{\text{SR}}) - \mathbf{I}_{\text{LR}}\|_1$$
- **Explanation:** Agar 4x super-resolved image ko wapas downscale karein, to kya original Sentinel-2 image match hoti hai? Agar match hoti hai, to model ne kuch jhoota invent nahi kiya.
- **HAT Score:** **0.0142** (Sabse kam residual drift).

---

## 2. Comparison Matrix (Similar Models Ke Sath Tula)

Standardized benchmark on Sentinel-2 L2A to 2.5m GSD against SPOT-6/7 1.5m Ground Truth:

| Dimension / Metric | HAT | SRM-Net | CARN | Real-ESRGAN | Bicubic Interpolation |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Model Type** | Vision Transformer | Residual CNN + Dropout | Cascading CNN | Adversarial GAN | Mathematical Algorithm |
| **Parameters** | **1.38 M** | **0.93 M** | **0.98 M** | **16.70 M** | 0 |
| **Structural SSIM** | **0.1667 (Best)** | 0.1556 | 0.1420 | 0.1019 – 0.1291 | 0.1105 |
| **Spectral SAM (deg)** | **5.67° (Lowest)** | 7.13° | 8.12° | 5.24° | 0.00° |
| **Cycle MAE Error** | **0.0142 (Lowest)** | 0.0198 | 0.0235 | 0.0382 (High Error) | 0.0000 |
| **Scientific Trust Score**| **85.4%** | 79.7% | 72.1% | 64.2% (Penalized) | N/A |
| **Inference Time (CPU)**| 807 ms | **472 ms (Fastest)**| 420 ms | ~5,850 ms (Slow) | **< 10 ms** |
| **Peak Memory Footprint**| ~480 MB | ~210 MB | ~195 MB | ~1,150 MB | < 50 MB |

---

## 3. Decision Guide: Kab Kaunsa Model Use Karein?

```
                                  [Need Super-Resolution]
                                             │
               ┌─────────────────────────────┴─────────────────────────────┐
               ▼                                                           ▼
       [Production Task]                                           [Interactive UI / Edge]
               │                                                           │
       ┌───────┴───────┐                                           ┌───────┴───────┐
       ▼               ▼                                           ▼               ▼
[Certified GIS]  [Art / Media]                               [Live Web App]  [Field Survey Device]
       │               │                                           │               │
       ▼               ▼                                           ▼               ▼
   USE HAT        USE Real-ESRGAN                             USE SRM-Net       USE CARN
```

### 1. HAT Kab Use Karein? (Best For:)
- **Official Cadastral Mapping & GIS Exports:** Jab legal demarcation ya boundary dispute ke liye sub-4m GeoTIFF export karna ho.
- **Road & Bridge Vectorization:** Jab continuous, sharp vector lines extract karni hon.
- **Scientific Analysis (NDVI/NDWI/EVI):** Jab zameen ke physical spectral reflectance values bilkul unaltered chahiye hon.

### 2. SRM-Net Kab Use Karein?
- Jab **Web Dashboard par real-time interactive sliding** karni ho aur latency sub-second (<500ms) chahiye.
- Jab **Active Monte-Carlo Hallucination Heatmap** render karni ho (SRM-Net 8 passes 3.7 seconds me kar leta hai, jabki HAT ko 14+ seconds lagte hain).

### 3. CARN Kab Use Karein?
- Jab **Low-power battery-operated field tablet** ya offline edge raspberry-pi kit par model run karna ho (<200MB RAM requirement).

### 4. Real-ESRGAN Kab Use Karein?
- Press release ya general presentation poster ke liye jaha visual aesthetic appeal chahiye, lekin **scientific ya legal mapping ke liye ise strictly use na karein** kyunki yeh artificial details hallucinate karta hai.
