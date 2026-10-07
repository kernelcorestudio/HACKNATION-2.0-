# Real-ESRGAN (RRDBNet GAN Generator) — Working & Architecture

## 1. Formulas & Mathematical Equations

Real-ESRGAN ka backbone **Residual-in-Residual Dense Block (RRDB)** hai, aur iska training objective composite **Adversarial + Perceptual + Pixel L1 Loss** par based hai.

### 1.1 Residual Dense Block (RDB) Formulation
Ek single Residual Dense Block (RDB) ke andar 5 convolutional layers hoti hain jisme har layer apne se pehle ki sabhi layers ke outputs ko concatenate karti hai (Dense Connections):

$$\mathbf{x}_1 = \sigma(\mathbf{W}_1 * \mathbf{x}_0)$$
$$\mathbf{x}_2 = \sigma(\mathbf{W}_2 * [\mathbf{x}_0, \mathbf{x}_1])$$
$$\mathbf{x}_3 = \sigma(\mathbf{W}_3 * [\mathbf{x}_0, \mathbf{x}_1, \mathbf{x}_2])$$
$$\mathbf{x}_4 = \sigma(\mathbf{W}_4 * [\mathbf{x}_0, \mathbf{x}_1, \mathbf{x}_2, \mathbf{x}_3])$$
$$\mathbf{x}_{\text{RDB}} = \mathbf{W}_5 * [\mathbf{x}_0, \mathbf{x}_1, \mathbf{x}_2, \mathbf{x}_3, \mathbf{x}_4] \cdot \beta + \mathbf{x}_0, \quad \beta = 0.2$$

- **$[\cdot]$:** Channel concatenation. Har layer ke baad channels grow hote hain ($g = 32$ growth channels).
- **$\beta = 0.2$:** Residual scaling factor jo deep layers me numerical instability aur gradient exploding ko rokta hai.

---

### 1.2 Residual in Residual Dense Block (RRDB)
3 nested RDBs ko cascade karke unhe fir se ek global residual shortcut ke sath wrap kiya jata hai:

$$\mathbf{x}_{\text{RRDB}} = \text{RDB}_3(\text{RDB}_2(\text{RDB}_1(\mathbf{x}_0))) \cdot \beta + \mathbf{x}_0$$

- **Depth:** Pure generator me aise 23 RRDB blocks hote hain, jo 100 se zyada convolutional layers ki representational capacity banate hain.

---

### 1.3 Relativistic Average GAN (RaGAN) Loss
Standard GANs sirf yeh check karte hain ki image real hai ya fake. RaGAN Discriminator yeh evaluate karta hai ki *"kya real image fake se zyada realistic hai, aur fake image real se kam realistic hai?"*

Discriminator probability:
$$D_{\text{Ra}}(\mathbf{x}_r, \mathbf{x}_f) = \sigma\left( C(\mathbf{x}_r) - \mathbb{E}_{\mathbf{x}_f}[C(\mathbf{x}_f)] \right)$$

Generator Adversarial Loss:
$$\mathcal{L}_{\text{adv}}^G = - \mathbb{E}_{\mathbf{x}_r} \left[ \log \left( 1 - D_{\text{Ra}}(\mathbf{x}_r, \mathbf{x}_f) \right) \right] - \mathbb{E}_{\mathbf{x}_f} \left[ \log \left( D_{\text{Ra}}(\mathbf{x}_f, \mathbf{x}_r) \right) \right]$$

---

### 1.4 Perceptual VGG Loss
Pixel-by-pixel match karne ke bajaye, deep pretrained VGG-19 network ke activation feature maps ko compare kiya jata hai:

$$\mathcal{L}_{\text{percep}} = \sum_{i} \frac{1}{N_i} \|\phi_i(\mathbf{I}_{\text{SR}}) - \phi_i(\mathbf{I}_{\text{HR}})\|_1$$

- **Why this causes Hallucination:** VGG features natural images (cars, dogs, trees) par trained hote hain. Yeh model ko satellite imagery par bhi natural photographic textures zabardasti generate karne par majboor karta hai.

---

### 1.5 High-Order Synthetic Degradation Modeling
Real-world blur aur noise ko simulate karne ke liye cascading degradation pipeline:

$$\mathbf{I}_{\text{LR}} = \left[ \left( (\mathbf{I}_{\text{HR}} \otimes \mathbf{k}_1) \downarrow_s + \mathbf{n}_1 \right)_{\text{JPEG}_1} \otimes \mathbf{k}_2 \right] \downarrow_s + \mathbf{n}_2 + \text{sinc}$$

---

## 2. Structure & Model Architecture

Text-based architectural pipeline:

```
[Input Tensor: 3 x 64 x 64] (Sentinel-2 LR)
       │
       ▼
[First Feature Extraction Conv2d(3, 64, 3x3)] ──────────────┐
       │                                                    │
       ▼                                                    │
[Deep Trunk: 23x RRDB Blocks]                               │
  ┌───────────────────────────────────────────────┐         │
  │ • RRDB 1 (3x RDBs with Dense Skip Connections)│         │
  │ • RRDB 2                                      │         │
  │ • ...                                         │         │
  │ • RRDB 23                                     │         │
  └───────────────────────────────────────────────┘         │
       │                                                    │
       ▼                                                    │
[Trunk Conv2d(64, 64, 3x3)]                                 │
       │                                                    │
       ▼ (+) Global Long Residual Connection ◄──────────────┘
       │
[Upsampler Stage 1]
  Nearest Interpolate(2x) + Conv2d(64, 64) + LeakyReLU
       │
[Upsampler Stage 2]
  Nearest Interpolate(2x) + Conv2d(64, 64) + LeakyReLU
       │
       ▼
[High-Res Refinement Conv2d(64, 64, 3x3) + LeakyReLU]
       │
       ▼
[Final Radiance Exit Conv2d(64, 3, 3x3)]
       │
       ▼
[Perceptually Sharp 2.5m Image: 3 x 256 x 256]
```

### Module Specifications Table

| Component | Layer Type | Channels | Multipliers | Purpose |
| :--- | :--- | :---: | :---: | :--- |
| **First Conv** | `nn.Conv2d` | $3 \to 64$ | $3 \times 3$, pad 1 | Feature tokenization. |
| **RRDB Trunk** | 23 RRDB Blocks | $64 \to 64$ | $3 \times \text{RDB per RRDB}$ | Massive dense capacity (16.7M params). |
| ↳ *Dense Convs* | 5 Convs per RDB | $64 \to 32 \to 64$| Growth $g=32$ | Continuous feature reuse across layers. |
| **Trunk Conv** | `nn.Conv2d` | $64 \to 64$ | $3 \times 3$, pad 1 | Aggregates dense trunk features. |
| **Upsamplers** | `Nearest + Conv` $\times 2$ | $64 \to 64$ | $2\times$ factor each | Non-parametric nearest neighbor interpolation. |
| **Final Exit** | `nn.Conv2d` | $64 \to 3$ | $3 \times 3$, pad 1 | Outputs 3-channel RGB image. |

---

## 3. Workflow: Training Se Prediction Tak

```
[Training Workflow (Adversarial Setup)]
High-Resolution Dataset (DIV2K, Flickr2K, Outdoor Scenes)
  │
  ├─► High-Order Degradation Process: Apply Blur -> Downsample -> Noise -> JPEG -> Sinc Filter
  │
  ├─► Generator Forward Pass: RRDBNet generates synthetic SR image
  │
  ├─► Discriminator Forward Pass: U-Net Discriminator judges real vs synthetic
  │
  ├─► Composite Loss Backpropagation: L_total = L1 + 1.0 * L_percep + 0.1 * L_adv
  │
  ├─► Alternate Updates: Update Generator G, then update Discriminator D
  │
  └─► Export Generator Weights: weights/RealESRGAN_x4plus.pth (16.7M params, 67 MB)

[Prediction & Scientific Audit Pipeline in HACKNATION]
User selects AOI tile
  │
  ├─► Step 1: Preprocess raw Sentinel-2 radiance
  │
  ├─► Step 2: Forward pass through RRDBNet generator (~5,850 ms on CPU)
  │
  ├─► Step 3: Compute Cycle MAE Error: downsample(SR) vs input Sentinel-2
  │     => Real-ESRGAN shows HIGH cycle error (0.0382)
  │
  ├─► Step 4: Trust Layer flags phantom micro-structures and drops confidence to ~64%
  │
  └─► Step 5: Web UI displays the sharp image with an explicit "Hallucination Warning"
```

---

## 4. Hyperparameters & Unka Effect

| Hyperparameter | Value | Description | Tune Karne Par Effect |
| :--- | :---: | :--- | :--- |
| **`num_blocks`** | `23` | RRDB blocks ki ginti. | Deep generator capacity; kam karne se parameters ghatenge par textural sharpness degrade hogi. |
| **`growth_channels` ($g$)** | `32` | Har RDB convolutional layer ka feature growth. | Dense feature mixing determine karta hai. |
| **`residual_scale` ($\beta$)** | `0.2` | Residual branch scale factor. | Deep network ko explode hone se bachata hai. |
| **$\lambda_{\text{percep}}$** | `1.0` | VGG Perceptual Loss weight. | Isko badhane se visual edges sharp hoti hain par hallucination risk surge ho jata hai. |
| **$\lambda_{\text{adv}}$** | `0.1` | Adversarial GAN loss weight. | Discriminator ke feedback ka weight; high value artificial textures inject karti hai. |
