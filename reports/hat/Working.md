# HAT (Hybrid Attention Transformer) — Working & Architecture

## 1. Formulas & Mathematical Equations

HAT ka core principle do attention mechanisms ka hybrid fusion hai: **Window Multi-Head Self-Attention (Spatial)** aur **Channel Attention (Spectral)**.

### 1.1 Window-based Multi-Head Self-Attention (W-MSA)
Input feature map $\mathbf{X}$ ko chhote local windows $M \times M$ ($M=8$) me baanta jata hai. Har window ke liye Query ($\mathbf{Q}$), Key ($\mathbf{K}$), aur Value ($\mathbf{V}$) matrices calculate hoti hain:

$$\mathbf{Q} = \mathbf{X} \mathbf{W}_Q, \quad \mathbf{K} = \mathbf{X} \mathbf{W}_K, \quad \mathbf{V} = \mathbf{X} \mathbf{W}_V$$

Scaled Dot-Product Attention formula:
$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Softmax}\left(\frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}} + \mathbf{B}\right) \mathbf{V}$$

- **$d_k$:** Dimension per attention head ($64 / 4 = 16$).
- **$\mathbf{B}$:** Learned Relative Position Bias jo geometric alignment preserve karta hai.
- **Intuition:** Yeh formula window ke andar har pixel ko doosre pixel se compare karta hai. Agar ek pixel road ka hissa hai, to model aas-pass ke road pixels par zyada focus karta hai, jisse sharp boundary banti hai.

---

### 1.2 Channel Attention Block (CAB)
Inter-band spectral balance (Red, Green, Blue, NIR radiance ratios) maintain karne ke liye Squeeze-and-Excitation channel gating formula use hota hai:

$$\mathbf{z}_{\text{avg}} = \text{AdaptiveAvgPool2d}(\mathbf{X}), \quad \mathbf{z}_{\text{max}} = \text{AdaptiveMaxPool2d}(\mathbf{X})$$

$$\mathbf{s} = \sigma\left(\mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \cdot \mathbf{z}_{\text{avg}}) + \mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \cdot \mathbf{z}_{\text{max}})\right)$$

$$\mathbf{X}_{\text{CAB}} = \mathbf{X} \odot \mathbf{s}$$

- **$\sigma$:** Sigmoid function jo 0 se 1 ke beech scale factor deta hai.
- **$\mathbf{W}_1, \mathbf{W}_2$:** Channel reduction MLP weights ($r=16$).
- **$\odot$:** Element-wise multiplication.
- **Intuition:** Har spectral band ki importance dynamically adjust hoti hai, taaki jungle ka green aur mitti ka brown natural dikhe.

---

### 1.3 Sub-Pixel Convolution (PixelShuffle 4x)
Latent features ko $4\times$ expand karne ke liye do stages me sub-pixel convolution lagaya jata hai:

$$\mathbf{Y}_{c, \, 2y + j, \, 2x + i} = \mathbf{T}_{4c + 2j + i, \, y, \, x}, \quad i, j \in \{0, 1\}$$

- **Explanation:** $64$ channels ko pehle $256$ me expand kiya jata hai, fir PixelShuffle(2) se unhe $2\times$ spatial grid me rearrange kiya jata hai. Do bar repeat karne par $(64, H, W) \to (64, 4H, 4W)$ ban jata hai.

---

### 1.4 Radiometric Spectral Preservation Constraint
Color drift aur artificial saturation ko rokne ke liye output par physical radiance balance enforce hota hai:

$$\Delta_{\text{color}} = \frac{1}{HW} \sum_{x,y} \mathbf{I}_{\text{SR}}(x,y) - \frac{1}{HW} \sum_{x,y} \mathbf{I}_{\text{Bicubic}}(x,y)$$

$$\mathbf{I}_{\text{Final}} = \text{Clamp}\left(\mathbf{I}_{\text{SR}} - 0.75 \cdot \Delta_{\text{color}}, 0.0, 1.0\right)$$

---

### 1.5 Composite Training Loss Function
Model ko train karte waqt dono structural aur spectral accuracy ko optimize kiya jata hai:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{Charbonnier}}(\mathbf{I}_{\text{SR}}, \mathbf{I}_{\text{HR}}) + \lambda_{\text{SAM}} \cdot \mathcal{L}_{\text{SAM}}(\mathbf{I}_{\text{SR}}, \mathbf{I}_{\text{HR}})$$

$$\mathcal{L}_{\text{Charbonnier}} = \sqrt{\|\mathbf{I}_{\text{SR}} - \mathbf{I}_{\text{HR}}\|^2 + \epsilon^2}, \quad \epsilon = 10^{-3}$$

$$\mathcal{L}_{\text{SAM}} = \arccos\left( \frac{\mathbf{I}_{\text{SR}} \cdot \mathbf{I}_{\text{HR}}}{\|\mathbf{I}_{\text{SR}}\|_2 \|\mathbf{I}_{\text{HR}}\|_2} \right)$$

---

## 2. Structure & Model Architecture

```mermaid
graph TD
    A["Sentinel-2 Optical Input: (B, 3, 64, 64) [10m GSD]"] --> B["Shallow Feature Head: Conv2d(3, 64, 3x3)"]
    A --> C["Bicubic Interpolation Baseline Anchor (4x Upscale)"]
    
    B --> D["Trunk: 6x Hybrid Attention Blocks (HAB)"]
    
    subgraph HAB["Hybrid Attention Block (HAB)"]
        D1["BatchNorm2d"] --> D2["Window-based Multi-Head Self Attention (W-MSA, 8x8, 4 Heads)"]
        D2 --> D3["Spatial Dropout (p=0.15)"]
        D3 --> D4["Local Residual Addition (+)"]
        D4 --> D5["BatchNorm2d"]
        D5 --> D6["Channel Attention Block (CAB, Reduction r=16)"]
        D6 --> D7["Feed-Forward Conv2d Trunk + LeakyReLU"]
        D7 --> D8["Local Residual Addition (+)"]
    end
    
    D --> E["Trunk Fusion Layer: Conv2d(64, 64, 3x3)"]
    
    B --> F["Global Residual Addition (feat = shallow + trunk)"]
    E --> F
    
    F --> G["Upsampling Stage 1: Conv2d(64, 256) + PixelShuffle(2x) + LeakyReLU"]
    G --> H["Upsampling Stage 2: Conv2d(64, 256) + PixelShuffle(2x) + LeakyReLU"]
    
    H --> I["Synthesis Tail: Conv2d(64, 32) -> LeakyReLU -> Conv2d(32, 3)"]
    I --> J["High-Frequency Residual Detail (B, 3, 256, 256)"]
    
    C --> K["Additive Recombination: SR = Bicubic + Detail"]
    J --> K
    
    K --> L["Radiometric Spectral Preservation Constraint"]
    L --> M["Final Super-Resolved Raster: (B, 3, 256, 256) [2.5m GSD]"]

    classDef headStyle fill:#1e3a8a,stroke:#3b82f6,stroke-width:2px,color:#ffffff;
    classDef habStyle fill:#14532d,stroke:#22c55e,stroke-width:2px,color:#ffffff;
    classDef upStyle fill:#701a75,stroke:#d946ef,stroke-width:2px,color:#ffffff;
    classDef exitStyle fill:#7c2d12,stroke:#ea580c,stroke-width:2px,color:#ffffff;

    class B headStyle;
    class D1,D2,D3,D4,D5,D6,D7,D8 habStyle;
    class G,H upStyle;
    class L,M exitStyle;
```

### Module Specifications Table

| Component | Layer Type | Channels (In $\to$ Out) | Purpose / Details |
| :--- | :--- | :---: | :--- |
| **Shallow Head** | `nn.Conv2d` | $3 \to 64$ | Low-res radiance ko 64-dim latent features me convert karta hai. |
| **HAB Trunk (6x)** | `HybridAttentionBlock` | $64 \to 64$ | Spatial self-attention aur channel attention ko interleave karta hai. |
| **W-MSA** | `WindowMultiHeadAttention` | 4 heads (16 dim/head) | $8 \times 8$ local patch ke andar pixel-to-pixel correlations nikalta hai. |
| **CAB** | `ChannelAttentionBlock` | $64 \to 4 \to 64$ ($r=16$) | Inter-band radiometric balance preserve karta hai. |
| **Upsamplers** | `PixelShuffle(2)` $\times 2$ | $64 \to 256 \to 64$ | Spatial resolution ko $2\times \times 2\times = 4\times$ karta hai. |
| **Synthesis Tail** | `Conv + LReLU + Conv` | $64 \to 32 \to 3$ | Sharp high-frequency details synthesize karta hai. |

---

## 3. Workflow: Training Se Prediction Tak

```mermaid
flowchart TD
    subgraph DataIngestion["1. Satellite Ingestion & Preprocessing"]
        S1["User AOI Bounding Box (Web UI Leaflet Map)"] --> S2["Sentinel-2 L2A STAC API (Planetary Computer / Copernicus)"]
        S2 --> S3["Extract Optical Bands: B04 (Red), B03 (Green), B02 (Blue), B08 (NIR)"]
        S3 --> S4["Surface Reflectance Normalization: [0, 10000] -> [0.0, 1.0]"]
        S4 --> S5["SCL Cloud/Shadow Quality Masking"]
        S5 --> S6["Patch Tiling: Split into 64x64 Tiles with 8-pixel Window Padding"]
    end

    subgraph CoreEngine["2. HAT Core Super-Resolution Engine (4x GSD)"]
        S6 --> H1["Shallow Conv2d Feature Projection (3 -> 64)"]
        H1 --> H2["6x Cascaded Hybrid Attention Blocks (HAB)"]
        H2 -->|Window Self-Attention| H3["Sharp Edge Boundary Extraction (Roads & Parcels)"]
        H2 -->|Channel Attention| H4["Spectral Inter-Band Radiance Preservation"]
        H3 --> H5["Dual Sub-Pixel PixelShuffle (2x * 2x = 4x)"]
        H4 --> H5
        H5 --> H6["High-Frequency Detail Synthesis Tail"]
        H6 --> H7["Recombination with 4x Bicubic Anchor"]
        H7 --> H8["Radiometric Color Clamp (Delta_color <= 0.75)"]
    end

    subgraph TrustLayer["3. Scientific Trust & Quality Validation Layer"]
        H8 --> T1["Cycle Invariance Check: Downsample 4x SR vs Raw Sentinel-2"]
        T1 -->|Cycle MAE < 0.02| T2["Radiance Conservation Passed"]
        T1 -->|Cycle MAE >= 0.02| T3["Hallucination Alert Triggered"]
        H8 --> T4["Spectral Angle Mapper: SAM < 6.0 deg vs Reference"]
        H8 --> T5["Epistemic Uncertainty Estimation (Monte-Carlo Sampling)"]
        T2 --> T6["Fused Scientific Confidence Score (0% - 100%)"]
        T4 --> T6
        T5 --> T6
    end

    subgraph Delivery["4. Dual Operational Delivery Paths"]
        T6 --> D1["Path A: Interactive Web UI Display"]
        D1 --> D2["Reinhard LAB Dynamic Color Transfer (Matched to Basemap)"]
        D2 --> D3["Before / After Split-Slider & Live Confidence Heatmap"]

        T6 --> D4["Path B: Certified Geospatial GIS Export"]
        D4 --> D5["Raw Untransferred Reflectance Preservation"]
        D5 --> D6["GeoTIFF 2.5m Multi-Spectral Export (EPSG CRS Embedded)"]
        D5 --> D7["High-Resolution NDVI / Water Index Computation (spyndex)"]
        D5 --> D8["Cadastral Demarcation & Infrastructure Vectorization"]
    end

    classDef ingStyle fill:#0f172a,stroke:#38bdf8,stroke-width:2px,color:#f8fafc;
    classDef engStyle fill:#14532d,stroke:#22c55e,stroke-width:2px,color:#f8fafc;
    classDef valStyle fill:#451a03,stroke:#f59e0b,stroke-width:2px,color:#f8fafc;
    classDef delStyle fill:#3b0764,stroke:#c084fc,stroke-width:2px,color:#f8fafc;

    class S1,S2,S3,S4,S5,S6 ingStyle;
    class H1,H2,H3,H4,H5,H6,H7,H8 engStyle;
    class T1,T2,T3,T4,T5,T6 valStyle;
    class D1,D2,D3,D4,D5,D6,D7,D8 delStyle;
```

---

## 4. Hyperparameters & Unka Effect

| Hyperparameter | Value | Description | Tune Karne Par Effect |
| :--- | :---: | :--- | :--- |
| **`window_size`** | `8` | Attention calculate karne ka local patch size ($8 \times 8$). | **Bada karne par:** Global context behtar hoga lekin computation $O(M^2)$ speed se slow ho jayegi. **Chhota karne par:** Fast chalega par broad road context miss ho sakta hai. |
| **`num_heads`** | `4` | Self-attention heads ki sankhya. | Head badhane se multi-scale patterns pakadte hain, par GPU memory zyada lagti hai. |
| **`num_blocks`** | `6` | Trunk me HAB blocks ki sankhya. | Depth badhane se finer edges aate hain; 6 blocks accuracy aur inference speed ka perfect sweet spot hai. |
| **`reduction_ratio`** | `16` | Channel Attention bottleneck ratio ($C/16$). | Parameter size kam rakhta hai; agar 8 karein to thodi better spectral distinction milegi par parameters badhenge. |
| **`dropout_rate`** | `0.15` | Spatial dropout probability. | Active inference me Monte-Carlo uncertainty estimation ke liye variance generate karta hai. Overfitting rokta hai. |
| **`lambda_sam`** | `0.05` | Training loss me Spectral Angle Mapper ka weight. | Isko zyada rakhne se true-color integrity perfect rehti hai, par over-smooth edges ho sakte hain. |
