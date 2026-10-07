# HACKNATION — AI/ML Architecture & Workflow Documentation

Yeh documentation project me use hue saare **Machine Learning (ML)** aur **Deep Learning (DL)** models, unke workflows, aur unke use-cases ko detail me explain karti hai. Project ka main goal Sentinel-2 satellite imagery (10m resolution) ko super-resolve karke <4m (2.5m) tak le jana hai, saath me ek "Hallucination-Aware Uncertainty" layer provide karna hai taaki AI ke banaye hue fake features (hallucinations) ko detect kiya ja sake.

---

## 1. Core Super Resolution (SR) Models
Image ko 10m se 2.5m (4x upscaling) karne ke liye project me state-of-the-art Deep Learning models ka use kiya gaya hai. User frontend se inme se koi bhi model select kar sakta hai:

### A. HAT (Hybrid Attention Transformer)
- **Kya hai:** Ek advanced Transformer-based model (CVPR 2023) jo image super-resolution me highest quality deta hai.
- **Kyu use kiya:** Transformers global aur local features ko bahut acche se samajhte hain, jisse sharp edges aur textures preserve hote hain. Remote sensing me fields aur buildings ki boundaries clear chahiye hoti hain, isliye HAT ek premium choice hai.

### B. CARN (Cascading Residual Network) / ESRGAN
- **Kya hai:** `sentinel2_superresolution` repo (EVOLAND project) se liya gaya pretrained model. CARN ek lightweight CNN (Convolutional Neural Network) based model hai, jabki ESRGAN ek Generative Adversarial Network hai.
- **Kyu use kiya:** Yeh models specially Sentinel-2 data (WorldStrat dataset) par trained hain. Inka inference fast hota hai aur yeh baseline ki tarah work karte hain.

### C. SRM-Net (Residual Attention + MC-Dropout)
- **Kya hai:** Yeh project ka custom variant hai jisme Residual blocks ke saath Monte-Carlo Dropout (MC-Dropout) integrate kiya gaya hai.
- **Kyu use kiya:** Normal models sirf ek sharp image output karte hain, par hume **uncertainty** bhi chahiye thi. MC-Dropout inference ke time active rehta hai, jisse model ek hi image ko kai baar process karta hai aur variance nikalta hai (ispe niche USP section me detail hai).

---

## 2. The USP: Hallucination-Aware Uncertainty Module
Sirf image ko sharp karna remote sensing me dangerous ho sakta hai kyuki AI aisi roads ya boundaries bana sakta hai jo real me exist nahi karti (Hallucinations). Isko solve karne ke liye humne ek "Scientific Trust Layer" banayi hai.

### A. Monte-Carlo Dropout (MC-Dropout)
- **Kaise kaam karta hai:** Inference (testing) ke time par hum dropout layers ko active rakhte hain. Isse network ke kuch neurons randomly band ho jate hain. Hum image ko ek baar nahi, balki `N` times (e.g., 8 passes) model se pass karte hain.
- **Kyu use kiya:** Har pass me thoda alag output aata hai. Agar AI confidently kisi feature (jaise road) ko upscale kar raha hai, toh har pass me road wahi banegi (low variance). Par agar AI guess/hallucinate kar raha hai, toh har pass me output alag aayega (high variance). In sab passes ka standard deviation nikal kar hum ek **Confidence Heatmap** generate karte hain.

### B. ESA `opensr-test` Benchmark
- **Kya hai:** European Space Agency (ESA) ka peer-reviewed framework jo spectral consistency aur hallucination check karta hai.
- **Kaise kaam karta hai:** Yeh check karta hai ki kya upscaled image ke colors (spectral bands) original Sentinel-2 data se physically match karte hain ya nahi.

---

## 3. Pre-processing ML Models
Data ko SR model me bhejne se pehle clean karna zaroori hai.

### A. Multi-Spectral Optical Cloud Masking
- **Kya hai:** Sentinel-2 multi-spectral bands (B04, B03, B02 visible + B08 NIR) ka physical optical algorithm jo clouds aur cloud-shadows detect karta hai.
- **Kyu use kiya:** Satellite images me badal (clouds) aam baat hai. Agar badal wali image model me jayegi, toh SR output kharab ho jayega. Yeh engine optical whiteness aur NIR reflection se badalo ko mask out kar deta hai.

---

## 4. Downstream Tasks: Infrastructure Detection
SR output (2.5m) ka real-world use-case show karne ke liye humne ek aur AI layer add ki hai jo objects detect karti hai.

### A. Segment Anything Model (SAM) via `segment-geospatial`
- **Kya hai:** Meta ka SAM (specifically LangSAM / Grounding DINO variant) jo text prompts ke basis par image segmentation karta hai.
- **Kyu use kiya:** Bina koi naya model train kiye (zero-shot learning), hum sirf text prompt `"bridge, road, building"` dete hain, aur yeh AI upscaled map par in infrastructures ko detect karke unka GeoJSON mask bana deta hai. Isse judges ko prove hota hai ki hamara 2.5m data kitna useful hai.

---

## 5. End-to-End AI/ML Workflow (Step-by-Step)

1. **Ingestion & Pre-processing:**
   - Copernicus CDSE API se Sentinel-2 (10m) tiles download hoti hain.
   - Multi-spectral optical engine clouds ko detect aur mask out karta hai.
   - Image ko chote patches (e.g., 128x128) me tile kiya jata hai kyuki DL models puri badi satellite image ek sath memory me process nahi kar sakte.

2. **Super-Resolution (Forward Pass):**
   - LR (Low-Res) tiles ko selected SR model (HAT/SRM-Net) me feed kiya jata hai.
   - Agar MC-Dropout selected hai (USP feature), toh model 8 baar alag-alag prediction karta hai.

3. **Post-processing (Co-registration & Confidence):**
   - **Co-registration:** AROSICS (Robust feature matching) ka use karke output aur Ground Truth ko exactly pixel-to-pixel align kiya jata hai taaki metrics sahi aayen.
   - **Variance Calculation:** 8 passes ki variance nikal kar `confidence.tif` (heatmap) banai jati hai.

4. **Metrics Calculation (Validation):**
   - No-Reference IQA (Blind Quality Assessment) ke liye `pyiqa` ka use hota hai jo **NIQE** aur **BRISQUE** scores nikalta hai.
   - Agar Ground Truth available hai, toh **PSNR**, **SSIM**, aur **SAM** (Spectral Angle Mapper) calculate hote hain.

5. **Application Layer:**
   - Output ko LangSAM me feed karke GeoJSON vectors (Roads/Bridges) extract hote hain aur frontend par render kiye jate hain.

---

## 6. Evaluation Metrics (Model ko judge kaise kar rahe hain?)

| Metric | Full Form | Kya check karta hai? | Ideal Value |
|---|---|---|---|
| **PSNR** | Peak Signal-to-Noise Ratio | Pixel-to-pixel kitna accurate reconstruction hua hai. | > 30 dB |
| **SSIM** | Structural Similarity Index | Edges aur texture kitne similar hain Ground truth se. | > 0.85 |
| **SAM** | Spectral Angle Mapper | Colors aur light bands model ne badal toh nahi diye (Spectral Fidelity). | < 3.5° (Lower is better) |
| **ERGAS** | Relative Dimensionless Global Error | Satellite imagery ka standard error metric. | < 3.0 (Lower is better) |
| **NIQE/BRISQUE** | Natural Image Quality Evaluator | Agar GT (Ground Truth) nahi hai, toh image kitni "natural" lag rahi hai. | Lower is better |

Yeh puri AI pipeline ensure karti hai ki output sirf visually accha na dikhe, balki scientifically accurate aur trustable ho.
