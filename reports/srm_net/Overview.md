# SRM-Net (Residual Attention CNN + MC-Dropout) — Overview

## 1. Introduction (Model Kya Hai?)

**SRM-Net (Super-Resolution Mapping Network)** ek lightweight, fast Deep Residual Convolutional Neural Network (CNN) hai jisme **Squeeze-and-Excitation Channel Attention** aur **Active Test-Time Monte-Carlo (MC) Dropout** shamil hai.

Simple shabdon me:
> **SRM-Net ka main kaam hai satellite images ko real-time me 4x upscale karna (10m se 2.5m) aur saath hi yeh batana ki model ko apne banaye gaye pixels par kitna bharosa hai (Active Uncertainty & Hallucination Heatmap).**

Heavy Transformer models (jaise HAT) visual quality me sharp hote hain par bohot slow hote hain. **SRM-Net ko speed aur trustworthiness ke liye banaya gaya hai**:
1. **Lightweight (< 1 Million parameters):** CPU par sirf **472 ms** me 4x upscale kar deta hai.
2. **Monte-Carlo Dropout (Active Bayesian Reasoning):** Inference ke waqt dropout ko on rakh kar yeh 8 bar stochastic prediction karta hai. Jin jagahon par model guess kar raha hota hai ya confuse hota hai, wahan variance calculate karke **Hallucination Risk Heatmap** render kar deta hai.

---

## 2. Real-Life Use Cases (Asli Duniya me Kaha Use Hota Hai?)

SRM-Net ko HACKNATION platform me **Real-Time Interactive Dashboard & Uncertainty Engine** banaya gaya hai. Real-life use cases:

### 1. Live Interactive Web Dashboard & Tile Exploration
- **Problem:** GIS officers aur disaster management teams ko web browser me instant zoom-in/zoom-out aur split-slider dekhna hota hai. Agar model 15 second lagaye to user experience kharab ho jata hai.
- **SRM-Net Solution:** Sub-500ms single-pass latency ki wajah se user map par kisi bhi AOI par drag karta hai to bina lag ke turant 2.5m view render ho jata hai.

### 2. Hallucination-Aware Land Demarcation & Legal Auditing
- **Problem:** AI models kabhi-kabhi shadows ya blurred mitti ko concrete road ya building samajhkar galat structure bana dete hain (hallucination), jisse zameen ke legal mamle me vivad ho sakta hai.
- **SRM-Net Solution:** SRM-Net har pixel par **Epistemic Uncertainty Heatmap** generate karta hai. Agar kisi pixel par confidence < 70% ho, to dashboard red alert show karta hai ki *"Is boundary ko physically verify karein"*.

### 3. Rapid Flood & Cyclone Disaster Reconnaissance (NDRF / State SDRF)
- **Problem:** Baadh ke waqt command center ko pure district ka quick scan chahiye hota hai taaki relief camps aur cut-off villages ki sankhya pata chal sake.
- **SRM-Net Solution:** 4.3x fast stochastic ensemble (3.7 seconds me 8 passes) ki badolat pure taluk ka risk assessment minutes me ho jata hai.

### 4. Edge-Server Deployment in Remote Field Offices
- **Problem:** North-East ke zila mukhyalayon (District HQs) me high-end Nvidia GPUs uplabdh nahi hote, sirf basic Intel Core-i5 laptops hote hain.
- **SRM-Net Solution:** Sirf ~210 MB RAM consumption ke sath basic CPU par smoothly run hota hai.

---

## 3. Advantages (Fayde)

- **Blazing Fast Inference (~472 ms on CPU):** HAT (807 ms) aur Real-ESRGAN (5850 ms) ke mukable sabse tez.
- **Native Epistemic Uncertainty Mapping:** Har pixel ka standard deviation $\sigma(x, y)$ calculate karta hai jisse scientific trust maintain rehta hai.
- **Ultra-Compact Memory Footprint (~210 MB RAM):** Kisi bhi lightweight laptop ya container me run ho sakta hai.
- **Strong Structural Fidelity (SSIM 0.1556):** Transformer jitna heavy na hone ke bawajood crisp edges produce karta hai.
- **No Heavy GPU Dependency:** Pure CPU environment me deployable.

---

## 4. Disadvantages (Kamiyan / Limitations)

- **Slightly Softer Edges than HAT:** Transformer ke global window attention ke mukable complex urban rooftops par edges thode soft ho sakte hain (SSIM 0.1556 vs HAT 0.1667).
- **Stochastic Multi-pass Overhead:** Single pass 472ms leta hai, lekin agar 8-pass Monte Carlo uncertainty chalayein to ~3.7 seconds lagte hain (halaanki yeh HAT ke 14s se bohot fast hai).
- **Channel Attention Bottleneck:** $1\times 1$ global pooling fine spatial micro-textures ko kabhi-kabhi average-out kar sakti hai.

---

## 5. Assumptions (Kis Data / Conditions Pe Best Kaam Karta Hai?)

1. **Standard Normalized Reflectance [0.0, 1.0]:** Sentinel-2 L2A BOA reflectance values $0$ se $1$ range me honi chahiye.
2. **Monte-Carlo Sample Count ($T \ge 8$):** Reliable variance map pane ke liye kam se kam 8 stochastic forward passes zaroori hain.
3. **Cloud-Masked AOI:** Cloudy pixels par variance artificially high aayegi, isliye SCL (Scene Classification Layer) cloud mask apply karna zaroori hai.
4. **Scale Factor 4x:** Architecture Sentinel-2 10m bands (B02, B03, B04, B08) ko 2.5m GSD me map karne ke liye specifically tuned hai.
