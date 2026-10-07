# Tech Stack Documentation: HACKNATION (GEO-SRM)

This document provides a detailed breakdown of the technologies, frameworks, and libraries utilized in the Super Resolution Mapping (SRM) platform. The tech stack is divided into Backend, AI/ML Engine, Geospatial Processing, and Frontend layers.

---

## 1. Backend Infrastructure & API Layer
The backend is designed for high-performance, asynchronous processing to handle heavy machine learning inferences and satellite tile fetching seamlessly.

*   **Language:** `Python 3.10+`
*   **Web Framework:** `FastAPI` (>=0.100.0)
    *   **Why:** Chosen for its extremely fast performance, native asynchronous support (`async/await`), and automatic OpenAPI documentation generation.
*   **ASGI Server:** `Uvicorn` (>=0.23.0)
    *   **Why:** Lightning-fast ASGI server implementation used to serve the FastAPI application.
*   **Data Validation:** `Pydantic` (>=2.0.0)
    *   **Why:** Enforces strict type hints and data validation for API requests (e.g., ensuring Bounding Boxes have correct float coordinates).
*   **File Handling:** `python-multipart` (>=0.0.6)
    *   **Why:** Required by FastAPI for handling custom user uploads of GeoTIFF or PNG satellite tiles.

## 2. Artificial Intelligence & Machine Learning (AI/ML)
The core super-resolution and uncertainty estimation engine relies on state-of-the-art Deep Learning libraries.

*   **Core Framework:** `PyTorch` (>=2.0.0) & `Torchvision` (>=0.15.0)
    *   **Why:** The industry standard for Deep Learning research and deployment. It handles the heavy tensor computations on both CPU and CUDA-enabled GPUs (e.g., T4/A100).
*   **Supported Architectures:**
    *   `SRMNet` (Residual Attention Network)
    *   `EVOLAND CARN` (Cascading Residual Network)
    *   `SwinIR` / `HAT` (Transformer-based backbones)
*   **Mathematical Computing:** `NumPy` (>=1.24.0) & `SciPy` (>=1.10.0)
    *   **Why:** Used for multi-dimensional array manipulation, MC-Dropout variance calculations, and low-frequency cycle consistency via `scipy.ndimage.zoom`.

## 3. Geospatial & Image Processing
Libraries specifically tailored for handling multi-band satellite data and image quality assessments.

*   **Satellite API Client:** `sentinelhub-py` (>=3.11.0)
    *   **Why:** The official Python interface used to securely query and fetch real-time Sentinel-2 L2A tiles from the Copernicus Data Space Ecosystem (CDSE).
*   **Image Processing:** `scikit-image` (>=0.22.0)
    *   **Why:** Used to calculate professional image quality metrics like **PSNR** (Peak Signal-to-Noise Ratio) and **SSIM** (Structural Similarity Index) against Ground Truth reference images.
*   **Remote Sensing Data Formats:** `tifffile` (>=2023.0.0)
    *   **Why:** Specialized library to read/write 16-bit and 32-bit float multi-band GeoTIFF formats without precision loss.
*   **Image Rendering:** `Pillow` (>=10.0.0)
    *   **Why:** Used to convert normalized float32 reflectance arrays into renderable Base64 PNG formats for the web frontend.

## 4. Frontend & User Interface
A zero-build-step, pure Vanilla JS frontend to keep the application lightweight, highly performant, and easy to deploy.

*   **Core Languages:** `HTML5`, `CSS3` (Custom styles, no Tailwind/Bootstrap to ensure total design control), `JavaScript (ES6+)`.
*   **Web Mapping:** `Leaflet.js` (v1.9.4) & `Leaflet.draw`
    *   **Why:** The leading open-source library for interactive maps. Used to render the world map and allow users to draw custom Area of Interest (AOI) bounding boxes.
*   **High-Resolution Image Rendering:** `OpenSeadragon` (v4.1.1)
    *   **Why:** A highly specialized library for rendering massive, high-resolution images smoothly. It provides the "Deep Zoom" capability and is used to synchronize the 10m and 2.5m viewports side-by-side without crashing the browser.
*   **Fonts:** `Google Fonts` (Inter, Space Grotesk, JetBrains Mono) for clean, professional typography.

## 5. Deployment & Cloud Tools
*   **Environment Variables:** `python-dotenv` (>=1.0.0) - Safely manages Copernicus API keys without hardcoding them into the repository.
*   **GPU Cloud Prototyping:** `Google Colab` & `ngrok` - Used to host the FastAPI server on a free T4 GPU and tunnel the localhost port to a public URL for hackathon demonstrations.
