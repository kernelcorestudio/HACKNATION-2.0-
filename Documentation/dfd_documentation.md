# Data Flow Diagram (DFD): HACKNATION (GEO-SRM)

This document provides a detailed Data Flow Diagram (DFD) analysis of the Super Resolution Mapping system. It is broken down into three levels of granularity: Level 0 (Context), Level 1 (High-Level Subsystems), and Level 2 (Detailed AI Data Flow).

---

## 1. Level 0 DFD: Context Diagram
The Context Diagram shows the entire GEO-SRM System as a single process and how it interacts with external entities (Users and External Cloud APIs).

```mermaid
flowchart TD
    %% External Entities
    User((End User / Judge))
    CDSE((Copernicus CDSE API))
    
    %% Main System
    System[0.0 GEO-SRM System]
    
    %% Data Flows
    User -- "AOI Bounding Box / Custom GeoTIFF" --> System
    System -- "Tile Request (Lat/Lon, Cloud limits)" --> CDSE
    CDSE -- "Sentinel-2 L2A Tile (10m)" --> System
    
    System -- "Interactive SR Map (2.5m)" --> User
    System -- "Confidence Heatmap & Validation Metrics" --> User
    System -- "Extracted Infrastructure GeoJSON" --> User
```

---

## 2. Level 1 DFD: Subsystems Breakdown
Level 1 breaks the main system into its primary functional processes. It shows how data moves between data ingestion, the AI processing engine, and the frontend presentation layers.

```mermaid
flowchart TD
    %% External Entities
    User((End User))
    CDSE((Copernicus CDSE API))
    
    %% Data Stores
    D1[(Local Tile Cache)]
    D2[(Ground Truth SPOT Ref)]
    
    %% Processes
    P1[1.0 Handle UI Requests]
    P2[2.0 Fetch & Preprocess Data]
    P3[3.0 Execute PyTorch Inference]
    P4[4.0 Compute USP Validation]
    P5[5.0 Format Output / Render]
    
    %% Data Flows
    User -- "Coordinates / Upload" --> P1
    P1 -- "Clean Bounding Box" --> P2
    
    P2 -- "Query Metadata" --> CDSE
    CDSE -- "Raw 10m Imagery" --> P2
    P2 -- "Save/Load Raw Tile" --> D1
    
    P2 -- "Normalized 10m Array" --> P3
    P3 -- "Super-Resolved 2.5m Array" --> P4
    P3 -- "MC-Dropout Variance Map" --> P4
    
    P4 -- "Query Paired Reference" --> D2
    D2 -- "SPOT 1.5m Reference" --> P4
    
    P4 -- "Metrics & Heatmap Array" --> P5
    P3 -- "Super-Resolved 2.5m Array" --> P5
    
    P5 -- "Base64 PNGs & GeoJSON" --> P1
    P1 -- "Visual Dashboard" --> User
```

---

## 3. Level 2 DFD: Deep Learning & USP Engine
Level 2 dives deep into Process 3.0 and 4.0, exposing the intricate data manipulation inside the PyTorch Super Resolution engine and the Hallucination-Aware Uncertainty Pipeline.

```mermaid
flowchart TD
    %% Data Inputs
    Input[Normalized 10m Input Array]
    
    %% Processes (Inference)
    P31[3.1 Extract Overlapping Patches]
    P32[3.2 Stochastic Forward Passes]
    P33[3.3 Calculate Mean Patch]
    P34[3.4 Calculate Epistemic Variance]
    P35[3.5 Reconstruct Image with 2D Cosine Blending]
    
    %% Processes (USP)
    P41[4.1 Compute Cycle Consistency]
    P42[4.2 Compute Spectral Angle Mapper]
    P43[4.3 Fuse Hallucination Risk Index]
    
    %% Data Flows (Inference)
    Input --> P31
    P31 -- "64x64 Tensors" --> P32
    
    P32 -- "N Pass Outputs" --> P33
    P32 -- "N Pass Outputs" --> P34
    
    P33 -- "Mean 256x256 Patches" --> P35
    P35 -- "Full SR Image (2.5m)" --> Output_SR[Final 2.5m Output]
    
    P34 -- "Patch Variance" --> Output_Var[Pixel Variance Map]
    
    %% Data Flows (USP)
    Input --> P41
    Output_SR --> P41
    P41 -- "MAE Residuals" --> P43
    
    Input --> P42
    Output_SR --> P42
    P42 -- "SAM Degrees" --> P43
    
    Output_Var --> P43
    P43 -- "Normalized Heatmap Array" --> Output_Heatmap[Confidence Heatmap]
```

## Summary of the DFD Levels
*   **Level 0** shows that a user simply gives a location (Bounding Box) and receives a high-resolution map and trust metrics in return, while the system quietly handles the satellite communication in the background.
*   **Level 1** explains that the backend maintains a local database (`D1` Cache) to save bandwidth and gracefully separates data fetching from machine learning processing.
*   **Level 2** is the most complex, illustrating the mathematical operations where a single satellite tile is sliced into tensor patches (`P3.1`), pushed through active PyTorch Dropout layers multiple times (`P3.2`), and finally audited by the SAM (`P4.2`) and Cycle Consistency (`P4.1`) modules to detect hallucinations.
