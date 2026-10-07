/**
 * HACKNATION - Super Resolution Mapping Frontend Application
 * Features:
 * 1. AOI Selection -> Real Sentinel-2 L2A tile fetching via Copernicus CDSE
 * 2. Deep Learning Super Resolution Inference (10m -> 2.5m GSD)
 * 3. Monte-Carlo Dropout Uncertainty & Hallucination Mapping (USP)
 * 4. Section 12: Automated Infrastructure Detection (Bridges, Roads, Buildings) & GeoJSON Export
 * 5. Section 13: High-Resolution Deep-Zoom Viewer (OpenSeadragon with Synchronized Viewports)
 */

document.addEventListener("DOMContentLoaded", () => {
    // Set speed to 0.75x. Going too slow (like 0.25) drops frames and causes visual lag.
    const bgVideo = document.querySelector('.landing-video-bg');
    if (bgVideo) bgVideo.playbackRate = 0.75; // Smooth slow down

    // Landing Page Transition Logic
    const landingPage = document.getElementById('landing-page');
    const dashboardContainer = document.getElementById('dashboard-container');
    const btnStartAnalysis = document.getElementById('btn-start-analysis');

    function launchDashboard() {
        if (!landingPage || !dashboardContainer) return;
        landingPage.classList.add('fade-out');
        setTimeout(() => {
            landingPage.style.display = 'none';
            dashboardContainer.style.display = 'block';
            void dashboardContainer.offsetWidth;
            dashboardContainer.style.opacity = '1';

            // Immediately refit Leaflet map & OpenSeadragon viewers
            if (typeof map !== 'undefined' && map) map.invalidateSize();
            if (typeof spectralMap !== 'undefined' && spectralMap) spectralMap.invalidateSize();
            if (typeof resetAllViewers === 'function') resetAllViewers();

            setTimeout(() => {
                window.dispatchEvent(new Event('resize'));
                if (typeof map !== 'undefined' && map) map.invalidateSize();
                if (typeof resetAllViewers === 'function') resetAllViewers();
                if (state.activePreset) {
                    selectPreset(state.activePreset);
                }
            }, 150);

            setTimeout(() => {
                if (typeof resetAllViewers === 'function') resetAllViewers();
            }, 550);
        }, 500);
    }

    function returnToLanding() {
        if (!landingPage || !dashboardContainer) return;
        dashboardContainer.style.opacity = '0';
        setTimeout(() => {
            dashboardContainer.style.display = 'none';
            landingPage.style.display = 'block';
            void landingPage.offsetWidth;
            landingPage.classList.remove('fade-out');
            landingPage.scrollTop = 0;
        }, 300);
    }

    ['btn-start-analysis', 'btn-explore-demo', 'btn-nav-demo', 'btn-open-console'].forEach(id => {
        const btn = document.getElementById(id);
        if (btn) {
            btn.addEventListener('click', (e) => {
                e.preventDefault();
                launchDashboard();
            });
        }
    });

    const btnBackToPortal = document.getElementById('btn-back-to-portal');
    if (btnBackToPortal) {
        btnBackToPortal.addEventListener('click', (e) => {
            e.preventDefault();
            returnToLanding();
        });
    }

    // Application State
    const state = {
        activePreset: "punjab_agri",
        activeBbox: [75.30, 30.55, 75.36, 30.60],
        mcSamples: 2,
        confidenceActive: false,
        confidenceOpacity: 0.75,
        infraActive: false,
        infraOpacity: 0.85,
        cloudActive: false,
        cloudOpacity: 0.80,
        viewMode: "split",
        presets: [],
        scaleFactor: 4,
        modelName: "hat",
        applyUnsharp: false,
        applyRealESRGANSharpen: false,
        acquisitionMode: "single",
        latestLrUrl: null,
        latestSrUrl: null,
        latestHrUrl: null,
        latestConfUrl: null,
        latestInfraUrl: null,
        latestCloudUrl: null,
        cloudWarning: null,
        cloudCoveragePct: 0.0,
        infraSummary: null,
        blockchainRecord: null,
        visualizationMode: "true_color",
        activeTopView: "sr",
        activeSpectralMode: "ndvi",
        latestSpectralLrUrl: null,
        latestSpectralSrUrl: null
    };

    // Top-Level Navigation & App Views
    const navBtnSr = document.getElementById("nav-btn-sr");
    const navBtnSpectral = document.getElementById("nav-btn-spectral");
    const viewSrMapping = document.getElementById("view-sr-mapping");
    const viewSpectralAnalysis = document.getElementById("view-spectral-analysis");

    // DOM Elements
    const presetsList = document.getElementById("presets-list");
    const aoiCoordsBadge = document.getElementById("aoi-coords-badge");
    const inputCustomLat = document.getElementById("input-custom-lat");
    const inputCustomLon = document.getElementById("input-custom-lon");
    const btnGotoCoords = document.getElementById("btn-goto-coords");
    const sliderMcSamples = document.getElementById("slider-mc-samples");
    const valMcSamples = document.getElementById("val-mc-samples");
    const selectScale = document.getElementById("select-scale");
    const selectModel = document.getElementById("select-model");
    const selectVisMode = document.getElementById("select-vis-mode");
    const checkRealESRGANSharpen = document.getElementById("check-realesrgan-sharpen");
    const checkUnsharp = document.getElementById("check-unsharp");
    const btnModeSingle = document.getElementById("btn-mode-single");
    const btnModeMulti = document.getElementById("btn-mode-multi");

    // Zoom Controls DOM
    const btnZoomIn = document.getElementById("btn-zoom-in");
    const btnZoomOut = document.getElementById("btn-zoom-out");
    const btnZoomReset = document.getElementById("btn-zoom-reset");
    const zoomLevelBadge = document.getElementById("zoom-level-badge");
    const inspectorDims = document.getElementById("inspector-dims");

    // Live Verification DOM
    const verifCheckpoint = document.getElementById("verif-checkpoint");
    const verifDropout = document.getElementById("verif-dropout");
    const verifVariance = document.getElementById("verif-variance");
    const verifDimensions = document.getElementById("verif-dimensions");
    const verifUnsharp = document.getElementById("verif-unsharp");
    const verifReference = document.getElementById("verif-reference");
    const verifTemporal = document.getElementById("verif-temporal");

    // Action buttons & inputs
    const btnRunSr = document.getElementById("btn-run-sr");
    const btnTriggerUpload = document.getElementById("btn-trigger-upload");
    const fileUpload = document.getElementById("file-upload");
    const btnDownloadSr = document.getElementById("btn-download-sr");
    const btnDownloadConf = document.getElementById("btn-download-conf");
    const btnDownloadGeojson = document.getElementById("btn-download-geojson");

    // View toggles & containers
    const tabSplit = document.getElementById("tab-split");
    const tabSide = document.getElementById("tab-side");
    const sliderContainer = document.getElementById("slider-container");
    const sideBySideContainer = document.getElementById("side-by-side-container");
    const srWrapper = document.getElementById("sr-wrapper");
    const sliderHandle = document.getElementById("slider-handle");
    const loadingScrim = document.getElementById("loading-scrim");

    // Heatmap, Infrastructure & Cloud Toolbar Controls
    const toggleConfidencePill = document.getElementById("toggle-confidence-pill");
    const sliderConfidenceOpacity = document.getElementById("slider-confidence-opacity");
    const toggleInfraPill = document.getElementById("toggle-infra-pill");
    const sliderInfraOpacity = document.getElementById("slider-infra-opacity");
    const toggleCloudPill = document.getElementById("toggle-cloud-pill");
    const sliderCloudOpacity = document.getElementById("slider-cloud-opacity");
    const cloudWarningBanner = document.getElementById("cloud-warning-banner");
    const cloudWarningText = document.getElementById("cloud-warning-text");
    const btnDismissCloudWarning = document.getElementById("btn-dismiss-cloud-warning");

    // Scientific Metrics HUD Elements
    const metricConfidence = (document.getElementById("a-metric-confidence") || document.getElementById("metric-confidence"));
    const metricUncertainty = (document.getElementById("a-metric-uncertainty") || document.getElementById("metric-uncertainty"));
    const metricSam = (document.getElementById("a-metric-sam") || document.getElementById("metric-sam"));
    const metricCycle = (document.getElementById("a-metric-cycle") || document.getElementById("metric-cycle"));
    const metricPsnr = (document.getElementById("a-metric-psnr") || document.getElementById("metric-psnr"));
    const metricSsim = (document.getElementById("a-metric-ssim") || document.getElementById("metric-ssim"));
    const metricErgas = (document.getElementById("a-metric-ergas") || document.getElementById("metric-ergas"));
    const metricRuntime = (document.getElementById("a-metric-runtime") || document.getElementById("metric-runtime"));

    // Dynamic Quality Assessment Panels (Paired vs No-Reference)
    const panelValidationPaired = document.getElementById("panel-validation-paired");
    const panelValidationNr = document.getElementById("panel-validation-nr");
    const metricNrNiqe = document.getElementById("metric-nr-niqe");
    const metricNrBrisque = document.getElementById("metric-nr-brisque");
    const metricNrConfidence = document.getElementById("metric-nr-confidence");
    const metricNrCycle = document.getElementById("metric-nr-cycle");

    // Infrastructure & Multi-Class HUD Elements (Section 12)
    const metricBuildings = (document.getElementById("a-metric-buildings") || document.getElementById("metric-buildings"));
    const metricRoads = (document.getElementById("a-metric-roads") || document.getElementById("metric-roads"));
    const metricInfraConf = (document.getElementById("a-metric-infra-conf") || document.getElementById("metric-infra-conf"));

    // =========================================================================
    // 1. OpenSeadragon Deep-Zoom High-Resolution Viewers (Section 13)
    // =========================================================================
    let viewerLR = null;
    let viewerSR = null;
    let viewerSideLR = null;
    let viewerSideSR = null;
    let viewerSideHR = null;

    // Overlay Elements for SR Viewer
    const confOverlay = document.createElement("img");
    confOverlay.id = "osd-conf-overlay";
    confOverlay.style.width = "100%";
    confOverlay.style.height = "100%";
    confOverlay.style.pointerEvents = "none";
    confOverlay.style.mixBlendMode = "screen";
    confOverlay.style.display = "none";

    const infraOverlay = document.createElement("img");
    infraOverlay.id = "osd-infra-overlay";
    infraOverlay.style.width = "100%";
    infraOverlay.style.height = "100%";
    infraOverlay.style.pointerEvents = "none";
    infraOverlay.style.display = "none";

    // Cloud Mask Occlusion Overlay Elements (Both Input LR and Output SR)
    const cloudOverlaySR = document.createElement("img");
    cloudOverlaySR.id = "osd-cloud-overlay-sr";
    cloudOverlaySR.style.width = "100%";
    cloudOverlaySR.style.height = "100%";
    cloudOverlaySR.style.pointerEvents = "none";
    cloudOverlaySR.style.display = "none";

    const cloudOverlayLR = document.createElement("img");
    cloudOverlayLR.id = "osd-cloud-overlay-lr";
    cloudOverlayLR.style.width = "100%";
    cloudOverlayLR.style.height = "100%";
    cloudOverlayLR.style.pointerEvents = "none";
    cloudOverlayLR.style.display = "none";

    function updateCloudWarning(warningText) {
        if (!cloudWarningBanner) return;
        if (warningText) {
            if (cloudWarningText) cloudWarningText.innerText = warningText;
            cloudWarningBanner.classList.remove("hidden");
        } else {
            cloudWarningBanner.classList.add("hidden");
        }
    }

    // Non-blocking in-app notification banner (replaces native alert popups)
    let inAppBannerTimeout = null;

    function showInAppNotification(message, icon = "🛰️", type = "info", durationMs = 7000) {
        const banner = document.getElementById("in-app-banner");
        const iconEl = document.getElementById("in-app-banner-icon");
        const textEl = document.getElementById("in-app-banner-text");
        if (!banner) return;

        if (inAppBannerTimeout) {
            clearTimeout(inAppBannerTimeout);
            inAppBannerTimeout = null;
        }

        if (iconEl) iconEl.textContent = icon;
        if (textEl) textEl.textContent = message;

        banner.classList.remove("banner-info", "banner-warning", "banner-success", "hidden");
        if (type === "warning") {
            banner.classList.add("banner-warning");
        } else if (type === "success") {
            banner.classList.add("banner-success");
        } else {
            banner.classList.add("banner-info");
        }

        if (durationMs > 0) {
            inAppBannerTimeout = setTimeout(() => {
                banner.classList.add("hidden");
                inAppBannerTimeout = null;
            }, durationMs);
        }
    }

    function dismissInAppNotification() {
        const banner = document.getElementById("in-app-banner");
        if (banner) banner.classList.add("hidden");
        if (inAppBannerTimeout) {
            clearTimeout(inAppBannerTimeout);
            inAppBannerTimeout = null;
        }
    }

    const btnDismissInAppBanner = document.getElementById("btn-dismiss-in-app-banner");
    if (btnDismissInAppBanner) {
        if (btnDismissInAppBanner) btnDismissInAppBanner.addEventListener("click", dismissInAppNotification);
    }
    if (btnDismissCloudWarning) {
        if (btnDismissCloudWarning) btnDismissCloudWarning.addEventListener("click", () => updateCloudWarning(null));
    }

    function initOSDViewer(elementId, showNav = false) {
        if (!window.OpenSeadragon) {
            console.error("[OSD] OpenSeadragon library not loaded!");
            return null;
        }
        try {
            return OpenSeadragon({
                id: elementId,
                prefixUrl: "https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.1/images/",
                showNavigator: showNav,
                navigatorPosition: "TOP_RIGHT",
                navigatorSizeRatio: 0.22,
                minZoomLevel: 0.2,
                maxZoomLevel: 100,
                minZoomImageRatio: 0.2,
                maxZoomPixelRatio: 100,
                defaultZoomLevel: 0,
                homeFillsViewer: false,
                visibilityRatio: 0.8,
                constrainDuringPan: false,
                constrain: true,
                wrapHorizontal: false,
                wrapVertical: false,
                showNavigationControl: false,
                imageLoaderLimit: 8,
                animationTime: 0.3,
                blendTime: 0.1,
                imageSmoothingEnabled: true,
                maxImageCacheCount: 200,
                gestureSettingsMouse: {
                    scrollToZoom: true,
                    clickToZoom: false,
                    dblClickToZoom: true,
                    dragToPan: true,
                    flickEnabled: true,
                    minScrollDelta: 20,
                    zoomPerScroll: 1.3
                },
                gestureSettingsTouch: {
                    scrollToZoom: true,
                    clickToZoom: false,
                    dblClickToZoom: true,
                    dragToPan: true,
                    pinchToZoom: true,
                    zoomPerScroll: 1.3
                }
            });
        } catch (e) {
            console.error(`[OSD] Failed to init viewer for ${elementId}:`, e);
            return null;
        }
    }

    function fitViewerToImage(viewer) {
        if (!viewer || !viewer.viewport) return;
        try {
            viewer.viewport.resize();
            if (viewer.isOpen()) {
                viewer.viewport.goHome(true);
                viewer.viewport.applyConstraints();
            }
            updateZoomBadge(viewer);
        } catch (e) { }
    }

    function getActiveViewer() {
        if (state.viewMode === "split") {
            return (viewerSR && viewerSR.isOpen()) ? viewerSR : ((viewerLR && viewerLR.isOpen()) ? viewerLR : viewerSR || viewerLR);
        } else {
            return (viewerSideSR && viewerSideSR.isOpen()) ? viewerSideSR : ((viewerSideLR && viewerSideLR.isOpen()) ? viewerSideLR : (viewerSideHR && viewerSideHR.isOpen() ? viewerSideHR : viewerSideSR || viewerSideLR));
        }
    }

    function resetAllViewers() {
        const viewers = [viewerLR, viewerSR, viewerSideLR, viewerSideSR, viewerSideHR, spectralViewerLR, spectralViewerSR, spectralViewerSingle];
        viewers.forEach(v => {
            if (v && v.viewport) {
                try {
                    v.viewport.resize();
                    if (v.isOpen()) {
                        v.viewport.goHome(true);
                        v.viewport.applyConstraints();
                    }
                } catch (e) {}
            }
        });
        if (state.viewMode === "split" && viewerSR && viewerSR.isOpen() && viewerLR && viewerLR.isOpen()) {
            try {
                const center = viewerSR.viewport.getCenter();
                const zoom = viewerSR.viewport.getZoom();
                viewerLR.viewport.panTo(center, true);
                viewerLR.viewport.zoomTo(zoom, null, true);
                viewerLR.viewport.applyConstraints();
            } catch (e) {}
        }
        const primary = getActiveViewer();
        if (primary) updateZoomBadge(primary);
    }

    // Initialize Viewers - showNavigator false to eliminate floating minimap glitch
    viewerLR = initOSDViewer("osd-lr", false);
    viewerSR = initOSDViewer("osd-sr", false);
    viewerSideLR = initOSDViewer("osd-side-lr", false);
    viewerSideSR = initOSDViewer("osd-side-sr", false);
    viewerSideHR = initOSDViewer("osd-side-hr", false);

    // Synchronize Viewports across OpenSeadragon instances with loop protection
    let isSyncing = false;
    function syncViewports(source, targets) {
        if (isSyncing || !source || !source.viewport || !source.isOpen()) return;
        const center = source.viewport.getCenter();
        const zoom = source.viewport.getZoom();
        if (!center || isNaN(center.x) || isNaN(center.y) || isNaN(zoom) || zoom <= 0) return;

        isSyncing = true;
        try {
            targets.forEach(target => {
                if (target && target.viewport && target.isOpen()) {
                    const tCenter = target.viewport.getCenter();
                    const tZoom = target.viewport.getZoom();
                    if (!tCenter || isNaN(tCenter.x)) return;
                    const dist = Math.hypot(tCenter.x - center.x, tCenter.y - center.y);
                    const zoomDiff = Math.abs(tZoom - zoom);
                    if (dist > 0.0001 || zoomDiff > 0.0001) {
                        target.viewport.panTo(center, true);
                        target.viewport.zoomTo(zoom, null, true);
                        target.viewport.applyConstraints();
                    }
                }
            });
        } catch (e) {
            // Ignore temporary view transition errors
        } finally {
            isSyncing = false;
        }
    }

    // Bind Split View Synchronization (LR <-> SR)
    if (viewerLR && viewerSR) {
        viewerLR.addHandler("pan", () => syncViewports(viewerLR, [viewerSR]));
        viewerLR.addHandler("zoom", () => syncViewports(viewerLR, [viewerSR]));
        viewerSR.addHandler("pan", () => syncViewports(viewerSR, [viewerLR]));
        viewerSR.addHandler("zoom", () => syncViewports(viewerSR, [viewerLR]));
    }

    // Dedicated View 2: Spectral & Disaster OpenSeadragon Viewers
    let spectralViewerLR = null;
    let spectralViewerSR = null;
    let spectralViewerSingle = null;

    function ensureSpectralViewers() {
        if (!spectralViewerLR) {
            spectralViewerLR = initOSDViewer("spectral-osd-lr", false);
            if (spectralViewerLR) {
                spectralViewerLR.addHandler("zoom", () => updateSpectralZoomBadge(spectralViewerLR));
            }
        }
        if (!spectralViewerSR) {
            spectralViewerSR = initOSDViewer("spectral-osd-sr", false);
            if (spectralViewerSR) {
                spectralViewerSR.addHandler("zoom", () => updateSpectralZoomBadge(spectralViewerSR));
            }
        }
        if (!spectralViewerSingle) {
            spectralViewerSingle = initOSDViewer("spectral-osd-single", false);
            if (spectralViewerSingle) {
                spectralViewerSingle.addHandler("zoom", () => updateSpectralZoomBadge(spectralViewerSingle));
            }
        }
        if (spectralViewerLR && spectralViewerSR) {
            try {
                spectralViewerLR.addHandler("pan", () => syncViewports(spectralViewerLR, [spectralViewerSR]));
                spectralViewerLR.addHandler("zoom", () => syncViewports(spectralViewerLR, [spectralViewerSR]));
                spectralViewerSR.addHandler("pan", () => syncViewports(spectralViewerSR, [spectralViewerLR]));
                spectralViewerSR.addHandler("zoom", () => syncViewports(spectralViewerSR, [spectralViewerLR]));
            } catch (e) { }
        }
    }

    // Bind Side-by-Side View Synchronization (SideLR <-> SideSR <-> SideHR)
    if (viewerSideLR && viewerSideSR && viewerSideHR) {
        viewerSideLR.addHandler("pan", () => syncViewports(viewerSideLR, [viewerSideSR, viewerSideHR]));
        viewerSideLR.addHandler("zoom", () => syncViewports(viewerSideLR, [viewerSideSR, viewerSideHR]));
        viewerSideSR.addHandler("pan", () => syncViewports(viewerSideSR, [viewerSideLR, viewerSideHR]));
        viewerSideSR.addHandler("zoom", () => syncViewports(viewerSideSR, [viewerSideLR, viewerSideHR]));
        viewerSideHR.addHandler("pan", () => syncViewports(viewerSideHR, [viewerSideLR, viewerSideSR]));
        viewerSideHR.addHandler("zoom", () => syncViewports(viewerSideHR, [viewerSideLR, viewerSideSR]));
    }

    // OpenSeadragon Deep-Zoom UI Controls (Relative 1.0x on fit, up to 100x magnification)
    function updateZoomBadge(viewer) {
        const v = viewer || getActiveViewer();
        if (!zoomLevelBadge || !v || !v.viewport || !v.isOpen()) return;
        try {
            const homeZoom = v.viewport.getHomeZoom() || 1.0;
            const currentZoom = v.viewport.getZoom();
            if (homeZoom <= 0 || currentZoom <= 0 || isNaN(homeZoom) || isNaN(currentZoom)) return;
            const relative = currentZoom / homeZoom;
            zoomLevelBadge.innerText = (relative >= 1.0) ? `${relative.toFixed(1)}x` : `${relative.toFixed(2)}x`;
        } catch (e) { }
    }

    [viewerLR, viewerSR, viewerSideLR, viewerSideSR, viewerSideHR].forEach(v => {
        if (v) {
            v.addHandler("zoom", () => updateZoomBadge(v));
        }
    });

    if (viewerSR) {
        viewerSR.addHandler("update-viewport", () => {
            if (!state.confidenceActive && confOverlay) {
                confOverlay.style.setProperty("display", "none", "important");
                confOverlay.style.visibility = "hidden";
                confOverlay.style.opacity = "0";
            }
            if (!state.infraActive && infraOverlay) {
                infraOverlay.style.setProperty("display", "none", "important");
                infraOverlay.style.visibility = "hidden";
                infraOverlay.style.opacity = "0";
            }
            if (!state.cloudActive && cloudOverlaySR) {
                cloudOverlaySR.style.setProperty("display", "none", "important");
                cloudOverlaySR.style.visibility = "hidden";
                cloudOverlaySR.style.opacity = "0";
            }
        });
    }

    if (viewerLR) {
        viewerLR.addHandler("update-viewport", () => {
            if (!state.cloudActive && cloudOverlayLR) {
                cloudOverlayLR.style.setProperty("display", "none", "important");
                cloudOverlayLR.style.visibility = "hidden";
                cloudOverlayLR.style.opacity = "0";
            }
        });
    }

    if (btnZoomIn) {
        btnZoomIn.addEventListener("click", () => {
            const v = getActiveViewer();
            if (v && v.viewport) {
                v.viewport.zoomBy(1.4);
                v.viewport.applyConstraints();
                updateZoomBadge(v);
                if (state.viewMode === "split" && viewerLR && viewerSR) {
                    syncViewports(v, (v === viewerSR) ? [viewerLR] : [viewerSR]);
                } else if (state.viewMode === "side" && viewerSideLR && viewerSideSR && viewerSideHR) {
                    syncViewports(v, [viewerSideLR, viewerSideSR, viewerSideHR].filter(x => x !== v));
                }
            }
        });
    }

    if (btnZoomOut) {
        btnZoomOut.addEventListener("click", () => {
            const v = getActiveViewer();
            if (v && v.viewport) {
                v.viewport.zoomBy(0.71);
                v.viewport.applyConstraints();
                updateZoomBadge(v);
                if (state.viewMode === "split" && viewerLR && viewerSR) {
                    syncViewports(v, (v === viewerSR) ? [viewerLR] : [viewerSR]);
                } else if (state.viewMode === "side" && viewerSideLR && viewerSideSR && viewerSideHR) {
                    syncViewports(v, [viewerSideLR, viewerSideSR, viewerSideHR].filter(x => x !== v));
                }
            }
        });
    }

    if (btnZoomReset) {
        btnZoomReset.addEventListener("click", () => {
            resetAllViewers();
        });
    }

    function loadViewerImage(viewer, dataUrl, isSR = false) {
        if (!viewer || !dataUrl) return;
        viewer.open({
            type: "image",
            url: dataUrl,
            buildPyramid: true,
            tileSize: 256,
            tileOverlap: 0
        });

        viewer.addOnceHandler("open", () => {
            try {
                fitViewerToImage(viewer);

                // If in split mode and both viewers are loaded, center and align
                if (state.viewMode === "split" && viewerSR && viewerSR.isOpen() && viewerLR && viewerLR.isOpen()) {
                    const center = new OpenSeadragon.Point(0.5, 0.5);
                    viewerSR.viewport.panTo(center, true);
                    viewerLR.viewport.panTo(center, true);
                    const z = viewerSR.viewport.getZoom();
                    if (z > 0) viewerLR.viewport.zoomTo(z, null, true);
                    viewerSR.viewport.applyConstraints();
                    viewerLR.viewport.applyConstraints();
                    updateZoomBadge(viewerSR);
                } else {
                    updateZoomBadge(viewer);
                }

                if (isSR) {
                    viewer.clearOverlays();
                    updateOverlayDisplay();
                } else if (viewer === viewerLR) {
                    viewer.clearOverlays();
                    updateOverlayDisplay();
                }
            } catch (e) {
                console.warn("[OSD] Open handler warning:", e);
            }
        });
    }

    function hasOverlay(viewer, el) {
        if (!viewer || !viewer.currentOverlays || !el) return false;
        return viewer.currentOverlays.some(o => o.element === el || (o.element && o.element.id && o.element.id === el.id));
    }

    function removeOverlaySafe(viewer, el) {
        if (!viewer) return;
        try {
            if (hasOverlay(viewer, el)) {
                viewer.removeOverlay(el);
            }
        } catch (e) { }
    }

    function addOverlaySafe(viewer, el, bounds) {
        if (!viewer || !viewer.isOpen() || !el) return;
        try {
            if (!hasOverlay(viewer, el)) {
                viewer.addOverlay({
                    element: el,
                    location: bounds
                });
            }
        } catch (e) { }
    }

    function syncCloudOverlays() {
        updateOverlayDisplay();
    }

    function updateOverlayDisplay() {
        // 1. USP Confidence Heatmap Overlay
        if (confOverlay) {
            if (state.confidenceActive && state.latestConfUrl) {
                confOverlay.classList.remove("overlay-inactive");
                confOverlay.src = state.latestConfUrl;
                confOverlay.style.setProperty("display", "block", "important");
                confOverlay.style.visibility = "visible";
                confOverlay.style.opacity = state.confidenceOpacity;
                confOverlay.style.pointerEvents = "none";
                if (viewerSR && viewerSR.isOpen()) {
                    try {
                        const imgBounds = (viewerSR.world && viewerSR.world.getItemAt(0)) ? viewerSR.world.getItemAt(0).getBounds() : new OpenSeadragon.Rect(0, 0, 1, 1);
                        addOverlaySafe(viewerSR, confOverlay, imgBounds);
                    } catch (e) { }
                }
            } else {
                removeOverlaySafe(viewerSR, confOverlay);
                confOverlay.classList.add("overlay-inactive");
                confOverlay.style.setProperty("display", "none", "important");
                confOverlay.style.visibility = "hidden";
                confOverlay.style.opacity = "0";
                confOverlay.style.pointerEvents = "none";
            }
        }

        // 2. Infrastructure Detection Overlay (Buildings & Roads)
        if (infraOverlay) {
            if (state.infraActive && state.latestInfraUrl) {
                infraOverlay.classList.remove("overlay-inactive");
                infraOverlay.src = state.latestInfraUrl;
                infraOverlay.style.setProperty("display", "block", "important");
                infraOverlay.style.visibility = "visible";
                infraOverlay.style.opacity = state.infraOpacity;
                infraOverlay.style.pointerEvents = "none";
                if (viewerSR && viewerSR.isOpen()) {
                    try {
                        const imgBounds = (viewerSR.world && viewerSR.world.getItemAt(0)) ? viewerSR.world.getItemAt(0).getBounds() : new OpenSeadragon.Rect(0, 0, 1, 1);
                        addOverlaySafe(viewerSR, infraOverlay, imgBounds);
                    } catch (e) { }
                }
            } else {
                removeOverlaySafe(viewerSR, infraOverlay);
                infraOverlay.classList.add("overlay-inactive");
                infraOverlay.style.setProperty("display", "none", "important");
                infraOverlay.style.visibility = "hidden";
                infraOverlay.style.opacity = "0";
                infraOverlay.style.pointerEvents = "none";
            }
        }

        // 3. Cloud Mask Occlusion Overlays (LR and SR)
        if (state.cloudActive && state.latestCloudUrl) {
            if (cloudOverlaySR) {
                cloudOverlaySR.classList.remove("overlay-inactive");
                cloudOverlaySR.src = state.latestCloudUrl;
                cloudOverlaySR.style.setProperty("display", "block", "important");
                cloudOverlaySR.style.visibility = "visible";
                cloudOverlaySR.style.opacity = state.cloudOpacity;
                cloudOverlaySR.style.pointerEvents = "none";
                if (viewerSR && viewerSR.isOpen()) {
                    try {
                        const srBounds = (viewerSR.world && viewerSR.world.getItemAt(0)) ? viewerSR.world.getItemAt(0).getBounds() : new OpenSeadragon.Rect(0, 0, 1, 1);
                        addOverlaySafe(viewerSR, cloudOverlaySR, srBounds);
                    } catch (e) { }
                }
            }
            if (cloudOverlayLR) {
                cloudOverlayLR.classList.remove("overlay-inactive");
                cloudOverlayLR.src = state.latestCloudUrl;
                cloudOverlayLR.style.setProperty("display", "block", "important");
                cloudOverlayLR.style.visibility = "visible";
                cloudOverlayLR.style.opacity = state.cloudOpacity;
                cloudOverlayLR.style.pointerEvents = "none";
                if (viewerLR && viewerLR.isOpen()) {
                    try {
                        const lrBounds = (viewerLR.world && viewerLR.world.getItemAt(0)) ? viewerLR.world.getItemAt(0).getBounds() : new OpenSeadragon.Rect(0, 0, 1, 1);
                        addOverlaySafe(viewerLR, cloudOverlayLR, lrBounds);
                    } catch (e) { }
                }
            }
        } else {
            if (cloudOverlaySR) {
                removeOverlaySafe(viewerSR, cloudOverlaySR);
                cloudOverlaySR.classList.add("overlay-inactive");
                cloudOverlaySR.style.setProperty("display", "none", "important");
                cloudOverlaySR.style.visibility = "hidden";
                cloudOverlaySR.style.opacity = "0";
                cloudOverlaySR.style.pointerEvents = "none";
            }
            if (cloudOverlayLR) {
                removeOverlaySafe(viewerLR, cloudOverlayLR);
                cloudOverlayLR.classList.add("overlay-inactive");
                cloudOverlayLR.style.setProperty("display", "none", "important");
                cloudOverlayLR.style.visibility = "hidden";
                cloudOverlayLR.style.opacity = "0";
                cloudOverlayLR.style.pointerEvents = "none";
            }
        }
    }

    // =========================================================================
    // 2. Leaflet AOI Map Setup
    // =========================================================================
    const map = L.map("aoi-map", {
        zoomControl: false,
        attributionControl: false
    }).setView([30.575, 75.33], 12);

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 18,
    }).addTo(map);

    let aoiRectangle = L.rectangle([
        [30.55, 75.30],
        [30.60, 75.36]
    ], {
        color: "#00E599",
        weight: 2,
        fillColor: "#00E599",
        fillOpacity: 0.15
    }).addTo(map);

    const drawnItems = new L.FeatureGroup();
    map.addLayer(drawnItems);

    const drawControl = new L.Control.Draw({
        position: 'topright',
        draw: {
            polygon: false,
            polyline: false,
            circle: false,
            circlemarker: false,
            marker: false,
            rectangle: {
                shapeOptions: {
                    color: "#00E599",
                    weight: 2,
                    fillColor: "#00E599",
                    fillOpacity: 0.18
                }
            }
        },
        edit: { featureGroup: drawnItems }
    });
    map.addControl(drawControl);

    let currentFetchSeq = 0;

    function syncAoiUI(bounds, bbox, presetId, lat, lon) {
        if (lat !== undefined && lon !== undefined) {
            let nLat = parseFloat(lat);
            let nLon = parseFloat(lon);
            if (!isNaN(nLat) && !isNaN(nLon)) {
                const _IS_LIKE_INDIA = (nLon > 65 && nLon < 100 && nLat >= 5 && nLat <= 40);
                if (_IS_LIKE_INDIA && Math.abs(nLat) > 60) {
                    showInAppNotification(`[Auto-Correct] Swapped lat/lon detected (${nLat.toFixed(2)}, ${nLon.toFixed(2)}) — correcting order.`, "⚠️", "warning", 5000);
                    [nLat, nLon] = [nLon, nLat];
                }
            }
            if (inputCustomLat) inputCustomLat.value = isNaN(nLat) ? "" : nLat.toFixed(4);
            if (inputCustomLon) inputCustomLon.value = isNaN(nLon) ? "" : nLon.toFixed(4);
            if (aoiCoordsBadge) aoiCoordsBadge.innerText = `Lat: ${isNaN(nLat) ? '--' : nLat.toFixed(3)} | Lon: ${isNaN(nLon) ? '--' : nLon.toFixed(3)}`;

            const spLat = document.getElementById("spectral-input-lat");
            const spLon = document.getElementById("spectral-input-lon");
            const spBadge = document.getElementById("spectral-aoi-coords-badge");
            if (spLat) spLat.value = isNaN(nLat) ? "" : nLat.toFixed(4);
            if (spLon) spLon.value = isNaN(nLon) ? "" : nLon.toFixed(4);
            if (spBadge) spBadge.innerText = `Lat: ${isNaN(nLat) ? '--' : nLat.toFixed(3)} | Lon: ${isNaN(nLon) ? '--' : nLon.toFixed(3)}`;
            lat = nLat;
            lon = nLon;
        }

        if (aoiRectangle && bounds) {
            if (map && !map.hasLayer(aoiRectangle)) {
                aoiRectangle.addTo(map);
            }
            aoiRectangle.setBounds(bounds);
            if (aoiRectangle.bringToFront) aoiRectangle.bringToFront();
        }
        if (spectralAoiRectangle && bounds) {
            if (spectralMap && !spectralMap.hasLayer(spectralAoiRectangle)) {
                spectralAoiRectangle.addTo(spectralMap);
            }
            spectralAoiRectangle.setBounds(bounds);
            if (spectralAoiRectangle.bringToFront) spectralAoiRectangle.bringToFront();
        }

        if (lat !== undefined && lon !== undefined) {
            const numLat = parseFloat(lat);
            const numLon = parseFloat(lon);
            if (inputCustomLat) inputCustomLat.value = numLat.toFixed(4);
            if (inputCustomLon) inputCustomLon.value = numLon.toFixed(4);
            if (aoiCoordsBadge) aoiCoordsBadge.innerText = `Lat: ${numLat.toFixed(3)} | Lon: ${numLon.toFixed(3)}`;

            const spLat = document.getElementById("spectral-input-lat");
            const spLon = document.getElementById("spectral-input-lon");
            const spBadge = document.getElementById("spectral-aoi-coords-badge");
            if (spLat) spLat.value = numLat.toFixed(4);
            if (spLon) spLon.value = numLon.toFixed(4);
            if (spBadge) spBadge.innerText = `Lat: ${numLat.toFixed(3)} | Lon: ${numLon.toFixed(3)}`;
        }

        document.querySelectorAll(".aoi-card").forEach(c => {
            c.classList.toggle("active", c.dataset.id === presetId);
        });

        const spMetaAoi = document.getElementById("spectral-meta-aoi");
        if (spMetaAoi) {
            const preset = state.presets.find(p => p.id === presetId);
            spMetaAoi.innerText = preset ? preset.title : (presetId === "custom_drawn_aoi" ? "Custom Drawn AOI" : "Custom Coordinates");
        }
    }

    map.on(L.Draw.Event.CREATED, function (e) {
        const layer = e.layer;
        drawnItems.clearLayers();
        drawnItems.addLayer(layer);

        const bounds = layer.getBounds();
        const bbox = [
            parseFloat(bounds.getWest().toFixed(4)),
            parseFloat(bounds.getSouth().toFixed(4)),
            parseFloat(bounds.getEast().toFixed(4)),
            parseFloat(bounds.getNorth().toFixed(4))
        ];

        state.activeBbox = bbox;
        state.activePreset = "custom_drawn_aoi";
        const center = bounds.getCenter();
        syncAoiUI(bounds, bbox, "custom_drawn_aoi", center.lat, center.lng);
        fetchTileForBbox(bbox, "custom_drawn_aoi");
    });

    map.on(L.Draw.Event.EDITED, function (e) {
        const layers = e.layers;
        layers.eachLayer(function (layer) {
            const bounds = layer.getBounds();
            const bbox = [
                parseFloat(bounds.getWest().toFixed(4)),
                parseFloat(bounds.getSouth().toFixed(4)),
                parseFloat(bounds.getEast().toFixed(4)),
                parseFloat(bounds.getNorth().toFixed(4))
            ];
            state.activeBbox = bbox;
            state.activePreset = "custom_drawn_aoi";
            const center = bounds.getCenter();
            syncAoiUI(bounds, bbox, "custom_drawn_aoi", center.lat, center.lng);
            fetchTileForBbox(bbox, "custom_drawn_aoi");
        });
    });

    map.on(L.Draw.Event.DELETED, function () {
        if (state.presets && state.presets.length > 0) {
            selectPreset(state.presets[0].id);
        }
    });

    const btnDrawRect = document.getElementById("btn-draw-rect");
    if (btnDrawRect) {
        btnDrawRect.addEventListener("click", () => {
            const openBtn = document.getElementById('open-map-modal');
            if (openBtn) openBtn.click();
            setTimeout(() => {
                new L.Draw.Rectangle(map, drawControl.options.draw.rectangle).enable();
            }, 150);
        });
    }

    map.on("click", (e) => {
        if (document.querySelector(".leaflet-draw-actions")) return;
        const lat = e.latlng.lat;
        const lon = e.latlng.lng;
        const span = 0.03;
        const customBbox = [
            parseFloat((lon - span).toFixed(4)),
            parseFloat((lat - span).toFixed(4)),
            parseFloat((lon + span).toFixed(4)),
            parseFloat((lat + span).toFixed(4))
        ];

        state.activeBbox = customBbox;
        state.activePreset = "custom_click";
        const bounds = [
            [customBbox[1], customBbox[0]],
            [customBbox[3], customBbox[2]]
        ];

        if (drawnItems) drawnItems.clearLayers();
        if (spectralDrawnItems) spectralDrawnItems.clearLayers();
        if (map) map.panTo([lat, lon]);
        if (spectralMap) spectralMap.panTo([lat, lon]);

        syncAoiUI(bounds, customBbox, "custom_click", lat, lon);
        fetchTileForBbox(customBbox, "custom_click");
    });

    // Manual Coordinate Navigation
    function navigateToCoordinates(lat, lon) {
        let nLat = parseFloat(lat);
        let nLon = parseFloat(lon);
        if (!isNaN(nLat) && !isNaN(nLon)) {
            if (Math.abs(nLat) > 90 || Math.abs(nLon) > 180) {
                showInAppNotification("Invalid coordinates — numbers out of valid range.", "⚠️", "warning", 6000);
                return;
            }
            if (Math.abs(nLat) > 60 && Math.abs(nLon) <= 60) {
                showInAppNotification(`[Auto-Correct] Suspected lat/lon swap: (${nLat.toFixed(2)}, ${nLon.toFixed(2)}). Using (${nLon.toFixed(2)}, ${nLat.toFixed(2)}) instead.`, "⚠️", "warning", 6000);
                [nLat, nLon] = [nLon, nLat];
            }
        }
        if (isNaN(nLat) || nLat < -90 || nLat > 90) {
            showInAppNotification("Please enter a valid Latitude between -90 and 90.", "⚠️", "warning", 6000);
            if (inputCustomLat) inputCustomLat.focus();
            return;
        }
        if (isNaN(nLon) || nLon < -180 || nLon > 180) {
            showInAppNotification("Please enter a valid Longitude between -180 and 180.", "⚠️", "warning", 6000);
            if (inputCustomLon) inputCustomLon.focus();
            return;
        }

        const span = 0.03;
        const customBbox = [
            parseFloat((nLon - span).toFixed(4)),
            parseFloat((nLat - span).toFixed(4)),
            parseFloat((nLon + span).toFixed(4)),
            parseFloat((nLat + span).toFixed(4))
        ];

        state.activeBbox = customBbox;
        state.activePreset = "custom_coords";

        const bounds = [
            [customBbox[1], customBbox[0]],
            [customBbox[3], customBbox[2]]
        ];

        if (drawnItems) drawnItems.clearLayers();
        if (spectralDrawnItems) spectralDrawnItems.clearLayers();
        if (map) {
            map.invalidateSize();
            map.flyToBounds(bounds, { padding: [20, 20], duration: 0.8 });
        }
        if (spectralMap) {
            spectralMap.invalidateSize();
            spectralMap.flyToBounds(bounds, { padding: [20, 20], duration: 0.8 });
        }

        syncAoiUI(bounds, customBbox, "custom_coords", nLat, nLon);
        fetchTileForBbox(customBbox, "custom_coords");
        if (state.activeTopView === "spectral") {
            selectSpectralMode(state.activeSpectralMode || "ndvi");
        }
    }

    if (btnGotoCoords) {
        if (btnGotoCoords) btnGotoCoords.addEventListener("click", () => {
            const lat = parseFloat(inputCustomLat ? inputCustomLat.value.trim() : NaN);
            const lon = parseFloat(inputCustomLon ? inputCustomLon.value.trim() : NaN);
            navigateToCoordinates(lat, lon);
        });
    }

    const handleCoordEnter = (e) => {
        if (e.key === "Enter") {
            const lat = parseFloat(inputCustomLat ? inputCustomLat.value.trim() : NaN);
            const lon = parseFloat(inputCustomLon ? inputCustomLon.value.trim() : NaN);
            navigateToCoordinates(lat, lon);
        }
    };
    if (inputCustomLat) inputCustomLat.addEventListener("keydown", handleCoordEnter);
    if (inputCustomLon) inputCustomLon.addEventListener("keydown", handleCoordEnter);

    // =========================================================================
    // 3. API Communication & Pipelines
    // =========================================================================
    async function loadPresets() {
        try {
            const res = await fetch("/api/presets");
            const data = await res.json();
            state.presets = data.presets;
            renderPresetChips();
            selectPreset("punjab_agri");
        } catch (err) {
            console.error("[GEO-SRM] Failed to load presets:", err);
        }
    }

    function renderPresetChips() {
        // presetsList.innerHTML = "";
        const spectralPresetsList = document.getElementById("spectral-presets-list");
        if (spectralPresetsList) spectralPresetsList.innerHTML = "";

        state.presets.forEach(p => {
            const chip = document.createElement("div");
            chip.className = `aoi-card ${p.id === state.activePreset ? 'active' : ''}`;
            chip.dataset.id = p.id;
            chip.innerHTML = `
                <h4>${p.title}</h4>
                <p>${p.description}</p>
            `;
            chip.addEventListener("click", () => selectPreset(p.id));
            presetsList.appendChild(chip);

            if (spectralPresetsList) {
                const spChip = chip.cloneNode(true);
                if (spChip) spChip.addEventListener("click", () => selectPreset(p.id));
                spectralPresetsList.appendChild(spChip);
            }
        });
    }

    function selectPreset(presetId) {
        const preset = state.presets.find(p => p.id === presetId);
        if (!preset) return;

        state.activePreset = presetId;
        state.activeBbox = preset.bbox;

        if (drawnItems) drawnItems.clearLayers();
        if (spectralDrawnItems) spectralDrawnItems.clearLayers();

        const bounds = [
            [preset.bbox[1], preset.bbox[0]],
            [preset.bbox[3], preset.bbox[2]]
        ];
        if (map && map.getSize().x > 0) {
            map.invalidateSize();
            map.flyToBounds(bounds, { padding: [20, 20], duration: 0.8 });
        }
        if (spectralMap && spectralMap.getSize().x > 0) {
            spectralMap.invalidateSize();
            spectralMap.flyToBounds(bounds, { padding: [20, 20], duration: 0.8 });
        }

        const lat = preset.coords ? preset.coords[1] : (preset.bbox[1] + preset.bbox[3]) / 2;
        const lon = preset.coords ? preset.coords[0] : (preset.bbox[0] + preset.bbox[2]) / 2;
        syncAoiUI(bounds, preset.bbox, presetId, lat, lon);

        fetchTileForBbox(preset.bbox, preset.id);
        if (state.activeTopView === "spectral") {
            selectSpectralMode(state.activeSpectralMode || "ndvi");
        }
    }

    async function fetchTileForBbox(bbox, aoiId) {
        const fetchSeq = ++currentFetchSeq;
        loadingScrim.classList.remove("hidden");
        try {
            const res = await fetch("/api/fetch-tile", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    bbox: bbox,
                    aoi_id: aoiId,
                    max_cloud: 20
                })
            });
            const data = await res.json();
            if (fetchSeq !== currentFetchSeq) return;

            if (data.status === "success") {
                state.hasReference = !!data.has_reference;
                state.isScientific = !!data.is_scientific_ground_truth;
                state.referenceTier = data.reference_tier;
                state.referenceTierLabel = data.reference_tier_label;
                state.latestLrUrl = data.lr_preview;
                state.latestHrUrl = data.hr_preview;
                state.latestCloudUrl = data.cloud_mask_preview || null;
                state.cloudCoveragePct = data.cloud_coverage_pct || 0.0;
                updateCloudWarning(data.cloud_warning);

                // Load OpenSeadragon Viewers
                loadViewerImage(viewerLR, data.lr_preview);
                loadViewerImage(viewerSideLR, data.lr_preview);

                // Extract & cache live Copernicus CDSE telemetry
                if (data.metadata && data.metadata.acquisition_date) {
                    state.acquisitionDate = data.metadata.acquisition_date;
                    state.sceneName = data.metadata.scene_name || "";
                }
                const acqText = state.acquisitionDate || "Recent 2026 Orbit";

                // Update 1st Side-by-Side Panel Header & Split-Slider Tag with dynamic live telemetry
                const tagLr = document.getElementById("tag-lr");
                if (tagLr) {
                    tagLr.innerHTML = `Input: Sentinel-2 L2A (10m) <span style="color: #00E599; font-size: 0.72rem; margin-left: 6px; font-weight: 600; text-transform: uppercase;">· Live Pass: ${acqText}</span>`;
                }

                const headerSideLr = document.getElementById("header-side-lr");
                if (headerSideLr) {
                    headerSideLr.innerHTML = `Input: Sentinel-2 L2A (10m) <span style="color: #00E599; font-size: 0.72rem; margin-left: 4px;">(${acqText})</span>`;
                }

                const spectralTagLr = document.getElementById("spectral-tag-lr");
                if (spectralTagLr) {
                    spectralTagLr.innerHTML = `Input: Sentinel-2 (10m) <span style="color: #00E599; font-size: 0.72rem; margin-left: 4px;">· ${acqText}</span>`;
                }

                // Update 3rd Side-by-Side Panel Header strictly per tier rules
                const headerSideHr = document.getElementById("header-side-hr");
                if (headerSideHr) {
                    if (data.is_scientific_ground_truth) {
                        headerSideHr.innerText = "Reference: Ground Truth HR (1.5m)";
                    } else if (data.has_reference) {
                        headerSideHr.innerText = "Reference: Visual Basemap";
                    } else {
                        headerSideHr.innerText = "Reference: None Available";
                    }
                }

                if (data.has_reference && data.hr_preview) {
                    loadViewerImage(viewerSideHR, data.hr_preview);
                } else {
                    if (viewerSideHR) viewerSideHR.close();
                }

                if (data.is_scientific_ground_truth) {
                    if (panelValidationPaired) panelValidationPaired.style.display = "block";
                    if (panelValidationNr) panelValidationNr.style.display = "none";
                } else {
                    if (panelValidationPaired) panelValidationPaired.style.display = "none";
                    if (panelValidationNr) panelValidationNr.style.display = "block";
                }

                if (verifReference) {
                    if (data.reference_file) {
                        verifReference.innerText = data.reference_file;
                        verifReference.style.color = data.is_scientific_ground_truth ? "#38bdf8" : "#facc15";
                    } else {
                        verifReference.innerText = "None (Unpaired AOI — No False Fallback)";
                        verifReference.style.color = "#f87171";
                    }
                }

                if (data.visualization_info) {
                    updateVisualizationLegend(data.visualization_info);
                }

                // Trigger PyTorch Super-Resolution
                await runSuperResolution(fetchSeq);
            } else {
                showInAppNotification(`Fetch error: ${data.detail || 'Could not load tile'}`, "❌", "warning", 7000);
            }
        } catch (err) {
            if (fetchSeq === currentFetchSeq) {
                console.error("[GEO-SRM] Fetch tile failed:", err);
            }
        } finally {
            if (fetchSeq === currentFetchSeq) {
                loadingScrim.classList.add("hidden");
            }
        }
    }

    // Real Super Resolution Inference: Calls POST /api/superresolve
    async function runSuperResolution(fetchSeq = null) {
        if (fetchSeq !== null && fetchSeq !== currentFetchSeq) return;
        loadingScrim.classList.remove("hidden");
        const elCoregStatusInit = document.getElementById("coreg-shift-status");
        const elComputedAtInit = document.getElementById("metric-computed-at");
        if (elCoregStatusInit) elCoregStatusInit.innerHTML = '<span style="color: #38bdf8;">🔄 AROSICS: Calculating spatial shifts...</span>';
        if (elComputedAtInit) elComputedAtInit.innerText = 'Executing inference...';

        try {
            const res = await fetch("/api/superresolve", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    num_mc_samples: state.mcSamples,
                    scale_factor: state.scaleFactor,
                    model_name: state.modelName,
                    apply_unsharp: state.applyUnsharp,
                    apply_realesrgan_sharpen: state.applyRealESRGANSharpen,
                    visualization_mode: state.visualizationMode || "true_color"
                })
            });
            const data = await res.json();
            if (fetchSeq !== null && fetchSeq !== currentFetchSeq) return;
            if (data.status === "success") {
                state.latestSrUrl = data.sr_preview;
                if (data.lr_preview) {
                    state.latestLrUrl = data.lr_preview;
                    loadViewerImage(viewerLR, data.lr_preview, false);
                    loadViewerImage(viewerSideLR, data.lr_preview, false);
                }
                state.latestConfUrl = data.confidence_heatmap;
                state.latestInfraUrl = data.infrastructure_overlay;
                state.infraSummary = data.infrastructure_summary;
                if (data.cloud_mask_preview) state.latestCloudUrl = data.cloud_mask_preview;
                if (data.cloud_coverage_pct !== undefined) state.cloudCoveragePct = data.cloud_coverage_pct;
                if (data.cloud_warning) updateCloudWarning(data.cloud_warning);

                if (data.visualization_info) {
                    updateVisualizationLegend(data.visualization_info);
                }

                // Load OpenSeadragon SR Viewers with Overlays
                loadViewerImage(viewerSR, data.sr_preview, true);
                loadViewerImage(viewerSideSR, data.sr_preview, false);
                syncCloudOverlays();

                // Update 1st Side-by-Side Panel Header & Split-Slider Tag for Input Sentinel-2
                if (data.satellite_metadata && data.satellite_metadata.acquisition_date) {
                    state.acquisitionDate = data.satellite_metadata.acquisition_date;
                }
                const acqText = state.acquisitionDate || "Recent 2026 Orbit";
                const tagLr = document.getElementById("tag-lr");
                if (tagLr) {
                    tagLr.innerHTML = `Input: Sentinel-2 L2A (10m) <span style="color: #00E599; font-size: 0.72rem; margin-left: 6px; font-weight: 600; text-transform: uppercase;">· Live Pass: ${acqText}</span>`;
                }
                const headerSideLr = document.getElementById("header-side-lr");
                if (headerSideLr) {
                    headerSideLr.innerHTML = `Input: Sentinel-2 L2A (10m) <span style="color: #00E599; font-size: 0.72rem; margin-left: 4px;">(${acqText})</span>`;
                }
                const spectralTagLr = document.getElementById("spectral-tag-lr");
                if (spectralTagLr) {
                    spectralTagLr.innerHTML = `Input: Sentinel-2 (10m) <span style="color: #00E599; font-size: 0.72rem; margin-left: 4px;">· ${acqText}</span>`;
                }

                // Update 2nd Side-by-Side Panel Header & Slider Tag for AI Output
                const headerSideSr = document.getElementById("header-side-sr");
                const modelNames = {
                    hat: "HAT",
                    evoland: "HAT",
                    ldsr_s2: "LDSR-S2",
                    srmnet: "SRM-Net",
                    carn: "CARN",
                    evoland_carn: "CARN"
                };
                const activeModel = modelNames[state.modelName] || "HAT";
                const targetRes = (10.0 / state.scaleFactor).toFixed(1);
                if (headerSideSr) {
                    headerSideSr.innerText = `Output: ${activeModel} (${targetRes}m)`;
                }
                const tagSr = document.getElementById("tag-sr");
                if (tagSr) {
                    tagSr.innerText = `Output: ${activeModel} (${targetRes}m)`;
                }

                const spMetaModel = document.getElementById("spectral-meta-model");
                if (spMetaModel) {
                    spMetaModel.innerText = `${activeModel} (${targetRes}m GSD)`;
                }
                if (state.activeTopView === "spectral") {
                    selectSpectralMode(state.activeSpectralMode || "ndvi");
                }

                // Update 3rd Side-by-Side Panel Header strictly per tier rules
                const headerSideHr = document.getElementById("header-side-hr");
                if (headerSideHr) {
                    if (data.is_scientific_ground_truth) {
                        headerSideHr.innerText = "Reference: Ground Truth HR (1.5m)";
                    } else if (data.has_reference) {
                        headerSideHr.innerText = "Reference: Visual Basemap";
                    } else {
                        headerSideHr.innerText = "Reference: None Available";
                    }
                }

                // Update Array Dimensions in Inspector Badge
                if (inspectorDims && data.lr_dimensions && data.sr_dimensions) {
                    inspectorDims.innerText = `LR: ${data.lr_dimensions} → SR: ${data.sr_dimensions} (${data.pixel_expansion_ratio})`;
                }

                // Update Scientific USP Metrics (reference-free via opensr-test)
                metricConfidence.innerText = `${data.usp_metrics.avg_confidence}%`;
                metricUncertainty.innerText = data.usp_metrics.mean_uncertainty;
                metricSam.innerText = `${data.usp_metrics.sam_degrees}°`;
                metricCycle.innerText = data.usp_metrics.cycle_consistency_mae;

                // Quality Assessment Mode Swapping (Paired vs No-Reference)
                const isPaired = (data.assessment_mode === "paired" || (data.is_scientific_ground_truth && data.validation_metrics && data.validation_metrics.has_reference));

                if (isPaired) {
                    // Activate Paired Ground Truth Panel (Punjab, Delhi, Varanasi)
                    if (panelValidationPaired) panelValidationPaired.style.display = "block";
                    if (panelValidationNr) panelValidationNr.style.display = "none";

                    if (data.validation_metrics) {
                        if (metricPsnr) metricPsnr.innerText = `${data.validation_metrics.psnr_db} dB`;
                        if (metricSsim) metricSsim.innerText = data.validation_metrics.ssim;
                        if (metricErgas) metricErgas.innerText = data.validation_metrics.ergas;
                        if (metricRuntime) metricRuntime.innerText = `${data.inference_time_ms} ms (${data.device})`;

                        const elCoregStatus = document.getElementById("coreg-shift-status");
                        const elComputedAt = document.getElementById("metric-computed-at");
                        if (elComputedAt && data.validation_metrics.computed_at) {
                            elComputedAt.innerText = `Fresh: ${data.validation_metrics.computed_at} [ID: ${data.validation_metrics.execution_id || 'live'}]`;
                        }
                        if (elCoregStatus && data.validation_metrics.coregistration) {
                            const c = data.validation_metrics.coregistration;
                            if (c.success) {
                                elCoregStatus.innerHTML = `<span style="color: #4ade80;">✓ Co-Reg Shift:</span> X: <strong>${c.x_shift_px > 0 ? '+' : ''}${c.x_shift_px}px</strong> (${c.x_shift_m > 0 ? '+' : ''}${c.x_shift_m}m), Y: <strong>${c.y_shift_px > 0 ? '+' : ''}${c.y_shift_px}px</strong> (${c.y_shift_m > 0 ? '+' : ''}${c.y_shift_m}m)`;
                            } else {
                                elCoregStatus.innerHTML = `<span style="color: #f59e0b;">⚠️ Coreg:</span> ${c.method || 'Fallback'}`;
                            }
                        }
                    }
                } else {
                    // Activate No-Reference Quality Assessment Panel (pyiqa NIQE & BRISQUE + opensr-test)
                    if (panelValidationPaired) panelValidationPaired.style.display = "none";
                    if (panelValidationNr) panelValidationNr.style.display = "block";

                    const validationNoRefBanner = document.getElementById("validation-no-ref-banner");
                    if (validationNoRefBanner) {
                        if (data.reference_tier === 3 || (!data.is_scientific_ground_truth && data.has_reference)) {
                            validationNoRefBanner.innerHTML = `
                                <div style="font-weight: 600; color: #c084fc; margin-bottom: 4px; display: flex; align-items: center; gap: 6px;">
                                    <span style="font-size: 0.95rem;">🌐</span> Tier-3 Global Reference Basemap
                                </div>
                                Loaded Esri WorldImagery via <code>leafmap</code> for visual reference. Because this is an uncalibrated basemap with variable capture resolution and sun angles, full-reference PSNR/SSIM metrics are omitted. Evaluating blind No-Reference Image Quality Assessment (pyiqa NIQE &amp; BRISQUE) + opensr-test trust metrics.
                            `;
                        } else {
                            validationNoRefBanner.innerHTML = `
                                <div style="font-weight: 600; color: #c084fc; margin-bottom: 4px; display: flex; align-items: center; gap: 6px;">
                                    <span style="font-size: 0.95rem;">⚠️</span> Unpaired Scene Notification
                                </div>
                                No ground truth reference available for this AOI — validation metrics require a paired high-resolution reference tile. Activating Blind/No-Reference Image Quality Assessment (pyiqa NIQE &amp; BRISQUE) + opensr-test trust metrics.
                            `;
                        }
                    }

                    if (data.no_reference_metrics) {
                        if (metricNrNiqe) metricNrNiqe.innerText = data.no_reference_metrics.niqe;
                        if (metricNrBrisque) metricNrBrisque.innerText = data.no_reference_metrics.brisque;
                    }
                    if (metricNrConfidence) metricNrConfidence.innerText = `${data.usp_metrics.avg_confidence}%`;
                    if (metricNrCycle) metricNrCycle.innerText = data.usp_metrics.cycle_consistency_mae;
                }

                // Update Infrastructure Analytics HUD (Section 12 - Live SamGeo Building & Road Detection)
                if (data.infrastructure_summary) {
                    const isum = data.infrastructure_summary;
                    if (metricBuildings) metricBuildings.innerText = isum.buildings_count ?? "--";
                    if (metricRoads) metricRoads.innerText = (isum.roads_length_km !== undefined) ? `${isum.roads_length_km} km` : "-- km";
                    if (metricInfraConf) metricInfraConf.innerText = `${isum.avg_detection_confidence}%`;
                }

                // Update Live Pipeline Verification HUD
                if (data.model_info && verifCheckpoint) {
                    verifCheckpoint.innerHTML = `<strong>${data.model_info.checkpoint_file}</strong> [${data.model_info.model_architecture || data.model_info.model_name}] (${data.model_info.trained_scale}x / ${data.model_info.target_gsd})`;
                }
                if (data.model_info && verifDropout) {
                    verifDropout.innerText = data.model_info.dropout_mode;
                }
                if (data.mc_variance_stats && verifVariance) {
                    const st = data.mc_variance_stats;
                    verifVariance.innerText = `min: ${st.raw_variance_min.toExponential(2)} | max: ${st.raw_variance_max.toExponential(2)} | mean: ${st.raw_variance_mean.toExponential(2)}`;
                }
                if (verifDimensions && data.lr_dimensions && data.sr_dimensions) {
                    verifDimensions.innerText = `LR: ${data.lr_dimensions} → SR: ${data.sr_dimensions} (${data.pixel_expansion_ratio})`;
                }
                if (verifUnsharp) {
                    if (data.post_processing_stage) {
                        verifUnsharp.innerText = data.post_processing_stage;
                    } else if (data.realesrgan_sharpened) {
                        verifUnsharp.innerText = "Real-ESRGAN x4plus (outscale=1, tile=256)";
                    } else if (data.unsharp_mask_applied) {
                        verifUnsharp.innerText = "skimage.filters.unsharp_mask (radius=1.0, amount=1.2)";
                    } else {
                        verifUnsharp.innerText = "Disabled (Raw Model Output)";
                    }
                }
                if (verifReference) {
                    if (data.reference_provenance) {
                        verifReference.innerText = data.reference_provenance;
                        verifReference.style.color = data.is_scientific_ground_truth ? "#38bdf8" : "#facc15";
                    } else {
                        verifReference.innerText = "None (Unpaired AOI — No False Fallback)";
                        verifReference.style.color = "#f87171";
                    }
                }



                if (verifTemporal) {
                    if (state.acquisitionMode === "multi_temporal") {
                        verifTemporal.innerText = "3-Pass Temporal Median (Active)";
                        verifTemporal.className = "verif-code-emerald";
                    } else {
                        verifTemporal.innerText = "Single-Pass (Default)";
                        verifTemporal.className = "verif-code-cyan";
                    }
                }

                // Update NETRA Blockchain Provenance Telemetry
                if (data.blockchain_provenance) {
                    state.blockchainRecord = data.blockchain_provenance;
                    updateBlockchainTelemetry(data.blockchain_provenance);
                }
            }
        } catch (err) {
            console.error("[GEO-SRM] Super-resolution request failed:", err);
        } finally {
            if (fetchSeq === null || fetchSeq === currentFetchSeq) {
                loadingScrim.classList.add("hidden");
            }
        }
    }

    // =========================================================================
    // 4. Split Slider Interactive Swipe with OpenSeadragon Viewports
    // =========================================================================
    let isDragging = false;

    function updateSliderPosition(clientX) {
        const rect = sliderContainer.getBoundingClientRect();
        let offsetX = clientX - rect.left;
        let percentage = (offsetX / rect.width) * 100;
        percentage = Math.max(0, Math.min(100, percentage));

        sliderHandle.style.left = `${percentage}%`;
        srWrapper.style.clipPath = `polygon(${percentage}% 0, 100% 0, 100% 100%, ${percentage}% 100%)`;
    }

    if (sliderHandle) sliderHandle.addEventListener("mousedown", (e) => {
        isDragging = true;
        e.preventDefault();
    });

    window.addEventListener("mouseup", () => {
        isDragging = false;
    });

    window.addEventListener("mousemove", (e) => {
        if (!isDragging) return;
        updateSliderPosition(e.clientX);
    });

    if (sliderHandle) sliderHandle.addEventListener("touchstart", () => {
        isDragging = true;
    });

    window.addEventListener("touchend", () => {
        isDragging = false;
    });

    window.addEventListener("touchmove", (e) => {
        if (!isDragging || !e.touches.length) return;
        updateSliderPosition(e.touches[0].clientX);
    });



    // =========================================================================
    // 5. Parameter Controls & Toolbar Event Listeners
    // =========================================================================
    if (sliderMcSamples) sliderMcSamples.addEventListener("input", (e) => {
        state.mcSamples = parseInt(e.target.value);
        valMcSamples.innerText = `${state.mcSamples} passes`;
    });

    if (selectScale) {
        if (selectScale) selectScale.addEventListener("change", (e) => {
            state.scaleFactor = parseInt(e.target.value);
            console.log(`[GEO-SRM] Target scale changed to ${state.scaleFactor}x (${10.0 / state.scaleFactor}m GSD)`);
            runSuperResolution();
        });
    }

    if (selectModel) {
        const ldsrNotice = document.getElementById("ldsr-diffusion-notice");
        if (selectModel) selectModel.addEventListener("change", (e) => {
            state.modelName = e.target.value;
            console.log(`[GEO-SRM] Model changed to: ${state.modelName}`);
            if (state.modelName === "ldsr_s2") {
                if (ldsrNotice) ldsrNotice.classList.remove("hidden");
                showInAppNotification(
                    "LDSR-S2 Latent Diffusion Active: Generates sharper high-frequency detail. Monitor the Confidence HUD to assess hallucination risk.",
                    "⚡",
                    "warning",
                    7000
                );
            } else {
                if (ldsrNotice) ldsrNotice.classList.add("hidden");
            }
            runSuperResolution();
        });
    }

    if (selectVisMode) {
        if (selectVisMode) selectVisMode.addEventListener("change", async (e) => {
            const mode = e.target.value;
            state.visualizationMode = mode;
            console.log(`[GEO-SRM] Multi-Band Spectral Mode changed to: ${mode}`);
            await switchVisualizationMode(mode);
        });
    }

    async function switchVisualizationMode(mode) {
        loadingScrim.classList.remove("hidden");
        try {
            const res = await fetch("/api/visualize", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    mode: mode,
                    apply_realesrgan_sharpen: state.applyRealESRGANSharpen
                })
            });
            const data = await res.json();
            if (data.status === "success") {
                if (data.lr_preview) {
                    state.latestLrUrl = data.lr_preview;
                    loadViewerImage(viewerLR, data.lr_preview, false);
                    loadViewerImage(viewerSideLR, data.lr_preview, false);
                }
                if (data.sr_preview) {
                    state.latestSrUrl = data.sr_preview;
                    loadViewerImage(viewerSR, data.sr_preview, true);
                    loadViewerImage(viewerSideSR, data.sr_preview, false);
                    syncCloudOverlays();
                }
                updateVisualizationLegend(data);
            } else if (data.status === "unavailable") {
                console.warn("[GEO-SRM] Visualization unavailable:", data.message);
                const titleEl = document.getElementById("spectral-legend-title");
                const statsEl = document.getElementById("spectral-legend-stats");
                const barContainer = document.getElementById("spectral-legend-bar-container");
                if (titleEl) titleEl.innerText = `⚠️ ${data.title || mode.toUpperCase()} (unavailable)`;
                if (statsEl) statsEl.innerText = data.message || "Required spectral band missing — unavailable.";
                if (barContainer) barContainer.style.display = "none";
            }
        } catch (err) {
            console.error("[GEO-SRM] Failed to switch spectral visualization mode:", err);
        } finally {
            loadingScrim.classList.add("hidden");
        }
    }

    function updateVisualizationLegend(info) {
        if (!info) return;
        const titleEl = document.getElementById("spectral-legend-title");
        const statsEl = document.getElementById("spectral-legend-stats");
        const barContainer = document.getElementById("spectral-legend-bar-container");
        const barEl = document.getElementById("spectral-legend-bar");
        const tickMin = document.getElementById("spectral-tick-min");
        const tickMid = document.getElementById("spectral-tick-mid");
        const tickMax = document.getElementById("spectral-tick-max");
        const chipsContainer = document.getElementById("spectral-chips-container");

        const modeIcons = {
            true_color: "🌿",
            false_color_ir: "🔴",
            ndvi: "🌱",
            ndwi: "💧",
            ndbi: "🏢",
            nbr: "🔥"
        };
        const icon = modeIcons[info.mode] || "🛰️";
        if (titleEl) {
            titleEl.innerText = `${icon} ${info.title || info.name || info.mode.toUpperCase()}`;
        }

        if (info.mode === "true_color" || (!info.is_index && info.mode !== "false_color_ir")) {
            if (statsEl) statsEl.innerText = "Surface Reflectance (B04-B03-B02)";
            if (barContainer) barContainer.style.display = "none";
            if (chipsContainer) chipsContainer.style.display = "none";
        } else if (info.mode === "false_color_ir") {
            if (statsEl) statsEl.innerText = "NIR-Red-Green Composite";
            if (barContainer) barContainer.style.display = "none";
            if (chipsContainer) {
                chipsContainer.style.display = "flex";
                chipsContainer.innerHTML = `
                    <span class="spectral-chip"><span class="spectral-chip-dot" style="background:#e31a1c;"></span> Veg / Canopy</span>
                    <span class="spectral-chip"><span class="spectral-chip-dot" style="background:#73b2d8;"></span> Urban / Roads</span>
                    <span class="spectral-chip"><span class="spectral-chip-dot" style="background:#081d58;"></span> Water</span>
                    <span class="spectral-chip"><span class="spectral-chip-dot" style="background:#d2b48c;"></span> Bare Soil</span>
                `;
            }
        } else if (info.is_index && info.legend && info.legend.type === "continuous") {
            if (chipsContainer) chipsContainer.style.display = "none";
            if (barContainer) barContainer.style.display = "flex";
            if (barEl) barEl.style.background = info.legend.gradient_css || "linear-gradient(to right, #000, #fff)";
            if (tickMin) tickMin.innerText = info.legend.min_label || `${info.legend.min}`;
            if (tickMid) tickMid.innerText = info.legend.mid_label || "0.0";
            if (tickMax) tickMax.innerText = info.legend.max_label || `${info.legend.max}`;

            if (statsEl && info.stats) {
                statsEl.innerText = `μ=${info.stats.mean} [${info.stats.min}, ${info.stats.max}]`;
            } else if (statsEl) {
                statsEl.innerText = info.subtitle || "Spectral Index";
            }
        }
    }

    if (checkRealESRGANSharpen) {
        if (checkRealESRGANSharpen) checkRealESRGANSharpen.addEventListener("change", (e) => {
            state.applyRealESRGANSharpen = e.target.checked;
            console.log(`[GEO-SRM] Stage 2 Real-ESRGAN sharpening toggled: ${state.applyRealESRGANSharpen}`);
            runSuperResolution();
        });
    }



    if (checkUnsharp) {
        if (checkUnsharp) checkUnsharp.addEventListener("change", (e) => {
            state.applyUnsharp = e.target.checked;
            console.log(`[GEO-SRM] Edge enhancement toggled: ${state.applyUnsharp}`);
            runSuperResolution();
        });
    }

    if (btnModeSingle) {
        if (btnModeSingle) btnModeSingle.addEventListener("click", async () => {
            btnModeSingle.classList.add("active");
            if (btnModeMulti) btnModeMulti.classList.remove("active");
            state.acquisitionMode = "single";
            if (verifTemporal) {
                verifTemporal.innerText = "Single-Pass (Default)";
                verifTemporal.className = "verif-code-cyan";
            }
            await fetchTileForBbox(state.activeBbox, state.activePreset);
        });
    }

    if (btnModeMulti) {
        if (btnModeMulti) btnModeMulti.addEventListener("click", async () => {
            btnModeMulti.classList.add("active");
            if (btnModeSingle) btnModeSingle.classList.remove("active");
            state.acquisitionMode = "multi_temporal";
            if (verifTemporal) {
                verifTemporal.innerText = "3-Pass Temporal Median (Active)";
                verifTemporal.className = "verif-code-emerald";
            }
            loadingScrim.classList.remove("hidden");
            try {
                const res = await fetch("/api/fetch-multi-temporal", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        aoi_id: state.activePreset || "punjab_agri",
                        bbox: state.activeBbox
                    })
                });
                const data = await res.json();
                if (data.status === "success") {
                    state.latestLrUrl = data.lr_preview;
                    state.latestHrUrl = data.hr_preview;
                    if (data.cloud_mask_preview) state.latestCloudUrl = data.cloud_mask_preview;
                    if (data.cloud_coverage_pct !== undefined) state.cloudCoveragePct = data.cloud_coverage_pct;
                    if (data.cloud_warning) updateCloudWarning(data.cloud_warning);
                    loadViewerImage(viewerLR, data.lr_preview);
                    loadViewerImage(viewerSideLR, data.lr_preview);
                    loadViewerImage(viewerSideHR, data.hr_preview);
                    if (verifReference) verifReference.innerText = data.reference_file;

                    if (data.quality_comparison) {
                        console.log("[GEO-SRM] Multi-temporal benchmark improvement:", data.quality_comparison);
                    }
                    await runSuperResolution();
                }
            } catch (err) {
                console.error("[GEO-SRM] Multi-temporal fetch failed:", err);
            } finally {
                loadingScrim.classList.add("hidden");
            }
        });
    }

    // Confidence Heatmap Toggle
    if (toggleConfidencePill) toggleConfidencePill.addEventListener("click", () => {
        state.confidenceActive = !state.confidenceActive;
        toggleConfidencePill.classList.toggle("active", state.confidenceActive);
        updateOverlayDisplay();
    });

    if (sliderConfidenceOpacity) sliderConfidenceOpacity.addEventListener("input", (e) => {
        state.confidenceOpacity = parseFloat(e.target.value);
        updateOverlayDisplay();
    });

    // Infrastructure Detection Toggle (Section 12)
    if (toggleInfraPill) {
        if (toggleInfraPill) toggleInfraPill.addEventListener("click", () => {
            state.infraActive = !state.infraActive;
            toggleInfraPill.classList.toggle("active", state.infraActive);
            toggleInfraPill.classList.toggle("infra-active", state.infraActive);
            updateOverlayDisplay();
        });
    }

    if (sliderInfraOpacity) {
        if (sliderInfraOpacity) sliderInfraOpacity.addEventListener("input", (e) => {
            state.infraOpacity = parseFloat(e.target.value);
            updateOverlayDisplay();
        });
    }

    // Cloud Mask Occlusion Toggle
    if (toggleCloudPill) {
        if (toggleCloudPill) toggleCloudPill.addEventListener("click", () => {
            state.cloudActive = !state.cloudActive;
            toggleCloudPill.classList.toggle("active", state.cloudActive);
            toggleCloudPill.classList.toggle("cloud-active", state.cloudActive);
            syncCloudOverlays();
        });
    }

    if (sliderCloudOpacity) {
        if (sliderCloudOpacity) sliderCloudOpacity.addEventListener("input", (e) => {
            state.cloudOpacity = parseFloat(e.target.value);
            updateOverlayDisplay();
        });
    }

    if (btnDismissCloudWarning) {
        if (btnDismissCloudWarning) btnDismissCloudWarning.addEventListener("click", () => {
            updateCloudWarning(null);
        });
    }

    // View Mode Switcher
    if (tabSplit) tabSplit.addEventListener("click", () => {
        state.viewMode = "split";
        tabSplit.classList.add("active");
        tabSide.classList.remove("active");
        sliderContainer.classList.remove("hidden");
        sideBySideContainer.classList.add("hidden");
        setTimeout(() => {
            [viewerLR, viewerSR].forEach(v => {
                if (v && v.isOpen()) fitViewerToImage(v);
            });
        }, 50);
    });

    if (tabSide) tabSide.addEventListener("click", () => {
        state.viewMode = "side";
        tabSide.classList.add("active");
        tabSplit.classList.remove("active");
        sliderContainer.classList.add("hidden");
        sideBySideContainer.classList.remove("hidden");

        // Force viewport refresh for side-by-side OpenSeadragon instances
        if (state.latestLrUrl) loadViewerImage(viewerSideLR, state.latestLrUrl);
        if (state.latestSrUrl) loadViewerImage(viewerSideSR, state.latestSrUrl);
        if (state.latestHrUrl) loadViewerImage(viewerSideHR, state.latestHrUrl);
        setTimeout(() => {
            [viewerSideLR, viewerSideSR, viewerSideHR].forEach(v => {
                if (v && v.isOpen()) fitViewerToImage(v);
            });
        }, 50);
    });

    // Run SR Button (if present)
    if (btnRunSr) {
        if (btnRunSr) btnRunSr.addEventListener("click", () => {
            runSuperResolution();
        });
    }

    // File Upload
    if (btnTriggerUpload) btnTriggerUpload.addEventListener("click", () => fileUpload && fileUpload.click());

    if (fileUpload) fileUpload.addEventListener("change", async (e) => {
        if (!e.target.files.length) return;
        const file = e.target.files[0];
        const formData = new FormData();
        formData.append("file", file);

        loadingScrim.classList.remove("hidden");
        try {
            const res = await fetch("/api/upload", {
                method: "POST",
                body: formData
            });
            const data = await res.json();
            if (data.status === "success") {
                state.latestLrUrl = data.lr_preview;
                state.latestHrUrl = data.hr_preview;
                if (data.cloud_mask_preview) state.latestCloudUrl = data.cloud_mask_preview;
                if (data.cloud_coverage_pct !== undefined) state.cloudCoveragePct = data.cloud_coverage_pct;
                if (data.cloud_warning) updateCloudWarning(data.cloud_warning);
                loadViewerImage(viewerLR, data.lr_preview);
                loadViewerImage(viewerSideLR, data.lr_preview);
                loadViewerImage(viewerSideHR, data.hr_preview);
                document.querySelectorAll(".aoi-card").forEach(c => c.classList.remove("active"));
                await runSuperResolution();
            }
        } catch (err) {
            console.error("[GEO-SRM] Upload failed:", err);
        } finally {
            loadingScrim.classList.add("hidden");
        }
    });

    // Export Handlers
    if (btnDownloadSr) btnDownloadSr.addEventListener("click", () => {
        window.open("/api/download/sr", "_blank");
    });

    if (btnDownloadConf) btnDownloadConf.addEventListener("click", () => {
        window.open("/api/download/confidence", "_blank");
    });

    if (btnDownloadGeojson) {
        if (btnDownloadGeojson) btnDownloadGeojson.addEventListener("click", () => {
            window.open("/api/download/infrastructure", "_blank");
        });
    }

    // =========================================================================
    // NETRA Blockchain Provenance Module UI Handlers (Polygon Amoy Testnet)
    // =========================================================================
    const headerChainVersion = document.getElementById("header-chain-version");
    const verifBlockchainStatus = document.getElementById("verif-blockchain-status");
    const btnOpenBlockchain = document.getElementById("btn-open-blockchain");
    const btnFooterOpenBlockchain = document.getElementById("btn-footer-open-blockchain");
    const modalBlockchain = document.getElementById("modal-blockchain");
    const btnCloseBlockchainModal = document.getElementById("btn-close-blockchain-modal");
    const btnDoneBlockchain = document.getElementById("btn-done-blockchain");
    const tabBcHistory = document.getElementById("tab-bc-history");
    const tabBcVerify = document.getElementById("tab-bc-verify");
    const paneBcHistory = document.getElementById("pane-bc-history");
    const paneBcVerify = document.getElementById("pane-bc-verify");
    const bcTimelineList = document.getElementById("bc-timeline-list");
    const inputVerifyFile = document.getElementById("input-verify-file");
    const bcVerifyResult = document.getElementById("bc-verify-result");

    function updateBlockchainTelemetry(record) {
        if (!record) return;
        const verTag = `v${record.versionNumber} Verified`;
        if (headerChainVersion) {
            headerChainVersion.innerText = verTag;
        }
        if (verifBlockchainStatus) {
            const shortHash = record.imageHash ? `${record.imageHash.slice(0, 10)}...` : "";
            verifBlockchainStatus.innerText = `v${record.versionNumber} (${shortHash}) • Anchored`;
        }

        const bcTileId = document.getElementById("bc-tile-id");
        const bcVersionTag = document.getElementById("bc-version-tag");
        const bcCoords = document.getElementById("bc-coords");
        const bcHash = document.getElementById("bc-hash");
        const bcTxLink = document.getElementById("bc-tx-link");

        if (bcTileId) bcTileId.innerText = record.tileId || state.activeAoiId || "hacknation_tile_001";
        if (bcVersionTag) bcVersionTag.innerText = `v${record.versionNumber} (Verified On-Chain)`;
        if (bcCoords && record.realCoordinates) {
            bcCoords.innerText = `${record.realCoordinates.latCenter.toFixed(4)}°N, ${record.realCoordinates.lonCenter.toFixed(4)}°E`;
        }
        if (bcHash) {
            bcHash.innerText = record.imageHash ? `${record.imageHash.slice(0, 18)}...` : "0x...";
            bcHash.title = record.imageHash || "";
        }
        if (bcTxLink && record.txLink) {
            bcTxLink.href = record.txLink;
        }
    }

    async function loadBlockchainHistory() {
        const tileId = state.blockchainRecord?.tileId || state.activeAoiId || "punjab_agri";
        if (!bcTimelineList) return;

        bcTimelineList.innerHTML = `<div style="color: #9ca3af; font-size: 0.82rem; padding: 1rem 0;">Querying on-chain version history chain...</div>`;

        try {
            const res = await fetch(`/api/blockchain/history/${tileId}`);
            const data = await res.json();
            if (data.status === "success" && data.history) {
                if (data.history.length === 0) {
                    bcTimelineList.innerHTML = `<div style="color: #9ca3af; font-size: 0.82rem; padding: 1rem 0;">No on-chain records found for tile '${tileId}'. Run Super-Resolution to register v1.</div>`;
                    return;
                }

                let html = "";
                const items = data.history;
                items.forEach((item, index) => {
                    const isLatest = (index === items.length - 1);
                    const isFirst = item.is_first_version;
                    html += `
                        <div class="bc-timeline-item">
                            <div class="bc-timeline-indicator">
                                <div class="bc-timeline-circle ${isLatest ? 'latest' : ''}">${item.version}</div>
                                ${!isLatest ? '<div class="bc-timeline-connector"></div>' : ''}
                            </div>
                            <div class="bc-timeline-card">
                                <div class="bc-card-header">
                                    <div class="bc-card-title">${item.version_tag} • ${item.model_name} (${item.scale_factor}x) ${isLatest ? '<span style="color:#10b981; font-size:0.75rem; margin-left:6px;">● Latest Active</span>' : ''}</div>
                                    <span class="bc-card-time">${item.timestamp_iso}</span>
                                </div>
                                <div class="bc-card-hashes">
                                    <div class="bc-hash-row">
                                        <span class="bc-hash-label">Current SHA-256:</span>
                                        <span class="bc-hash-val">${item.image_hash_short}</span>
                                    </div>
                                    <div class="bc-hash-row">
                                        <span class="bc-hash-label">Linked Previous Hash:</span>
                                        <span class="bc-hash-prev" style="${isFirst ? 'color:#6b7280;' : ''}">${isFirst ? '0x00000000... (Genesis Root v1)' : item.previous_hash_short}</span>
                                    </div>
                                </div>
                                <div class="bc-card-meta">
                                    <span>Submitter: <code style="color:#c084fc; font-family:monospace;">${item.submitter_short}</code></span>
                                    <span>Centre: ${item.lat.toFixed(4)}°N, ${item.lon.toFixed(4)}°E</span>
                                    <a href="${item.tx_link}" target="_blank" rel="noopener noreferrer" style="color:#38bdf8; text-decoration:none;">View Block ↗</a>
                                </div>
                            </div>
                        </div>
                    `;
                });
                bcTimelineList.innerHTML = html;
            }
        } catch (err) {
            bcTimelineList.innerHTML = `<div style="color: #f87171; font-size: 0.82rem; padding: 1rem 0;">Error fetching history: ${err.message}</div>`;
        }
    }

    function openBlockchainModal() {
        if (!modalBlockchain) return;
        modalBlockchain.classList.remove("hidden");
        if (state.blockchainRecord) {
            updateBlockchainTelemetry(state.blockchainRecord);
        }
        loadBlockchainHistory();
    }

    function closeBlockchainModal() {
        if (!modalBlockchain) return;
        modalBlockchain.classList.add("hidden");
    }

    if (btnOpenBlockchain) btnOpenBlockchain.addEventListener("click", openBlockchainModal);
    if (btnFooterOpenBlockchain) btnFooterOpenBlockchain.addEventListener("click", openBlockchainModal);
    if (btnCloseBlockchainModal) btnCloseBlockchainModal.addEventListener("click", closeBlockchainModal);
    if (btnDoneBlockchain) btnDoneBlockchain.addEventListener("click", closeBlockchainModal);

    if (modalBlockchain) {
        if (modalBlockchain) modalBlockchain.addEventListener("click", (e) => {
            if (e.target === modalBlockchain) closeBlockchainModal();
        });
    }

    // Modal Tabs
    if (tabBcHistory && tabBcVerify && paneBcHistory && paneBcVerify) {
        if (tabBcHistory) tabBcHistory.addEventListener("click", () => {
            tabBcHistory.classList.add("active");
            tabBcVerify.classList.remove("active");
            paneBcHistory.classList.remove("hidden");
            paneBcVerify.classList.add("hidden");
        });
        if (tabBcVerify) tabBcVerify.addEventListener("click", () => {
            tabBcVerify.classList.add("active");
            tabBcHistory.classList.remove("active");
            paneBcVerify.classList.remove("hidden");
            paneBcHistory.classList.add("hidden");
        });
    }

    // File Verifier Handler
    if (inputVerifyFile && bcVerifyResult) {
        if (inputVerifyFile) inputVerifyFile.addEventListener("change", async (e) => {
            if (!e.target.files.length) return;
            const file = e.target.files[0];
            const tileId = state.blockchainRecord?.tileId || state.activeAoiId || "punjab_agri";

            bcVerifyResult.className = "bc-verify-result";
            bcVerifyResult.classList.remove("hidden");
            bcVerifyResult.innerHTML = `<div>⟳ Computing SHA-256 and verifying against Polygon Amoy on-chain record for '<strong>${tileId}</strong>'...</div>`;

            const formData = new FormData();
            formData.append("file", file);

            try {
                const res = await fetch(`/api/blockchain/verify-file?tile_id=${tileId}`, {
                    method: "POST",
                    body: formData
                });
                const data = await res.json();
                if (data.status === "success" && data.result) {
                    const r = data.result;
                    if (r.verified) {
                        bcVerifyResult.className = "bc-verify-result success";
                        bcVerifyResult.innerHTML = `
                            <strong>✓ On-Chain Cryptographic Match (Authentic)</strong>
                            <div style="margin-top: 4px; font-size: 0.76rem; color: #d1fae5;">
                                File matches <strong>Version ${r.version_number}</strong> on Polygon Amoy testnet.<br>
                                Content SHA-256: <code style="color:#a7f3d0;">${r.provided_hash.slice(0, 20)}...</code><br>
                                Registered Timestamp: ${new Date(r.timestamp * 1000).toUTCString()}
                            </div>
                        `;
                    } else {
                        bcVerifyResult.className = "bc-verify-result error";
                        bcVerifyResult.innerHTML = `
                            <strong>⚠ Tamper Detected: Hash Mismatch</strong>
                            <div style="margin-top: 4px; font-size: 0.76rem; color: #fecaca;">
                                The uploaded file does not match the latest registered version on-chain.<br>
                                Uploaded Hash: <code style="color:#fda4af;">${r.provided_hash.slice(0, 20)}...</code><br>
                                Expected On-Chain Hash: <code style="color:#6ee7b7;">${r.registered_hash.slice(0, 20)}...</code>
                            </div>
                        `;
                    }
                } else {
                    bcVerifyResult.className = "bc-verify-result error";
                    bcVerifyResult.innerText = `Verification error: ${data.detail || 'Unknown error'}`;
                }
            } catch (err) {
                bcVerifyResult.className = "bc-verify-result error";
                bcVerifyResult.innerText = `Failed to connect to verification endpoint: ${err.message}`;
            }
        });
    }

    // =========================================================================
    // Dedicated Spectral & Disaster Analysis View (View 2) Logic
    // =========================================================================
    let spectralMap = null;
    let spectralAoiRectangle = null;
    let spectralDrawnItems = null;
    let spectralDrawControl = null;

    function initSpectralMap() {
        const mapEl = document.getElementById("spectral-aoi-map");
        if (!mapEl || spectralMap) return;

        spectralMap = L.map("spectral-aoi-map", {
            zoomControl: false,
            attributionControl: false
        }).setView([30.575, 75.33], 12);

        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            maxZoom: 18,
        }).addTo(spectralMap);

        spectralAoiRectangle = L.rectangle([
            [state.activeBbox[1], state.activeBbox[0]],
            [state.activeBbox[3], state.activeBbox[2]]
        ], {
            color: "#10b981",
            weight: 2,
            fillColor: "#10b981",
            fillOpacity: 0.18
        }).addTo(spectralMap);

        spectralDrawnItems = new L.FeatureGroup();
        spectralMap.addLayer(spectralDrawnItems);

        spectralDrawControl = new L.Control.Draw({
            position: 'topright',
            draw: {
                polygon: false,
                polyline: false,
                circle: false,
                circlemarker: false,
                marker: false,
                rectangle: {
                    shapeOptions: {
                        color: "#10b981",
                        weight: 2,
                        fillColor: "#10b981",
                        fillOpacity: 0.20
                    }
                }
            },
            edit: { featureGroup: spectralDrawnItems }
        });
        spectralMap.addControl(spectralDrawControl);

        spectralMap.on(L.Draw.Event.CREATED, function (e) {
            const layer = e.layer;
            spectralDrawnItems.clearLayers();
            spectralDrawnItems.addLayer(layer);

            const bounds = layer.getBounds();
            const bbox = [
                parseFloat(bounds.getWest().toFixed(4)),
                parseFloat(bounds.getSouth().toFixed(4)),
                parseFloat(bounds.getEast().toFixed(4)),
                parseFloat(bounds.getNorth().toFixed(4))
            ];

            state.activeBbox = bbox;
            state.activePreset = "custom_drawn_aoi";
            const center = bounds.getCenter();
            syncAoiUI(bounds, bbox, "custom_drawn_aoi", center.lat, center.lng);
            fetchTileForBbox(bbox, "custom_drawn_aoi");
        });

        spectralMap.on(L.Draw.Event.EDITED, function (e) {
            const layers = e.layers;
            layers.eachLayer(function (layer) {
                const bounds = layer.getBounds();
                const bbox = [
                    parseFloat(bounds.getWest().toFixed(4)),
                    parseFloat(bounds.getSouth().toFixed(4)),
                    parseFloat(bounds.getEast().toFixed(4)),
                    parseFloat(bounds.getNorth().toFixed(4))
                ];
                state.activeBbox = bbox;
                state.activePreset = "custom_drawn_aoi";
                const center = bounds.getCenter();
                syncAoiUI(bounds, bbox, "custom_drawn_aoi", center.lat, center.lng);
                fetchTileForBbox(bbox, "custom_drawn_aoi");
            });
        });

        spectralMap.on(L.Draw.Event.DELETED, function () {
            if (state.presets && state.presets.length > 0) {
                selectPreset(state.presets[0].id);
            }
        });

        spectralMap.on("click", (e) => {
            if (document.querySelector(".leaflet-draw-actions")) return;
            const lat = e.latlng.lat;
            const lon = e.latlng.lng;
            const span = 0.03;
            const customBbox = [
                parseFloat((lon - span).toFixed(4)),
                parseFloat((lat - span).toFixed(4)),
                parseFloat((lon + span).toFixed(4)),
                parseFloat((lat + span).toFixed(4))
            ];

            state.activeBbox = customBbox;
            state.activePreset = "custom_click";
            const bounds = [
                [customBbox[1], customBbox[0]],
                [customBbox[3], customBbox[2]]
            ];

            if (spectralDrawnItems) spectralDrawnItems.clearLayers();
            if (drawnItems) drawnItems.clearLayers();
            if (spectralMap) spectralMap.panTo([lat, lon]);
            if (map) map.panTo([lat, lon]);

            syncAoiUI(bounds, customBbox, "custom_click", lat, lon);
            fetchTileForBbox(customBbox, "custom_click");
        });
    }

    // View 2 Manual Coordinate Navigation and Draw Buttons
    const spectralBtnGotoCoords = document.getElementById("spectral-btn-goto-coords");
    const spectralInputLat = document.getElementById("spectral-input-lat");
    const spectralInputLon = document.getElementById("spectral-input-lon");
    const spectralBtnDrawRect = document.getElementById("spectral-btn-draw-rect");

    if (spectralBtnGotoCoords) {
        if (spectralBtnGotoCoords) spectralBtnGotoCoords.addEventListener("click", () => {
            const lat = parseFloat(spectralInputLat ? spectralInputLat.value.trim() : NaN);
            const lon = parseFloat(spectralInputLon ? spectralInputLon.value.trim() : NaN);
            navigateToCoordinates(lat, lon);
        });
    }

    const handleSpectralCoordEnter = (e) => {
        if (e.key === "Enter") {
            const lat = parseFloat(spectralInputLat ? spectralInputLat.value.trim() : NaN);
            const lon = parseFloat(spectralInputLon ? spectralInputLon.value.trim() : NaN);
            navigateToCoordinates(lat, lon);
        }
    };
    if (spectralInputLat) spectralInputLat.addEventListener("keydown", handleSpectralCoordEnter);
    if (spectralInputLon) spectralInputLon.addEventListener("keydown", handleSpectralCoordEnter);

    if (spectralBtnDrawRect) {
        if (spectralBtnDrawRect) spectralBtnDrawRect.addEventListener("click", () => {
            initSpectralMap();
            if (spectralMap && spectralDrawControl) {
                new L.Draw.Rectangle(spectralMap, spectralDrawControl.options.draw.rectangle).enable();
            }
        });
    }

    // Top-Level View Switching
    window.onViewSwitched = function onViewSwitched(viewName) {
        state.activeTopView = viewName;
        console.log(`[GEO-SRM] View switch callback active: ${viewName}`);

        if (viewName === "spectral") {
            try {
                initSpectralMap();
            } catch (e) {
                console.warn("[GEO-SRM] Leaflet spectral map init notice:", e);
            }

            const spectralContainer = document.getElementById("view-spectral-analysis");
            if (spectralContainer) void spectralContainer.offsetHeight;

            try {
                ensureSpectralViewers();
            } catch (e) {
                console.warn("[GEO-SRM] Spectral viewers init notice:", e);
            }

            setSpectralSliderPercent(50);

            setTimeout(() => {
                try {
                    if (spectralMap) {
                        spectralMap.invalidateSize();
                        if (state.activeBbox) {
                            spectralMap.fitBounds([
                                [state.activeBbox[1], state.activeBbox[0]],
                                [state.activeBbox[3], state.activeBbox[2]]
                            ]);
                        }
                    }
                } catch (e) { }

                [spectralViewerLR, spectralViewerSR, spectralViewerSingle].forEach(v => {
                    try {
                        if (v && v.viewport && v.isOpen()) {
                            fitViewerToImage(v);
                        }
                    } catch (e) { }
                });
                window.dispatchEvent(new Event("resize"));
            }, 100);

            const currentMode = state.activeSpectralMode || "ndvi";
            selectSpectralMode(currentMode);
        } else {
            setTimeout(() => {
                try {
                    if (map) {
                        map.invalidateSize();
                        if (state.activeBbox) {
                            map.fitBounds([
                                [state.activeBbox[1], state.activeBbox[0]],
                                [state.activeBbox[3], state.activeBbox[2]]
                            ]);
                        }
                    }
                } catch (e) { }

                [viewerLR, viewerSR, viewerSideLR, viewerSideSR, viewerSideHR].forEach(v => {
                    try {
                        if (v && v.viewport && v.isOpen()) {
                            fitViewerToImage(v);
                        }
                    } catch (e) { }
                });
                window.dispatchEvent(new Event("resize"));
            }, 100);
        }
    };

    // If inline script recorded a pending switch before app.js was ready, handle it now
    if (window._pendingViewSwitch) {
        const pending = window._pendingViewSwitch;
        window._pendingViewSwitch = null;
        window.onViewSwitched(pending);
    }

    if (navBtnSr) navBtnSr.addEventListener("click", () => window.switchTopView("sr"));
    if (navBtnSpectral) navBtnSpectral.addEventListener("click", () => window.switchTopView("spectral"));

    // Dedicated Spectral Mode Cards Click Handlers
    async function selectSpectralMode(mode) {
        state.activeSpectralMode = mode;

        document.querySelectorAll(".spectral-mode-tab").forEach(tab => {
            tab.classList.toggle("active", tab.dataset.mode === mode);
        });

        const spectralLoadingScrim = document.getElementById("spectral-loading-scrim");
        if (spectralLoadingScrim) spectralLoadingScrim.classList.remove("hidden");

        try {
            const res = await fetch("/api/visualize", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    mode: mode,
                    apply_realesrgan_sharpen: false
                })
            });
            const data = await res.json();
            if (data.status === "success") {
                if (data.lr_preview) {
                    state.latestSpectralLrUrl = data.lr_preview;
                    const imgLr = document.getElementById("spectral-img-lr");
                    if (imgLr) imgLr.src = data.lr_preview;
                    if (spectralViewerLR) loadViewerImage(spectralViewerLR, data.lr_preview, false);
                }
                if (data.sr_preview) {
                    state.latestSpectralSrUrl = data.sr_preview;
                    const imgSr = document.getElementById("spectral-img-sr");
                    const imgSingle = document.getElementById("spectral-img-single");
                    if (imgSr) imgSr.src = data.sr_preview;
                    if (imgSingle) imgSingle.src = data.sr_preview;
                    if (spectralViewerSR) loadViewerImage(spectralViewerSR, data.sr_preview, false);
                    if (spectralViewerSingle) loadViewerImage(spectralViewerSingle, data.sr_preview, false);
                }

                // Update Viewport Labels
                const tagSr = document.getElementById("spectral-tag-sr");
                if (tagSr) {
                    tagSr.innerText = data.swir_label 
                        ? `Output: HAT 2.5m (${data.title}) [${data.swir_label}]`
                        : `Output: HAT 2.5m (${data.title})`;
                }
                const singleHeader = document.getElementById("spectral-single-header");
                if (singleHeader) {
                    singleHeader.innerText = data.swir_label
                        ? `Super-Resolved 2.5m — ${data.title} (${data.swir_label})`
                        : `Super-Resolved 2.5m — ${data.title}`;
                }

                // Update Primary Area Coverage Stat Badge (Km² affected / covered)
                const statTitle = document.getElementById("stat-category-title");
                const statBadge = document.getElementById("stat-threshold-badge");
                const statDesc = document.getElementById("stat-category-desc");
                const statAreaKm2 = document.getElementById("stat-area-km2");
                const statAreaPct = document.getElementById("stat-area-pct");
                const statTotalAoi = document.getElementById("stat-total-aoi");

                if (data.coverage_stats) {
                    const cs = data.coverage_stats;
                    if (statTitle) statTitle.innerText = cs.category;
                    if (statBadge) statBadge.innerText = cs.threshold;
                    if (statAreaKm2) statAreaKm2.innerText = `${cs.area_km2} km²`;
                    if (statAreaPct) statAreaPct.innerText = `${cs.area_pct}% of AOI`;
                    if (statTotalAoi) statTotalAoi.innerText = `${cs.total_aoi_km2} km²`;
                }

                // Update Numerical Index Distribution
                const elMean = document.getElementById("metric-sp-mean");
                const elMax = document.getElementById("metric-sp-max");
                const elMin = document.getElementById("metric-sp-min");
                const elStd = document.getElementById("metric-sp-std");

                if (data.stats) {
                    if (elMean) elMean.innerText = data.stats.mean !== undefined ? data.stats.mean.toFixed(3) : "--";
                    if (elMax) elMax.innerText = data.stats.max !== undefined ? data.stats.max.toFixed(3) : "--";
                    if (elMin) elMin.innerText = data.stats.min !== undefined ? data.stats.min.toFixed(3) : "--";
                    if (elStd) elStd.innerText = data.stats.std !== undefined ? data.stats.std.toFixed(3) : "--";
                }

                // Update Scientific Legend / Continuous Ramp
                const spLegendTitle = document.getElementById("sp-legend-title");
                const spLegendStats = document.getElementById("sp-legend-stats");
                const spLegendBarContainer = document.getElementById("sp-legend-bar-container");
                const spLegendBar = document.getElementById("sp-legend-bar");
                const spTickMin = document.getElementById("sp-tick-min");
                const spTickMid = document.getElementById("sp-tick-mid");
                const spTickMax = document.getElementById("sp-tick-max");

                const modeIcons = {
                    true_color: "🌿",
                    ndvi: "🌱",
                    ndwi: "💧",
                    nbr: "🔥",
                    ndbi: "🏢"
                };
                const icon = modeIcons[mode] || "🛰️";
                if (spLegendTitle) spLegendTitle.innerText = `${icon} ${data.title}`;
                if (spLegendStats) spLegendStats.innerText = `${data.subtitle} | Bands: ${data.bands_used.join("-")}`;

                if (data.is_index && data.legend && data.legend.type === "continuous") {
                    if (spLegendBarContainer) spLegendBarContainer.style.display = "flex";
                    if (spLegendBar) spLegendBar.style.background = data.legend.gradient_css || "linear-gradient(to right, #000, #fff)";
                    if (spTickMin) spTickMin.innerText = data.legend.min_label || `${data.legend.min}`;
                    if (spTickMid) spTickMid.innerText = data.legend.mid_label || "0.0";
                    if (spTickMax) spTickMax.innerText = data.legend.max_label || `${data.legend.max}`;
                } else {
                    if (spLegendBarContainer) spLegendBarContainer.style.display = "none";
                }

                // Update Scientific Decision Support Callout
                const decisionBox = document.getElementById("spectral-decision-box");
                if (decisionBox) {
                    const interpretations = {
                        ndvi: "<strong>Agricultural Monitoring:</strong> NDVI ≥ 0.40 isolates active crop canopies and chlorophyll absorption. Resolves individual agricultural parcel boundaries and irrigation methods at 2.5m GSD.",
                        ndwi: "<strong>Flood & Hydrological Inundation:</strong> NDWI ≥ 0.00 detects standing surface water, flooded farmland, and riverbank breaching with zero spectral ambiguity.",
                        nbr: "<strong>Disaster Burn Severity:</strong> NBR ≤ 0.10 highlights active fire burn scars, agricultural stubble combustion zones, and deforested terrain.",
                        ndbi: "<strong>Urban Built-up Extent:</strong> NDBI ≥ 0.00 isolates concrete, asphalt, building footprints, and dense settlement infrastructure.",
                        true_color: "<strong>Natural Surface Reflectance:</strong> True Color (B04-Red, B03-Green, B02-Blue) with dynamic AOI radiometric color transfer for natural visual basemap consistency."
                    };
                    decisionBox.innerHTML = interpretations[mode] || `<strong>Multi-Band Analysis:</strong> ${data.description}`;
                }
            } else if (data.status === "unavailable") {
                console.warn("[GEO-SRM] Spectral mode unavailable:", data.message);
                const tagSr = document.getElementById("spectral-tag-sr");
                if (tagSr) tagSr.innerText = `Output: ${data.title} (unavailable)`;
                const singleHeader = document.getElementById("spectral-single-header");
                if (singleHeader) singleHeader.innerText = `${data.title} — unavailable`;

                const statTitle = document.getElementById("stat-category-title");
                const statBadge = document.getElementById("stat-threshold-badge");
                const statAreaKm2 = document.getElementById("stat-area-km2");
                const statAreaPct = document.getElementById("stat-area-pct");
                if (statTitle) statTitle.innerText = `${data.title} Unavailable`;
                if (statBadge) statBadge.innerText = "unavailable";
                if (statAreaKm2) statAreaKm2.innerText = "unavailable";
                if (statAreaPct) statAreaPct.innerText = "unavailable";

                const elMean = document.getElementById("metric-sp-mean");
                const elMax = document.getElementById("metric-sp-max");
                const elMin = document.getElementById("metric-sp-min");
                const elStd = document.getElementById("metric-sp-std");
                if (elMean) elMean.innerText = "unavailable";
                if (elMax) elMax.innerText = "unavailable";
                if (elMin) elMin.innerText = "unavailable";
                if (elStd) elStd.innerText = "unavailable";

                const spLegendTitle = document.getElementById("sp-legend-title");
                const spLegendStats = document.getElementById("sp-legend-stats");
                const spLegendBarContainer = document.getElementById("sp-legend-bar-container");
                if (spLegendTitle) spLegendTitle.innerText = `⚠️ ${data.title} (unavailable)`;
                if (spLegendStats) spLegendStats.innerText = data.message || "Required spectral band missing — calculation unavailable.";
                if (spLegendBarContainer) spLegendBarContainer.style.display = "none";
            }
        } catch (err) {
            console.error("[GEO-SRM] Failed to switch spectral mode:", err);
        } finally {
            if (spectralLoadingScrim) spectralLoadingScrim.classList.add("hidden");
        }
    }

    document.querySelectorAll(".spectral-mode-tab").forEach(tab => {
        tab.addEventListener("click", () => {
            const mode = tab.dataset.mode;
            if (mode) selectSpectralMode(mode);
        });
    });

    // Dedicated Spectral View Split-Slider Drag Interaction
    const spectralSliderContainer = document.getElementById("spectral-slider-container");
    const spectralSliderHandle = document.getElementById("spectral-slider-handle");
    const spectralSrWrapper = document.getElementById("spectral-sr-wrapper");

    function setSpectralSliderPercent(percentage) {
        if (!spectralSliderHandle || !spectralSrWrapper) return;
        percentage = Math.max(0, Math.min(100, percentage));
        spectralSliderHandle.style.left = `${percentage}%`;
        spectralSrWrapper.style.clipPath = `polygon(${percentage}% 0, 100% 0, 100% 100%, ${percentage}% 100%)`;
    }

    let isSpectralDragging = false;
    function updateSpectralSliderPosition(clientX) {
        if (!spectralSliderContainer) return;
        const rect = spectralSliderContainer.getBoundingClientRect();
        if (rect.width <= 0) return;
        let offsetX = clientX - rect.left;
        let percentage = (offsetX / rect.width) * 100;
        setSpectralSliderPercent(percentage);
    }

    if (spectralSliderHandle && spectralSliderContainer) {
        if (spectralSliderHandle) spectralSliderHandle.addEventListener("mousedown", (e) => {
            isSpectralDragging = true;
            e.preventDefault();
        });

        window.addEventListener("mouseup", () => {
            isSpectralDragging = false;
        });

        window.addEventListener("mousemove", (e) => {
            if (!isSpectralDragging) return;
            updateSpectralSliderPosition(e.clientX);
        });

        if (spectralSliderHandle) spectralSliderHandle.addEventListener("touchstart", () => {
            isSpectralDragging = true;
        });

        window.addEventListener("touchend", () => {
            isSpectralDragging = false;
        });

        window.addEventListener("touchmove", (e) => {
            if (!isSpectralDragging || !e.touches.length) return;
            updateSpectralSliderPosition(e.touches[0].clientX);
        });

        if (spectralSliderContainer) spectralSliderContainer.addEventListener("click", (e) => {
            if (e.target.closest("#spectral-slider-handle")) return;
            updateSpectralSliderPosition(e.clientX);
        });
    }

    // View 2 Split vs Single View Switcher
    const tabSpectralSplit = document.getElementById("tab-spectral-split");
    const tabSpectralSingle = document.getElementById("tab-spectral-single");
    const spectralSingleContainer = document.getElementById("spectral-single-container");

    if (tabSpectralSplit && tabSpectralSingle && spectralSliderContainer && spectralSingleContainer) {
        if (tabSpectralSplit) tabSpectralSplit.addEventListener("click", () => {
            tabSpectralSplit.classList.add("active");
            tabSpectralSingle.classList.remove("active");
            spectralSingleContainer.classList.add("hidden");
            spectralSingleContainer.style.display = "none";
            spectralSliderContainer.style.display = "block";
            setSpectralSliderPercent(50);
            if (spectralViewerSR && spectralViewerSR.viewport) {
                fitViewerToImage(spectralViewerSR);
            }
            if (spectralViewerLR && spectralViewerLR.viewport) {
                fitViewerToImage(spectralViewerLR);
            }
        });

        if (tabSpectralSingle) tabSpectralSingle.addEventListener("click", () => {
            tabSpectralSingle.classList.add("active");
            tabSpectralSplit.classList.remove("active");
            spectralSliderContainer.style.display = "none";
            spectralSingleContainer.classList.remove("hidden");
            spectralSingleContainer.style.display = "flex";
            const imgSingle = document.getElementById("spectral-img-single");
            if (imgSingle && state.latestSpectralSrUrl) {
                imgSingle.src = state.latestSpectralSrUrl;
            }
            if (spectralViewerSingle && spectralViewerSingle.viewport) {
                fitViewerToImage(spectralViewerSingle);
            }
        });
    }

    // View 2 OpenSeadragon Zoom Controls
    const btnSpectralZoomIn = document.getElementById("btn-spectral-zoom-in");
    const btnSpectralZoomOut = document.getElementById("btn-spectral-zoom-out");
    const btnSpectralZoomReset = document.getElementById("btn-spectral-zoom-reset");
    const spectralZoomBadge = document.getElementById("spectral-zoom-badge");

    function updateSpectralZoomBadge(v) {
        const isSingle = tabSpectralSingle && tabSpectralSingle.classList.contains("active");
        const activeViewer = v || (isSingle 
            ? spectralViewerSingle 
            : ((spectralViewerSR && spectralViewerSR.isOpen()) ? spectralViewerSR : spectralViewerLR));
        if (!spectralZoomBadge || !activeViewer || !activeViewer.viewport || !activeViewer.isOpen()) return;
        try {
            const hz = activeViewer.viewport.getHomeZoom() || 1.0;
            const currentZoom = activeViewer.viewport.getZoom();
            if (hz <= 0 || currentZoom <= 0 || isNaN(hz) || isNaN(currentZoom)) return;
            const relative = currentZoom / hz;
            spectralZoomBadge.innerText = (relative >= 1.0) ? `${relative.toFixed(1)}x` : `${relative.toFixed(2)}x`;
        } catch (e) { }
    }

    if (spectralViewerLR) {
        spectralViewerLR.addHandler("zoom", () => updateSpectralZoomBadge(spectralViewerLR));
    }
    if (spectralViewerSR) {
        spectralViewerSR.addHandler("zoom", () => updateSpectralZoomBadge(spectralViewerSR));
    }
    if (spectralViewerSingle) {
        spectralViewerSingle.addHandler("zoom", () => updateSpectralZoomBadge(spectralViewerSingle));
    }

    if (btnSpectralZoomIn) {
        btnSpectralZoomIn.addEventListener("click", () => {
            const isSingle = tabSpectralSingle && tabSpectralSingle.classList.contains("active");
            const v = isSingle 
                ? spectralViewerSingle 
                : ((spectralViewerSR && spectralViewerSR.isOpen()) ? spectralViewerSR : spectralViewerLR);
            if (v && v.viewport) {
                v.viewport.zoomBy(1.4);
                v.viewport.applyConstraints();
                updateSpectralZoomBadge(v);
            }
        });
    }
    if (btnSpectralZoomOut) {
        btnSpectralZoomOut.addEventListener("click", () => {
            const isSingle = tabSpectralSingle && tabSpectralSingle.classList.contains("active");
            const v = isSingle 
                ? spectralViewerSingle 
                : ((spectralViewerSR && spectralViewerSR.isOpen()) ? spectralViewerSR : spectralViewerLR);
            if (v && v.viewport) {
                v.viewport.zoomBy(0.71);
                v.viewport.applyConstraints();
                updateSpectralZoomBadge(v);
            }
        });
    }
    if (btnSpectralZoomReset) {
        btnSpectralZoomReset.addEventListener("click", () => {
            const isSingle = tabSpectralSingle && tabSpectralSingle.classList.contains("active");
            if (isSingle) {
                fitViewerToImage(spectralViewerSingle);
                updateSpectralZoomBadge(spectralViewerSingle);
            } else {
                const primary = (spectralViewerSR && spectralViewerSR.isOpen()) ? spectralViewerSR : spectralViewerLR;
                fitViewerToImage(primary);
                if (spectralViewerLR && primary !== spectralViewerLR && spectralViewerLR.isOpen()) fitViewerToImage(spectralViewerLR);
                if (spectralViewerSR && primary !== spectralViewerSR && spectralViewerSR.isOpen()) fitViewerToImage(spectralViewerSR);
                updateSpectralZoomBadge(primary);
            }
        });
    }

    // View 2 Export Handlers
    const btnSpectralDownloadPng = document.getElementById("btn-spectral-download-png");
    if (btnSpectralDownloadPng) {
        if (btnSpectralDownloadPng) btnSpectralDownloadPng.addEventListener("click", () => {
            if (state.latestSpectralSrUrl) {
                const a = document.createElement("a");
                a.href = state.latestSpectralSrUrl;
                a.download = `hacknation_spectral_${state.activeSpectralMode || 'ndvi'}_${state.activePreset || 'aoi'}.png`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
            } else {
                window.open("/api/download/sr", "_blank");
            }
        });
    }

    const btnSpectralDownloadTif = document.getElementById("btn-spectral-download-tif");
    if (btnSpectralDownloadTif) {
        if (btnSpectralDownloadTif) btnSpectralDownloadTif.addEventListener("click", () => {
            window.open("/api/download/geotiff", "_blank");
        });
    }

    // Initial load
    loadPresets();

    // =========================================================================
    // Map Modal & Geocoding Logic
    // =========================================================================

    const overlay = document.getElementById('map-modal-overlay');
    const openBtn = document.getElementById('open-map-modal');
    const closeBtn = document.getElementById('map-modal-close');
    const cancelBtn = document.getElementById('map-modal-cancel');
    const applyBtn = document.getElementById('map-modal-apply');
    const drawBtn = document.getElementById('map-modal-draw');

    const searchInput = document.getElementById('map-modal-search');
    const latInput = document.getElementById('map-modal-lat');
    const lonInput = document.getElementById('map-modal-lon');
    const jumpBtn = document.getElementById('map-modal-jump');
    const readout = document.getElementById('map-modal-readout');

    if (!overlay) return;

    let previousParent = null;
    let aoiMapElement = document.getElementById('aoi-map');

    function updateReadout() {
        if (!map) return;
        const center = map.getCenter();
        let text = `Selected: Lat ${center.lat.toFixed(4)} · Lon ${center.lng.toFixed(4)}`;
        if (state.activeBbox) {
            text += ` (BBox active)`;
        }
        if (readout) readout.innerText = text;
    }

    function openModal() {
        overlay.classList.add('open');
        previousParent = aoiMapElement.parentElement;
        const canvasContainer = document.getElementById('map-modal-canvas');
        if (canvasContainer) {
            canvasContainer.appendChild(aoiMapElement);
            setTimeout(() => {
                if (map) {
                    map.invalidateSize();
                    updateReadout();
                }
            }, 100);
        }
    }

    function closeModal() {
        overlay.classList.remove('open');
        if (previousParent) {
            previousParent.appendChild(aoiMapElement);
            setTimeout(() => {
                if (map) map.invalidateSize();
            }, 100);
        }
    }

    if (openBtn) openBtn.addEventListener('click', openModal);
    if (closeBtn) closeBtn.addEventListener('click', closeModal);
    if (cancelBtn) cancelBtn.addEventListener('click', closeModal);
    overlay.addEventListener('click', (e) => { if (e.target === overlay) closeModal(); });

    // Handle Geocoding Search
    if (searchInput) {
        searchInput.addEventListener('keydown', async (e) => {
            if (e.key === 'Enter') {
                const query = searchInput.value.trim();
                if (!query) return;
                try {
                    const res = await fetch(`https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(query)}`);
                    const data = await res.json();
                    if (data && data.length > 0) {
                        const lat = parseFloat(data[0].lat);
                        const lon = parseFloat(data[0].lon);
                        map.flyTo([lat, lon], 12);
                        if (latInput) latInput.value = lat.toFixed(4);
                        if (lonInput) lonInput.value = lon.toFixed(4);
                        updateReadout();
                    } else {
                        alert("Location not found.");
                    }
                } catch (err) {
                    console.error("Geocoding failed:", err);
                }
            }
        });
    }

    // Handle Lat/Lon Jump in Modal
    if (jumpBtn) {
        jumpBtn.addEventListener('click', () => {
            const lat = parseFloat(latInput.value);
            const lon = parseFloat(lonInput.value);
            if (!isNaN(lat) && !isNaN(lon)) {
                map.flyTo([lat, lon], 12);
                updateReadout();
            }
        });
    }

    // Handle Draw Bounding Box in Modal
    if (drawBtn) {
        drawBtn.addEventListener('click', () => {
            if (typeof drawControl !== 'undefined' && drawControl.options.draw.rectangle) {
                new L.Draw.Rectangle(map, drawControl.options.draw.rectangle).enable();
            }
        });
    }

    // Handle Apply Selection
    if (applyBtn) {
        applyBtn.addEventListener('click', () => {
            const center = map.getCenter();
            let bbox = state.activeBbox;

            if (!bbox || drawnItems.getLayers().length === 0) {
                // If no box drawn, just make a tiny box around the center
                const span = 0.03;
                bbox = [
                    parseFloat((center.lng - span).toFixed(4)),
                    parseFloat((center.lat - span).toFixed(4)),
                    parseFloat((center.lng + span).toFixed(4)),
                    parseFloat((center.lat + span).toFixed(4))
                ];
                state.activeBbox = bbox;
                state.activePreset = "custom_click";

                const bounds = [
                    [bbox[1], bbox[0]],
                    [bbox[3], bbox[2]]
                ];
                if (typeof syncAoiUI === 'function') syncAoiUI(bounds, bbox, "custom_click", center.lat, center.lng);
                if (typeof fetchTileForBbox === 'function') fetchTileForBbox(bbox, "custom_click");
            } else {
                // Bbox already drawn and handled by map.on(L.Draw.Event.CREATED)
                // Just sync coords
                const customLat = document.getElementById("input-custom-lat");
                const customLon = document.getElementById("input-custom-lon");
                if (customLat) customLat.value = center.lat.toFixed(4);
                if (customLon) customLon.value = center.lng.toFixed(4);
            }

            closeModal();
        });
    }

    // Update readout on map move
    if (typeof map !== 'undefined') {
        map.on('moveend', updateReadout);
        map.on('zoomend', updateReadout);
    }

    // Automatic Window Resize refit for full image visibility
    let windowResizeDebounce = null;
    window.addEventListener("resize", () => {
        if (windowResizeDebounce) clearTimeout(windowResizeDebounce);
        windowResizeDebounce = setTimeout(() => {
            const viewers = (state.activeTopView === "spectral")
                ? [spectralViewerLR, spectralViewerSR, spectralViewerSingle]
                : (state.viewMode === "split" ? [viewerLR, viewerSR] : [viewerSideLR, viewerSideSR, viewerSideHR]);
            viewers.forEach(v => {
                if (v && v.viewport && v.isOpen()) {
                    fitViewerToImage(v);
                }
            });
        }, 120);
    });

    // Expose global bridge handles for NETRA Landing Page and Inter-Component Coordination
    window.netraLaunchDashboard = launchDashboard;
    window.netraReturnToLanding = returnToLanding;
    window.netraSelectPreset = selectPreset;
    window.netraResetViewers = resetAllViewers;
    window.selectPreset = selectPreset;
    window.state = state;
});

