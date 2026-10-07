import React, { useState, useEffect } from "react";

/**
 * NETRA - Blockchain Provenance Component
 * Team KC Studio | Scope: Geo-Tagging + Version History Chain
 * Target Network: Polygon Amoy Testnet (Chain ID: 80002)
 *
 * Provides:
 * 1. Status Badge: 'vX - Verified on-chain' with centre lat/lon, timestamp & Polygonscan link.
 * 2. Visual Timeline: Connected node graph showing v1 -> v2 -> v3 with previousHash linkages.
 * 3. File Verification Tool: Direct SHA-256 integrity checker against on-chain record.
 */
export default function BlockchainProvenance({
  tileId = "punjab_agri",
  currentRecord = null,
  apiBaseUrl = "/api"
}) {
  const [history, setHistory] = useState([]);
  const [latest, setLatest] = useState(currentRecord);
  const [isExpanded, setIsExpanded] = useState(false);
  const [loading, setLoading] = useState(false);
  const [verifyStatus, setVerifyStatus] = useState(null);
  const [activeTab, setActiveTab] = useState("timeline"); // "timeline" | "verify"

  // Fetch version history when tileId changes or panel expands
  useEffect(() => {
    if (currentRecord) {
      setLatest(currentRecord);
    }
    fetchHistory();
  }, [tileId, currentRecord]);

  const fetchHistory = async () => {
    try {
      setLoading(true);
      const res = await fetch(`${apiBaseUrl}/blockchain/history/${tileId}`);
      const data = await res.json();
      if (data.status === "success" && data.history) {
        setHistory(data.history);
        if (data.history.length > 0 && !currentRecord) {
          setLatest(data.history[data.history.length - 1]);
        }
      }
    } catch (err) {
      console.error("[NETRA-Blockchain] Failed to fetch history:", err);
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    try {
      setVerifyStatus({ state: "checking", message: "Computing SHA-256 & querying on-chain record..." });
      const res = await fetch(`${apiBaseUrl}/blockchain/verify-file?tile_id=${tileId}`, {
        method: "POST",
        body: formData
      });
      const data = await res.json();
      if (data.status === "success" && data.result) {
        if (data.result.verified) {
          setVerifyStatus({
            state: "verified",
            message: `MATCH: File verified authentic (Version ${data.result.version_number})`,
            details: data.result
          });
        } else {
          setVerifyStatus({
            state: "tampered",
            message: "TAMPER DETECTED: Computed hash does not match latest on-chain record.",
            details: data.result
          });
        }
      }
    } catch (err) {
      setVerifyStatus({ state: "error", message: `Verification error: ${err.message}` });
    }
  };

  const versionNumber = latest?.versionNumber || latest?.version || (history.length > 0 ? history.length : 1);
  const latCenter = latest?.realCoordinates?.latCenter ?? latest?.lat ?? 31.1471;
  const lonCenter = latest?.realCoordinates?.lonCenter ?? latest?.lon ?? 75.3412;
  const txLink = latest?.txLink || latest?.tx_link || `https://amoy.polygonscan.com`;
  const imageHash = latest?.imageHash || latest?.image_hash || "0x00000000000000000000000000000000";
  const shortHash = `${imageHash.slice(0, 10)}...${imageHash.slice(-6)}`;

  return (
    <div className="netra-blockchain-card" style={styles.card}>
      {/* Header & Main Badge */}
      <div style={styles.header}>
        <div style={styles.badgeGroup}>
          <div style={styles.chainBadge}>
            <span style={styles.chainDot}></span>
            <span style={styles.chainText}>Polygon Amoy (80002)</span>
          </div>

          <div style={styles.versionBadge}>
            <span style={{ fontWeight: "700", color: "#FFD700" }}>v{versionNumber}</span>
            <span style={{ color: "#9ca3af", margin: "0 6px" }}>•</span>
            <span style={{ color: "#34d399" }}>Verified On-Chain</span>
          </div>
        </div>

        <button
          onClick={() => setIsExpanded(!isExpanded)}
          style={styles.toggleButton}
          aria-expanded={isExpanded}
        >
          {isExpanded ? "Collapse History ▲" : `View Chain (${history.length || versionNumber} versions) ▼`}
        </button>
      </div>

      {/* Metadata Telemetry Strip */}
      <div style={styles.telemetryStrip}>
        <div>
          <span style={styles.metaLabel}>Centre Point:</span>{" "}
          <span style={styles.metaVal}>{Number(latCenter).toFixed(4)}°N, {Number(lonCenter).toFixed(4)}°E</span>
        </div>
        <div>
          <span style={styles.metaLabel}>SHA-256 Hash:</span>{" "}
          <code style={styles.metaCode} title={imageHash}>{shortHash}</code>
        </div>
        <div>
          <span style={styles.metaLabel}>Tx Explorer:</span>{" "}
          <a href={txLink} target="_blank" rel="noopener noreferrer" style={styles.explorerLink}>
            Polygonscan ↗
          </a>
        </div>
      </div>

      {/* Expandable History & Audit Panel */}
      {isExpanded && (
        <div style={styles.expandedPanel}>
          {/* Sub-Tabs */}
          <div style={styles.tabBar}>
            <button
              onClick={() => setActiveTab("timeline")}
              style={{
                ...styles.tabItem,
                ...(activeTab === "timeline" ? styles.tabItemActive : {})
              }}
            >
              Version History Chain ({history.length})
            </button>
            <button
              onClick={() => setActiveTab("verify")}
              style={{
                ...styles.tabItem,
                ...(activeTab === "verify" ? styles.tabItemActive : {})
              }}
            >
              Verify Tile File
            </button>
          </div>

          {/* Timeline View */}
          {activeTab === "timeline" && (
            <div style={styles.timelineContainer}>
              {loading && <p style={{ color: "#9ca3af", fontSize: "0.85rem" }}>Loading on-chain records...</p>}

              {!loading && history.length === 0 && (
                <p style={{ color: "#9ca3af", fontSize: "0.85rem" }}>No previous versions registered for this tile.</p>
              )}

              {history.map((ver, idx) => (
                <div key={idx} style={styles.timelineNode}>
                  <div style={styles.nodeIndicator}>
                    <div style={styles.nodeCircle}>{ver.version}</div>
                    {idx < history.length - 1 && <div style={styles.nodeConnector}></div>}
                  </div>

                  <div style={styles.nodeContent}>
                    <div style={styles.nodeHeader}>
                      <span style={styles.nodeVersionTag}>{ver.version_tag}</span>
                      <span style={styles.nodeModel}>{ver.model_name} ({ver.scale_factor}x)</span>
                      <span style={styles.nodeTime}>{ver.timestamp_iso}</span>
                    </div>

                    <div style={styles.nodeHashes}>
                      <div style={styles.hashRow}>
                        <span style={styles.hashLabel}>Current Hash:</span>
                        <code style={styles.hashValue}>{ver.image_hash_short}</code>
                      </div>
                      <div style={styles.hashRow}>
                        <span style={styles.hashLabel}>Linked Previous Hash:</span>
                        <code style={{ ...styles.hashValue, color: ver.is_first_version ? "#6b7280" : "#a5f3fc" }}>
                          {ver.is_first_version ? "0x00000000... (Root v1)" : ver.previous_hash_short}
                        </code>
                      </div>
                    </div>

                    <div style={styles.nodeFooter}>
                      <span>Submitter: <code style={styles.submitterCode}>{ver.submitter_short}</code></span>
                      <a href={ver.tx_link} target="_blank" rel="noopener noreferrer" style={styles.smallTxLink}>
                        Verify Block ↗
                      </a>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Verification View */}
          {activeTab === "verify" && (
            <div style={styles.verifyContainer}>
              <p style={{ fontSize: "0.8rem", color: "#9ca3af", marginBottom: "10px" }}>
                Upload any exported GeoTIFF or PNG to verify its SHA-256 against Polygon Amoy on-chain provenance.
              </p>

              <input
                type="file"
                accept=".tif,.tiff,.png,.jpg"
                onChange={handleFileUpload}
                style={styles.fileInput}
              />

              {verifyStatus && (
                <div
                  style={{
                    ...styles.verifyAlert,
                    borderColor:
                      verifyStatus.state === "verified"
                        ? "#10b981"
                        : verifyStatus.state === "tampered"
                        ? "#f43f5e"
                        : "#FFD700",
                    background:
                      verifyStatus.state === "verified"
                        ? "rgba(16, 185, 129, 0.1)"
                        : verifyStatus.state === "tampered"
                        ? "rgba(244, 63, 94, 0.1)"
                        : "rgba(255, 215, 0, 0.1)"
                  }}
                >
                  <strong style={{ display: "block", marginBottom: "4px" }}>
                    {verifyStatus.state === "verified" && "✓ On-Chain Hash Verified"}
                    {verifyStatus.state === "tampered" && "⚠ Cryptographic Mismatch"}
                    {verifyStatus.state === "checking" && "⟳ Verifying..."}
                  </strong>
                  <span>{verifyStatus.message}</span>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

const styles = {
  card: {
    background: "rgba(10, 25, 47, 0.65)",
    backdropFilter: "blur(16px)",
    border: "1px solid rgba(255, 215, 0, 0.3)",
    borderRadius: "12px",
    padding: "1rem 1.25rem",
    boxShadow: "0 8px 32px rgba(0, 0, 0, 0.4)",
    fontFamily: "'Inter', -apple-system, sans-serif",
    color: "#f3f4f6",
    margin: "0.75rem 0"
  },
  header: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    flexWrap: "wrap",
    gap: "0.75rem"
  },
  badgeGroup: {
    display: "flex",
    alignItems: "center",
    gap: "0.75rem",
    flexWrap: "wrap"
  },
  chainBadge: {
    display: "flex",
    alignItems: "center",
    gap: "6px",
    background: "rgba(130, 71, 229, 0.15)",
    border: "1px solid rgba(130, 71, 229, 0.4)",
    padding: "4px 10px",
    borderRadius: "20px",
    fontSize: "0.75rem",
    color: "#c084fc",
    fontWeight: "600"
  },
  chainDot: {
    width: "7px",
    height: "7px",
    borderRadius: "50%",
    background: "#a855f7",
    boxShadow: "0 0 8px #a855f7"
  },
  chainText: {
    letterSpacing: "0.02em"
  },
  versionBadge: {
    display: "flex",
    alignItems: "center",
    background: "rgba(255, 215, 0, 0.08)",
    border: "1px solid rgba(255, 215, 0, 0.25)",
    padding: "4px 12px",
    borderRadius: "20px",
    fontSize: "0.78rem"
  },
  toggleButton: {
    background: "transparent",
    border: "1px solid rgba(255, 255, 255, 0.15)",
    color: "#38bdf8",
    padding: "5px 12px",
    borderRadius: "6px",
    fontSize: "0.75rem",
    cursor: "pointer",
    transition: "all 0.2s ease"
  },
  telemetryStrip: {
    display: "flex",
    alignItems: "center",
    gap: "1.5rem",
    marginTop: "0.75rem",
    paddingTop: "0.65rem",
    borderTop: "1px solid rgba(255, 255, 255, 0.08)",
    fontSize: "0.75rem",
    flexWrap: "wrap"
  },
  metaLabel: {
    color: "#9ca3af"
  },
  metaVal: {
    color: "#f3f4f6",
    fontWeight: "500"
  },
  metaCode: {
    fontFamily: "monospace",
    color: "#facc15",
    background: "rgba(250, 204, 21, 0.1)",
    padding: "2px 6px",
    borderRadius: "4px"
  },
  explorerLink: {
    color: "#38bdf8",
    textDecoration: "none",
    fontWeight: "500"
  },
  expandedPanel: {
    marginTop: "1rem",
    paddingTop: "0.75rem",
    borderTop: "1px solid rgba(255, 215, 0, 0.2)"
  },
  tabBar: {
    display: "flex",
    gap: "0.5rem",
    marginBottom: "1rem"
  },
  tabItem: {
    background: "rgba(255, 255, 255, 0.05)",
    border: "none",
    color: "#9ca3af",
    padding: "6px 14px",
    borderRadius: "6px",
    fontSize: "0.75rem",
    cursor: "pointer"
  },
  tabItemActive: {
    background: "rgba(255, 215, 0, 0.15)",
    color: "#FFD700",
    border: "1px solid rgba(255, 215, 0, 0.3)"
  },
  timelineContainer: {
    display: "flex",
    flexDirection: "column",
    gap: "1rem"
  },
  timelineNode: {
    display: "flex",
    gap: "1rem"
  },
  nodeIndicator: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    width: "28px"
  },
  nodeCircle: {
    width: "28px",
    height: "28px",
    borderRadius: "50%",
    background: "rgba(255, 215, 0, 0.15)",
    border: "2px solid #FFD700",
    color: "#FFD700",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontWeight: "700",
    fontSize: "0.75rem"
  },
  nodeConnector: {
    width: "2px",
    flex: 1,
    background: "rgba(255, 215, 0, 0.3)",
    margin: "4px 0"
  },
  nodeContent: {
    flex: 1,
    background: "rgba(255, 255, 255, 0.02)",
    border: "1px solid rgba(255, 255, 255, 0.06)",
    borderRadius: "8px",
    padding: "0.75rem 1rem"
  },
  nodeHeader: {
    display: "flex",
    alignItems: "center",
    gap: "0.75rem",
    marginBottom: "0.4rem",
    fontSize: "0.75rem"
  },
  nodeVersionTag: {
    fontWeight: "700",
    color: "#FFD700"
  },
  nodeModel: {
    color: "#cbd5e1"
  },
  nodeTime: {
    color: "#6b7280",
    marginLeft: "auto"
  },
  nodeHashes: {
    display: "flex",
    flexDirection: "column",
    gap: "3px",
    fontSize: "0.72rem",
    margin: "0.4rem 0"
  },
  hashRow: {
    display: "flex",
    alignItems: "center",
    gap: "0.5rem"
  },
  hashLabel: {
    color: "#9ca3af",
    minWidth: "130px"
  },
  hashValue: {
    fontFamily: "monospace",
    color: "#34d399"
  },
  nodeFooter: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginTop: "0.4rem",
    fontSize: "0.7rem",
    color: "#9ca3af"
  },
  submitterCode: {
    fontFamily: "monospace",
    color: "#a5f3fc"
  },
  smallTxLink: {
    color: "#38bdf8",
    textDecoration: "none"
  },
  verifyContainer: {
    padding: "0.5rem 0"
  },
  fileInput: {
    fontSize: "0.8rem",
    color: "#9ca3af",
    marginBottom: "0.75rem"
  },
  verifyAlert: {
    border: "1px solid",
    borderRadius: "8px",
    padding: "0.75rem 1rem",
    fontSize: "0.8rem",
    marginTop: "0.5rem"
  }
};

