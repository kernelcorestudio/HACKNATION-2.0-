"""
Configuration settings for HACKNATION Super Resolution Mapping (SRM) System.
"""
from pathlib import Path
import os
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env", override=True)
DATA_DIR = BASE_DIR / "data"
SAMPLE_DIR = BASE_DIR / "sample_data"
OUTPUT_DIR = BASE_DIR / "outputs"
WEIGHTS_DIR = BASE_DIR / "weights"

for folder in [DATA_DIR, SAMPLE_DIR, OUTPUT_DIR, WEIGHTS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Sensor Specs: Sentinel-2 MSI
S2_BANDS = {
    "10m": ["B02", "B03", "B04", "B08"],          # Blue, Green, Red, NIR
    "20m": ["B05", "B06", "B07", "B8A", "B11", "B12"], # RedEdge, SWIR
    "60m": ["B01", "B09", "B10"]                  # Coastal, WaterVapour, Cirrus
}

TARGET_RESOLUTION = 2.5  # meters (from 10m -> 2.5m = 4x upsampling)
DEFAULT_SCALE_FACTOR = 4

# Uncertainty USP Config
MC_DROPOUT_SAMPLES = 8  # Number of forward passes for epistemic uncertainty estimation
CONFIDENCE_THRESHOLD = 0.65  # Default confidence threshold for hallucination masking

# NETRA Blockchain Provenance Settings (Polygon Amoy Testnet, Chain ID 80002)
POLYGON_AMOY_RPC = os.getenv("POLYGON_AMOY_RPC", "https://polygon-amoy.drpc.org")
POLYGON_CHAIN_ID = int(os.getenv("POLYGON_CHAIN_ID", "80002"))
CONTRACT_ADDRESS = os.getenv("CONTRACT_ADDRESS", "")
POLYGON_PRIVATE_KEY = os.getenv("POLYGON_PRIVATE_KEY", "")
POLYGONSCAN_BASE_URL = os.getenv("POLYGONSCAN_BASE_URL", "https://amoy.polygonscan.com")
CONTRACT_ABI_PATH = BASE_DIR / "contracts" / "TileProvenance.json"
BLOCKCHAIN_LEDGER_PATH = BASE_DIR / "cache" / "blockchain_ledger.json"

