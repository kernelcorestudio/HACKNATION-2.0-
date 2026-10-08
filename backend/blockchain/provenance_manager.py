"""
NETRA - Blockchain Provenance Manager
Team KC Studio

Handles:
1. Geo-Tagging: Scaling lat/lon centre & bounding box (real * 1e6) and binding to SHA-256 hash.
2. Version History Chain: Chaining re-processed tile outputs into immutable records (v1 -> v2 -> v3).
3. Web3 Polygon Amoy Client with seamless persistent JSON fallback for offline/demo reliability.
"""
from __future__ import annotations
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Windows App Control compatibility shim for Web3 / eth-account
import sys
import unittest.mock
if "ckzg" not in sys.modules:
    sys.modules["ckzg"] = unittest.mock.MagicMock()

from backend.config import (
    BASE_DIR,
    POLYGON_AMOY_RPC,
    POLYGON_CHAIN_ID,
    CONTRACT_ADDRESS,
    POLYGON_PRIVATE_KEY,
    POLYGONSCAN_BASE_URL,
    CONTRACT_ABI_PATH,
    BLOCKCHAIN_LEDGER_PATH
)

# Scaling constant: 1e6 preserves ~11cm precision at the equator
COORD_SCALE: float = 1_000_000.0


def scale_coordinate(val: float) -> int:
    """Scales float latitude/longitude to int256 compatible integer (* 1e6)."""
    return int(round(float(val) * COORD_SCALE))


def unscale_coordinate(val: int) -> float:
    """Restores scaled int coordinate back to real float value."""
    return round(float(val) / COORD_SCALE, 6)


def compute_sha256(data: Union[bytes, np.ndarray, Path, str]) -> str:
    """
    Computes deterministic SHA-256 hash of arbitrary satellite tile data.
    Returns 0x-prefixed 64-character hex string (bytes32 compatible).
    """
    h = hashlib.sha256()

    if isinstance(data, (str, Path)):
        path = Path(data)
        if path.exists() and path.is_file():
            with open(path, "rb") as f:
                while chunk := f.read(65536):
                    h.update(chunk)
            return "0x" + h.hexdigest()
        else:
            # Hash string as utf-8
            h.update(str(data).encode("utf-8"))
            return "0x" + h.hexdigest()

    elif isinstance(data, np.ndarray):
        # Deterministic hashing of array contents
        contiguous = np.ascontiguousarray(data.astype(np.float32))
        h.update(contiguous.tobytes())
        return "0x" + h.hexdigest()

    elif isinstance(data, bytes):
        h.update(data)
        return "0x" + h.hexdigest()

    else:
        h.update(str(data).encode("utf-8"))
        return "0x" + h.hexdigest()


class ProvenanceManager:
    """
    Singleton manager for tile geo-tagging and version history chaining on Polygon Amoy.
    """
    _instance: Optional[ProvenanceManager] = None

    def __init__(self):
        self.rpc_url = POLYGON_AMOY_RPC
        self.chain_id = POLYGON_CHAIN_ID
        self.contract_address = CONTRACT_ADDRESS
        self.private_key = POLYGON_PRIVATE_KEY
        self.explorer_url = POLYGONSCAN_BASE_URL.rstrip("/")
        self.abi_path = CONTRACT_ABI_PATH
        self.ledger_path = BLOCKCHAIN_LEDGER_PATH

        self._web3 = None
        self._contract = None
        self._account = None
        self._is_onchain_ready = False

        # Ensure ledger directory exists
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_web3()

    @classmethod
    def get_instance(cls) -> ProvenanceManager:
        if cls._instance is None:
            cls._instance = ProvenanceManager()
        return cls._instance

    def _init_web3(self):
        """Attempts to connect to Polygon Amoy if web3 is installed and credentials exist."""
        try:
            from web3 import Web3
            from eth_account import Account

            if self.rpc_url:
                w3 = Web3(Web3.HTTPProvider(self.rpc_url))
                if w3.is_connected():
                    self._web3 = w3
                    print(f"[NETRA-Blockchain] Connected to Polygon Amoy RPC: {self.rpc_url}")

                    if self.private_key:
                        pk = self.private_key if not self.private_key.startswith("0x") else self.private_key[2:]
                        self._account = Account.from_key("0x" + pk)
                        print(f"[NETRA-Blockchain] Submitter wallet active: {self._account.address}")

                    if self.contract_address and self.abi_path.exists():
                        with open(self.abi_path, "r", encoding="utf-8") as f:
                            abi = json.load(f)
                        self._contract = w3.eth.contract(
                            address=Web3.to_checksum_address(self.contract_address),
                            abi=abi
                        )
                        self._is_onchain_ready = bool(self._account and self._contract)
                        print(f"[NETRA-Blockchain] TileProvenance contract loaded at: {self.contract_address}")
                else:
                    print(f"[NETRA-Blockchain] RPC {self.rpc_url} unreachable. Using persistent ledger simulation.")
        except ImportError:
            print("[NETRA-Blockchain] web3.py not installed in active environment. Using persistent ledger simulation.")
        except Exception as e:
            print(f"[NETRA-Blockchain] Web3 initialization warning: {e}. Using persistent ledger simulation.")

    def _load_ledger(self) -> Dict[str, List[Dict[str, Any]]]:
        """Loads local JSON ledger cache."""
        if self.ledger_path.exists():
            try:
                with open(self.ledger_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[NETRA-Blockchain] Error reading ledger: {e}")
        return {}

    def _save_ledger(self, ledger: Dict[str, List[Dict[str, Any]]]):
        """Persists local JSON ledger cache."""
        try:
            with open(self.ledger_path, "w", encoding="utf-8") as f:
                json.dump(ledger, f, indent=2)
        except Exception as e:
            print(f"[NETRA-Blockchain] Error saving ledger: {e}")

    def register_tile(
        self,
        tile_id: str,
        image_data: Union[bytes, np.ndarray, Path, str],
        bbox: Union[List[float], Tuple[float, float, float, float]],
        model_name: str = "HAT",
        scale_factor: int = 4,
        extra_meta: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Registers a processed tile on-chain or in persistent ledger:
        1. Computes SHA-256 hash of output.
        2. Computes and scales centre latitude/longitude and bbox bounds (* 1e6).
        3. Retrieves prior record to set previousHash and versionNumber (v1, v2, v3).
        4. Broadcasts to Polygon Amoy contract if available, otherwise commits to ledger.
        """
        clean_tile_id = str(tile_id).strip() or "hacknation_tile_001"
        image_hash = compute_sha256(image_data)

        # Parse Bounding Box: [min_lon, min_lat, max_lon, max_lat] or [west, south, east, north]
        min_lon, min_lat, max_lon, max_lat = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        lat_center = (min_lat + max_lat) / 2.0
        lon_center = (min_lon + max_lon) / 2.0

        # Integer scaled coordinates (sub-11cm precision)
        lat_center_int = scale_coordinate(lat_center)
        lon_center_int = scale_coordinate(lon_center)
        bbox_min_lat_int = scale_coordinate(min_lat)
        bbox_min_lon_int = scale_coordinate(min_lon)
        bbox_max_lat_int = scale_coordinate(max_lat)
        bbox_max_lon_int = scale_coordinate(max_lon)

        current_timestamp = int(time.time())

        # Check existing version history
        ledger = self._load_ledger()
        history = ledger.get(clean_tile_id, [])
        prior_count = len(history)

        if prior_count > 0:
            previous_hash = history[-1]["imageHash"]
            version_number = prior_count + 1
        else:
            previous_hash = "0x" + "00" * 32
            version_number = 1

        tx_hash: str = ""
        submitter_addr: str = self._account.address if self._account else "0x71C850...NETRA_Node"
        mode: str = "simulation"

        # Attempt on-chain broadcast if configured
        if self._is_onchain_ready and self._contract and self._web3 and self._account:
            try:
                from web3 import Web3
                image_hash_bytes = bytes.fromhex(image_hash[2:])
                
                nonce = self._web3.eth.get_transaction_count(self._account.address)
                tx = self._contract.functions.registerTile(
                    clean_tile_id,
                    image_hash_bytes,
                    lat_center_int,
                    lon_center_int,
                    bbox_min_lat_int,
                    bbox_min_lon_int,
                    bbox_max_lat_int,
                    bbox_max_lon_int
                ).build_transaction({
                    "chainId": self.chain_id,
                    "gas": 300000,
                    "maxFeePerGas": self._web3.to_wei("35", "gwei"),
                    "maxPriorityFeePerGas": self._web3.to_wei("30", "gwei"),
                    "nonce": nonce,
                    "from": self._account.address
                })

                signed_tx = self._web3.eth.account.sign_transaction(tx, private_key=self.private_key)
                raw_tx_bytes = getattr(signed_tx, "raw_transaction", None) or getattr(signed_tx, "rawTransaction", None)
                tx_raw_hash = self._web3.eth.send_raw_transaction(raw_tx_bytes)
                tx_hash = "0x" + tx_raw_hash.hex()
                mode = "polygon_amoy"
                print(f"[NETRA-Blockchain] On-chain tx sent to Polygon Amoy: {tx_hash}")
            except Exception as e:
                print(f"[NETRA-Blockchain] On-chain tx failed ({e}). Falling back to local simulation.")

        # Fallback simulation tx hash
        if not tx_hash:
            seed = f"{clean_tile_id}_{version_number}_{image_hash}_{current_timestamp}"
            tx_hash = "0x" + hashlib.sha256(seed.encode("utf-8")).hexdigest()

        tx_link = f"{self.explorer_url}/tx/{tx_hash}"

        # Construct TileRecord struct
        record: Dict[str, Any] = {
            "tileId": clean_tile_id,
            "versionNumber": version_number,
            "imageHash": image_hash,
            "previousHash": previous_hash,
            "latCenter": lat_center_int,
            "lonCenter": lon_center_int,
            "bboxMinLat": bbox_min_lat_int,
            "bboxMinLon": bbox_min_lon_int,
            "bboxMaxLat": bbox_max_lat_int,
            "bboxMaxLon": bbox_max_lon_int,
            "processedTimestamp": current_timestamp,
            "submitter": submitter_addr,
            "txHash": tx_hash,
            "txLink": tx_link,
            "mode": mode,
            "modelName": model_name,
            "scaleFactor": scale_factor,
            "realCoordinates": {
                "latCenter": lat_center,
                "lonCenter": lon_center,
                "bbox": [min_lon, min_lat, max_lon, max_lat]
            },
            "extra": extra_meta or {}
        }

        # Update and persist ledger
        history.append(record)
        ledger[clean_tile_id] = history
        self._save_ledger(ledger)

        print(f"[NETRA-Blockchain] Tile '{clean_tile_id}' v{version_number} registered. Hash: {image_hash[:10]}... Prev: {previous_hash[:10]}...")

        return record

    def verify_tile(self, tile_id: str, provided_hash_or_data: Union[str, bytes, np.ndarray, Path]) -> Dict[str, Any]:
        """
        Verifies whether a provided image/hash matches the latest on-chain record for a tile.
        """
        clean_tile_id = str(tile_id).strip()
        ledger = self._load_ledger()
        history = ledger.get(clean_tile_id, [])

        if not history:
            return {
                "verified": False,
                "status": "NOT_FOUND",
                "message": f"No blockchain record exists for tile '{clean_tile_id}'.",
                "tile_id": clean_tile_id
            }

        latest = history[-1]

        # Determine target hash
        if isinstance(provided_hash_or_data, str) and provided_hash_or_data.startswith("0x") and len(provided_hash_or_data) == 66:
            provided_hash = provided_hash_or_data.lower()
        else:
            provided_hash = compute_sha256(provided_hash_or_data).lower()

        matched = (provided_hash == latest["imageHash"].lower())

        return {
            "verified": matched,
            "status": "MATCH" if matched else "TAMPER_DETECTED",
            "tile_id": clean_tile_id,
            "version_number": latest["versionNumber"],
            "provided_hash": provided_hash,
            "registered_hash": latest["imageHash"],
            "previous_hash": latest["previousHash"],
            "lat": unscale_coordinate(latest["latCenter"]),
            "lon": unscale_coordinate(latest["lonCenter"]),
            "bbox": [
                unscale_coordinate(latest["bboxMinLon"]),
                unscale_coordinate(latest["bboxMinLat"]),
                unscale_coordinate(latest["bboxMaxLon"]),
                unscale_coordinate(latest["bboxMaxLat"])
            ],
            "timestamp": latest["processedTimestamp"],
            "submitter": latest["submitter"],
            "tx_hash": latest["txHash"],
            "tx_link": latest["txLink"],
            "mode": latest.get("mode", "simulation"),
            "model_name": latest.get("modelName", "SRM")
        }

    def get_history(self, tile_id: str) -> List[Dict[str, Any]]:
        """
        Returns full chronological version history chain for audit/rollback review.
        """
        clean_tile_id = str(tile_id).strip()
        ledger = self._load_ledger()
        history = ledger.get(clean_tile_id, [])

        # Format for clean API response
        formatted_history = []
        for rec in history:
            formatted_history.append({
                "version": rec["versionNumber"],
                "version_tag": f"v{rec['versionNumber']}",
                "image_hash": rec["imageHash"],
                "image_hash_short": f"{rec['imageHash'][:10]}...{rec['imageHash'][-6:]}",
                "previous_hash": rec["previousHash"],
                "previous_hash_short": f"{rec['previousHash'][:10]}...{rec['previousHash'][-6:]}",
                "is_first_version": rec["previousHash"] == ("0x" + "00" * 32),
                "timestamp": rec["processedTimestamp"],
                "timestamp_iso": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(rec["processedTimestamp"])),
                "submitter": rec["submitter"],
                "submitter_short": f"{rec['submitter'][:6]}...{rec['submitter'][-4:]}",
                "lat": unscale_coordinate(rec["latCenter"]),
                "lon": unscale_coordinate(rec["lonCenter"]),
                "bbox": [
                    unscale_coordinate(rec["bboxMinLon"]),
                    unscale_coordinate(rec["bboxMinLat"]),
                    unscale_coordinate(rec["bboxMaxLon"]),
                    unscale_coordinate(rec["bboxMaxLat"])
                ],
                "model_name": rec.get("modelName", "SRM"),
                "scale_factor": rec.get("scaleFactor", 4),
                "tx_hash": rec["txHash"],
                "tx_link": rec["txLink"],
                "mode": rec.get("mode", "simulation")
            })

        return formatted_history

    def get_latest(self, tile_id: str) -> Optional[Dict[str, Any]]:
        """Returns the latest registered record for a tile."""
        history = self.get_history(tile_id)
        return history[-1] if history else None


def get_provenance_manager() -> ProvenanceManager:
    """Convenience accessor for singleton instance."""
    return ProvenanceManager.get_instance()
