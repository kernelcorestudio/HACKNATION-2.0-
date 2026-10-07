"""
1-Click Deploy Script for NETRA TileProvenance.sol to Polygon Amoy Testnet (Chain ID: 80002)
Team KC Studio
"""
import sys
import os
import json
import time
import unittest.mock

# Windows DLL AppControl policy compatibility shim for ckzg
if "ckzg" not in sys.modules:
    sys.modules["ckzg"] = unittest.mock.MagicMock()

from pathlib import Path
from web3 import Web3
from eth_account import Account
import solcx
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

RPC_URL = os.getenv("POLYGON_AMOY_RPC", "https://polygon-amoy.drpc.org")
PRIVATE_KEY = os.getenv("POLYGON_PRIVATE_KEY", "")
EXPLORER_URL = os.getenv("POLYGONSCAN_BASE_URL", "https://amoy.polygonscan.com").rstrip("/")
CHAIN_ID = int(os.getenv("POLYGON_CHAIN_ID", "80002"))


def main():
    print("=" * 70)
    print("NETRA - Polygon Amoy Testnet 1-Click Deployment")
    print("=" * 70)

    if not PRIVATE_KEY:
        print("[ERROR] POLYGON_PRIVATE_KEY not found in .env!")
        return

    pk = PRIVATE_KEY if not PRIVATE_KEY.startswith("0x") else PRIVATE_KEY[2:]
    account = Account.from_key("0x" + pk)
    wallet_address = account.address

    print(f"Deployer Wallet Address : {wallet_address}")
    print(f"Polygon Amoy RPC URL    : {RPC_URL}")
    print(f"Chain ID                : {CHAIN_ID}")

    w3 = Web3(Web3.HTTPProvider(RPC_URL))
    if not w3.is_connected():
        print(f"[ERROR] Could not connect to RPC: {RPC_URL}")
        return

    balance_wei = w3.eth.get_balance(wallet_address)
    balance_pol = float(w3.from_wei(balance_wei, "ether"))
    print(f"Wallet Balance          : {balance_pol:.4f} POL")

    if balance_wei == 0:
        print("\n[WARNING] Wallet balance is 0 POL!")
        print(f"Please claim free test POL from the faucet to: {wallet_address}")
        print("Faucet: https://faucet.polygon.technology/ or https://www.alchemy.com/faucets/polygon-amoy")
        print("Run this script again once claimed.")
        return

    print("\n1. Compiling TileProvenance.sol with Solc 0.8.20...")
    contract_file = BASE_DIR / "contracts" / "TileProvenance.sol"
    compiled = solcx.compile_files([str(contract_file)], solc_version="0.8.20")
    contract_interface = (
        compiled.get(f"{contract_file}:TileProvenance")
        or compiled.get("contracts/TileProvenance.sol:TileProvenance")
        or next(iter(compiled.values()))
    )

    abi = contract_interface["abi"]
    bytecode = contract_interface["bin"]

    # Save ABI to contracts/TileProvenance.json
    abi_path = BASE_DIR / "contracts" / "TileProvenance.json"
    with open(abi_path, "w", encoding="utf-8") as f:
        json.dump(abi, f, indent=2)
    print(f"   ABI saved to: {abi_path}")

    print("\n2. Broadcasting Deployment Transaction to Polygon Amoy...")
    TileProvenance = w3.eth.contract(abi=abi, bytecode=bytecode)
    nonce = w3.eth.get_transaction_count(wallet_address)

    # Estimate gas and build deployment tx
    construct_txn = TileProvenance.constructor().build_transaction({
        "from": wallet_address,
        "nonce": nonce,
        "chainId": CHAIN_ID,
        "gas": 1500000,
        "maxFeePerGas": w3.to_wei("35", "gwei"),
        "maxPriorityFeePerGas": w3.to_wei("30", "gwei")
    })

    signed_txn = w3.eth.account.sign_transaction(construct_txn, private_key="0x" + pk)
    raw_bytes = getattr(signed_txn, "raw_transaction", None) or getattr(signed_txn, "rawTransaction", None)
    tx_hash = w3.eth.send_raw_transaction(raw_bytes)
    tx_hex = "0x" + tx_hash.hex()
    print(f"   Transaction Hash: {tx_hex}")
    print(f"   Polygonscan Tx  : {EXPLORER_URL}/tx/{tx_hex}")

    print("\n3. Waiting for block confirmation on Polygon Amoy...")
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
    contract_address = tx_receipt.contractAddress

    print("=" * 70)
    print("SUCCESS! CONTRACT DEPLOYED ON POLYGON AMOY")
    print("=" * 70)
    print(f"Contract Address : {contract_address}")
    print(f"Polygonscan URL  : {EXPLORER_URL}/address/{contract_address}")
    print(f"Block Number     : {tx_receipt.blockNumber}")
    print(f"Gas Used         : {tx_receipt.gasUsed}")

    # Automatically update .env
    env_path = BASE_DIR / ".env"
    if env_path.exists():
        env_content = env_path.read_text(encoding="utf-8")
        if "CONTRACT_ADDRESS=" in env_content:
            import re
            new_env = re.sub(r"CONTRACT_ADDRESS=.*", f"CONTRACT_ADDRESS={contract_address}", env_content)
            env_path.write_text(new_env, encoding="utf-8")
            print(f"\n[OK] Automatically updated CONTRACT_ADDRESS in .env!")

    print("\nYour NETRA project is now permanently linked to Polygonscan!")


if __name__ == "__main__":
    main()
