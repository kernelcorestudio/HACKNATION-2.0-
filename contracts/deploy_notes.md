# Polygon Amoy Testnet Deployment Guide

This guide describes how to deploy `TileProvenance.sol` to **Polygon Amoy testnet** (Chain ID: `80002`), claim test POL, and link the deployed contract to NETRA.

---

## 1. Network Details

| Parameter | Value |
| :--- | :--- |
| **Network Name** | Polygon Amoy Testnet |
| **RPC URL** | `https://rpc-amoy.polygon.technology` (or Alchemy / QuickNode) |
| **Chain ID** | `80002` |
| **Currency Symbol** | `POL` |
| **Block Explorer** | `https://amoy.polygonscan.com/` |

---

## 2. Claim Free Test POL (Faucet)

1. Open [Polygon Faucet](https://faucet.polygon.technology/) or [Alchemy Amoy Faucet](https://www.alchemy.com/faucets/polygon-amoy) / [QuickNode Faucet](https://faucet.quicknode.com/polygon/amoy).
2. Select **Polygon Amoy Testnet**.
3. Paste your MetaMask / Web3 wallet address.
4. Claim 0.5 to 1.0 POL (typically arrives in 10-30 seconds).

---

## 3. Deploy via Remix IDE (Recommended, Free & Fast)

1. Open [Remix IDE](https://remix.ethereum.org/).
2. Create a new file in `contracts/` named `TileProvenance.sol` and paste the contents of `contracts/TileProvenance.sol`.
3. In the **Solidity Compiler** tab:
   - Select compiler version `0.8.20` (or `0.8.x`).
   - Click **Compile TileProvenance.sol**.
4. In the **Deploy & Run Transactions** tab:
   - Environment: Select **Injected Provider - MetaMask**.
   - Ensure MetaMask is switched to **Polygon Amoy** (Chain ID `80002`).
   - Contract: Select `TileProvenance`.
   - Click **Deploy** and confirm the transaction in MetaMask.
5. Copy the deployed contract address (e.g. `0x1234...abcd`).

---

## 4. Link Contract to NETRA

Add the deployed contract address and RPC settings to your `.env` file:

```env
# Polygon Amoy Testnet Configuration
POLYGON_AMOY_RPC=https://rpc-amoy.polygon.technology
CONTRACT_ADDRESS=0xYourDeployedContractAddressHere
POLYGON_PRIVATE_KEY=your_wallet_private_key_without_0x
POLYGONSCAN_BASE_URL=https://amoy.polygonscan.com
```

> [!NOTE]
> When `POLYGON_PRIVATE_KEY` and `CONTRACT_ADDRESS` are configured, NETRA submits live on-chain transactions to Polygon Amoy on each super-resolution execution. If omitted, NETRA automatically runs in **Simulation Mode** using a persistent cryptographic JSON ledger (`cache/blockchain_ledger.json`), maintaining exact version chaining (v1 -> v2 -> v3) and SHA-256 integrity checks.
