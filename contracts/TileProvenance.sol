// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title TileProvenance
 * @author Team KC Studio - NETRA Project
 * @notice Permanent on-chain Geo-Tagging and Version History Chaining on Polygon Amoy (Chain ID 80002).
 * 
 * Features:
 * 1. Geo-Tagging: Permanently binds centre coordinates (lat, lon) and bounding box to tile SHA-256 content hash.
 *    Coordinates are scaled integers (real value * 1,000,000) for sub-11cm ground precision.
 * 2. Version History Chain: Every re-processing of a tile links previousHash to the preceding version,
 *    forming an immutable, auditable audit trail (v1 -> v2 -> v3).
 */
contract TileProvenance {
    struct TileRecord {
        bytes32 imageHash;          // SHA-256 content hash of processed GeoTIFF
        int256 latCenter;           // Centre latitude * 1,000,000
        int256 lonCenter;           // Centre longitude * 1,000,000
        int256 bboxMinLat;          // Bounding box min latitude * 1,000,000
        int256 bboxMinLon;          // Bounding box min longitude * 1,000,000
        int256 bboxMaxLat;          // Bounding box max latitude * 1,000,000
        int256 bboxMaxLon;          // Bounding box max longitude * 1,000,000
        uint256 processedTimestamp; // Block timestamp of on-chain confirmation
        bytes32 previousHash;       // Hash of preceding version (0x0 for version 1)
        uint256 versionNumber;      // 1, 2, 3... increments per tileId
        address submitter;          // Address of authorized processing node
    }

    address public owner;
    mapping(string => TileRecord[]) private history;
    mapping(address => bool) public authorized;

    event TileRegistered(
        string indexed tileId,
        uint256 indexed versionNumber,
        bytes32 imageHash,
        bytes32 previousHash,
        uint256 timestamp,
        address submitter
    );

    event SubmitterAuthorized(address indexed submitter);
    event SubmitterRevoked(address indexed submitter);

    modifier onlyOwner() {
        require(msg.sender == owner, "Only contract owner can perform this action");
        _;
    }

    modifier onlyAuthorized() {
        require(authorized[msg.sender] || msg.sender == owner, "Caller is not authorized to register tiles");
        _;
    }

    constructor() {
        owner = msg.sender;
        authorized[msg.sender] = true;
    }

    /**
     * @notice Authorize an address to submit tile provenance records.
     */
    function authorizeSubmitter(address submitter) external onlyOwner {
        authorized[submitter] = true;
        emit SubmitterAuthorized(submitter);
    }

    /**
     * @notice Revoke authorization from an address.
     */
    function revokeSubmitter(address submitter) external onlyOwner {
        authorized[submitter] = false;
        emit SubmitterRevoked(submitter);
    }

    /**
     * @notice Registers a new tile or re-processed version.
     * Automatically derives previousHash from on-chain history and increments versionNumber.
     */
    function registerTile(
        string calldata tileId,
        bytes32 imageHash,
        int256 latCenter,
        int256 lonCenter,
        int256 bboxMinLat,
        int256 bboxMinLon,
        int256 bboxMaxLat,
        int256 bboxMaxLon
    ) external onlyAuthorized returns (uint256 versionNumber, bytes32 previousHash) {
        require(bytes(tileId).length > 0, "Tile ID cannot be empty");
        require(imageHash != bytes32(0), "Image hash cannot be zero");

        uint256 count = history[tileId].length;
        if (count > 0) {
            previousHash = history[tileId][count - 1].imageHash;
            versionNumber = count + 1;
        } else {
            previousHash = bytes32(0);
            versionNumber = 1;
        }

        TileRecord memory newRecord = TileRecord({
            imageHash: imageHash,
            latCenter: latCenter,
            lonCenter: lonCenter,
            bboxMinLat: bboxMinLat,
            bboxMinLon: bboxMinLon,
            bboxMaxLat: bboxMaxLat,
            bboxMaxLon: bboxMaxLon,
            processedTimestamp: block.timestamp,
            previousHash: previousHash,
            versionNumber: versionNumber,
            submitter: msg.sender
        });

        history[tileId].push(newRecord);

        emit TileRegistered(
            tileId,
            versionNumber,
            imageHash,
            previousHash,
            block.timestamp,
            msg.sender
        );

        return (versionNumber, previousHash);
    }

    /**
     * @notice Verifies whether a given SHA-256 hash matches the latest registered version of a tile.
     */
    function verifyTile(string calldata tileId, bytes32 providedHash)
        external
        view
        returns (bool matched, TileRecord memory latest)
    {
        uint256 count = history[tileId].length;
        require(count > 0, "No on-chain records found for this tileId");
        latest = history[tileId][count - 1];
        matched = (latest.imageHash == providedHash);
        return (matched, latest);
    }

    /**
     * @notice Returns the full chronological version history chain for a tile.
     */
    function getHistory(string calldata tileId)
        external
        view
        returns (TileRecord[] memory)
    {
        return history[tileId];
    }

    /**
     * @notice Returns the total number of versions recorded for a tile.
     */
    function getVersionCount(string calldata tileId) external view returns (uint256) {
        return history[tileId].length;
    }

    /**
     * @notice Returns a specific version of a tile (1-indexed).
     */
    function getVersion(string calldata tileId, uint256 versionNumber)
        external
        view
        returns (TileRecord memory)
    {
        require(versionNumber >= 1 && versionNumber <= history[tileId].length, "Version out of range");
        return history[tileId][versionNumber - 1];
    }

    /**
     * @notice Returns the latest version of a tile.
     */
    function getLatest(string calldata tileId) external view returns (TileRecord memory) {
        uint256 count = history[tileId].length;
        require(count > 0, "No records found for this tileId");
        return history[tileId][count - 1];
    }
}
