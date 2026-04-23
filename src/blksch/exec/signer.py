"""EIP-712 signer for Polymarket CTF Exchange orders (fallback path).

This module is only used when :mod:`blksch.exec.clob_client.HttpCLOBClient`
is active (i.e. ``py-clob-client`` is unavailable or opted out). The primary
path piggybacks on py-clob-client's signer.

Polymarket's CTF Exchange order EIP-712 schema (v1, Polygon mainnet):

    domain = {
        name: "Polymarket CTF Exchange",
        version: "1",
        chainId: 137,
        verifyingContract: 0x4bFb41...,
    }
    Order = {
        salt: uint256,
        maker: address,
        signer: address,
        taker: address,
        tokenId: uint256,
        makerAmount: uint256,
        takerAmount: uint256,
        expiration: uint256,
        nonce: uint256,
        feeRateBps: uint256,
        side: uint8,          # 0=BUY, 1=SELL
        signatureType: uint8, # 0=EOA, 1=POLY_PROXY, 2=POLY_GNOSIS_SAFE
    }

The contract addresses and chain ids can change; don't hard-code beyond what's
defaulted here — pull from config when promoting to live.
"""

from __future__ import annotations

import secrets
from dataclasses import asdict, dataclass, field
from typing import Literal

from eth_account import Account
from eth_account.messages import encode_typed_data

# Polymarket CTF Exchange deployed addresses. Neg-risk exchange differs — see
# https://docs.polymarket.com/. Kept here as defaults; override via config
# (``CLOBConfig.verifying_contract`` or ``POLY_VERIFYING_CONTRACT`` env var).
POLY_CTF_EXCHANGE_MAINNET = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"
POLY_NEG_RISK_EXCHANGE_MAINNET = "0xC5d563A36AE78145C45a50134d48A1215220f80a"

# Polygon Amoy (test network; chainId 80002). Polymarket has not publicly
# committed to a stable testnet CTF Exchange deployment, so this is treated
# as an operator-supplied value — testnet runs MUST set
# ``POLY_VERIFYING_CONTRACT`` (or ``CLOBConfig.verifying_contract``) to the
# currently-live address. The zero-address default below is a deliberate
# "fail-closed" signal: any signed typed-data built against it will mint a
# recoverable-but-rejected signature, so the canary breaks loudly instead of
# silently posting a malformed order.
POLY_CTF_EXCHANGE_AMOY = "0x0000000000000000000000000000000000000000"


SIG_EOA = 0
SIG_POLY_PROXY = 1
SIG_POLY_GNOSIS_SAFE = 2


@dataclass(frozen=True)
class PolymarketOrder:
    """The on-chain order struct signed via EIP-712."""

    salt: int
    maker: str
    signer: str
    taker: str
    token_id: int
    maker_amount: int
    taker_amount: int
    expiration: int
    nonce: int
    fee_rate_bps: int
    side: Literal[0, 1]
    signature_type: Literal[0, 1, 2] = SIG_POLY_PROXY
    signature: str = field(default="", compare=False)

    def to_typed_message(self, *, chain_id: int, verifying_contract: str) -> dict:
        """Build the EIP-712 payload ready to pass to ``eth_account``."""

        return {
            "types": {
                "EIP712Domain": [
                    {"name": "name", "type": "string"},
                    {"name": "version", "type": "string"},
                    {"name": "chainId", "type": "uint256"},
                    {"name": "verifyingContract", "type": "address"},
                ],
                "Order": [
                    {"name": "salt", "type": "uint256"},
                    {"name": "maker", "type": "address"},
                    {"name": "signer", "type": "address"},
                    {"name": "taker", "type": "address"},
                    {"name": "tokenId", "type": "uint256"},
                    {"name": "makerAmount", "type": "uint256"},
                    {"name": "takerAmount", "type": "uint256"},
                    {"name": "expiration", "type": "uint256"},
                    {"name": "nonce", "type": "uint256"},
                    {"name": "feeRateBps", "type": "uint256"},
                    {"name": "side", "type": "uint8"},
                    {"name": "signatureType", "type": "uint8"},
                ],
            },
            "primaryType": "Order",
            "domain": {
                "name": "Polymarket CTF Exchange",
                "version": "1",
                "chainId": chain_id,
                "verifyingContract": verifying_contract,
            },
            "message": {
                "salt": self.salt,
                "maker": self.maker,
                "signer": self.signer,
                "taker": self.taker,
                "tokenId": self.token_id,
                "makerAmount": self.maker_amount,
                "takerAmount": self.taker_amount,
                "expiration": self.expiration,
                "nonce": self.nonce,
                "feeRateBps": self.fee_rate_bps,
                "side": self.side,
                "signatureType": self.signature_type,
            },
        }


def generate_salt() -> int:
    """256-bit random salt — Polymarket enforces uniqueness per-maker."""
    return secrets.randbits(256)


def sign_order(
    order: PolymarketOrder,
    private_key: str,
    *,
    chain_id: int = 137,
    verifying_contract: str = POLY_CTF_EXCHANGE_MAINNET,
) -> PolymarketOrder:
    """Return a copy of the order with the EIP-712 signature attached."""

    msg = order.to_typed_message(chain_id=chain_id, verifying_contract=verifying_contract)
    encoded = encode_typed_data(full_message=msg)
    signed = Account.from_key(private_key).sign_message(encoded)
    sig_hex = signed.signature.hex()
    if not sig_hex.startswith("0x"):
        sig_hex = "0x" + sig_hex
    # replace signature field (dataclass is frozen)
    data = asdict(order)
    data["signature"] = sig_hex
    return PolymarketOrder(**data)


def order_hash(
    order: PolymarketOrder,
    *,
    chain_id: int = 137,
    verifying_contract: str = POLY_CTF_EXCHANGE_MAINNET,
) -> bytes:
    """Return the EIP-712 digest that will be signed — useful for tests."""
    msg = order.to_typed_message(chain_id=chain_id, verifying_contract=verifying_contract)
    encoded = encode_typed_data(full_message=msg)
    return encoded.body  # keccak256 hash that eth_account signs
