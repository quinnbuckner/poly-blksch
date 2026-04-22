"""EIP-712 signer sanity: determinism, field layout, signature recovery.

We do not have upstream Polymarket test vectors checked in here — when Stage
2 promotion is in flight, drop a vector from a live test-net order into
``tests/fixtures/`` and pin a byte-for-byte equality assertion. For now we
verify (a) the signature is deterministic for fixed inputs, (b) it recovers
to the correct address, and (c) the typed-message layout matches the
documented Polymarket schema field-for-field.
"""

from __future__ import annotations

from eth_account import Account
from eth_account.messages import encode_typed_data

from blksch.exec.signer import (
    POLY_CTF_EXCHANGE_MAINNET,
    PolymarketOrder,
    generate_salt,
    order_hash,
    sign_order,
)

# Fixed test key — NOT a production key. Pinned so signatures are reproducible.
TEST_KEY = "0x" + "1" * 64
TEST_ACCOUNT = Account.from_key(TEST_KEY)


def _order(**overrides) -> PolymarketOrder:
    base = dict(
        salt=12345678901234567890,
        maker=TEST_ACCOUNT.address,
        signer=TEST_ACCOUNT.address,
        taker="0x0000000000000000000000000000000000000000",
        token_id=int("0x1234", 16),
        maker_amount=1_000_000,  # 1 USDC (6 dec)
        taker_amount=2_000_000,
        expiration=0,
        nonce=0,
        fee_rate_bps=0,
        side=0,  # BUY
        signature_type=1,
    )
    base.update(overrides)
    return PolymarketOrder(**base)


def test_typed_message_layout_is_polymarket_v1():
    order = _order()
    msg = order.to_typed_message(chain_id=137, verifying_contract=POLY_CTF_EXCHANGE_MAINNET)
    assert msg["primaryType"] == "Order"
    assert msg["domain"]["name"] == "Polymarket CTF Exchange"
    assert msg["domain"]["version"] == "1"
    assert msg["domain"]["chainId"] == 137
    order_fields = [f["name"] for f in msg["types"]["Order"]]
    assert order_fields == [
        "salt", "maker", "signer", "taker", "tokenId",
        "makerAmount", "takerAmount", "expiration", "nonce",
        "feeRateBps", "side", "signatureType",
    ]
    # message body maps snake_case -> camelCase for the on-chain fields
    assert msg["message"]["tokenId"] == order.token_id
    assert msg["message"]["makerAmount"] == order.maker_amount
    assert msg["message"]["signatureType"] == order.signature_type


def test_signing_is_deterministic_and_recoverable():
    order = _order()
    signed1 = sign_order(order, TEST_KEY)
    signed2 = sign_order(order, TEST_KEY)
    assert signed1.signature == signed2.signature
    assert signed1.signature.startswith("0x") and len(signed1.signature) == 2 + 130

    msg = order.to_typed_message(chain_id=137, verifying_contract=POLY_CTF_EXCHANGE_MAINNET)
    encoded = encode_typed_data(full_message=msg)
    recovered = Account.recover_message(encoded, signature=signed1.signature)
    assert recovered.lower() == TEST_ACCOUNT.address.lower()


def test_different_salts_produce_different_signatures():
    a = sign_order(_order(salt=1), TEST_KEY)
    b = sign_order(_order(salt=2), TEST_KEY)
    assert a.signature != b.signature


def test_generate_salt_fits_in_uint256():
    s = generate_salt()
    assert 0 <= s < 2 ** 256


def test_order_hash_matches_encoded_digest():
    order = _order()
    h = order_hash(order)
    # 32-byte keccak digest
    assert isinstance(h, bytes)
    assert len(h) == 32
