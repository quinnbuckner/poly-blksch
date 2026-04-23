"""Contract tests for Polymarket testnet wiring (Polygon Amoy, chainId 80002).

Pins the same offline-replay guarantee the mainnet suite gives us: if the
Amoy response shape drifts, or we mis-plumb chainId / base URL / verifying
contract, one of these tests will fail at the commit that introduces the
drift rather than when we arm the canary.

These tests never hit the network. They:

* Verify :class:`CLOBConfig` picks the right triple (chain_id + base URL +
  verifying_contract) for ``network="testnet"`` and for ``POLY_NETWORK=testnet``
  in the environment.
* Replay recorded Amoy responses through :class:`PyCLOBClient` with
  py-clob-client mocked out — same ``Order`` / ``BookSnap`` Pydantic output
  as mainnet.
* Confirm the EIP-712 signer produces distinct signatures when the domain's
  ``chainId`` / ``verifyingContract`` flip to testnet — so a mis-configured
  canary cannot silently sign a mainnet-valid order against a testnet run.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from blksch.exec.clob_client import (
    POLY_CHAIN_ID_AMOY,
    POLY_CHAIN_ID_MAINNET,
    POLY_CLOB_BASE,
    POLY_CLOB_BASE_TESTNET,
    CLOBConfig,
    PyCLOBClient,
)
from blksch.exec.signer import (
    POLY_CTF_EXCHANGE_AMOY,
    POLY_CTF_EXCHANGE_MAINNET,
    PolymarketOrder,
    sign_order,
)
from blksch.schemas import BookSnap, OrderSide, OrderStatus

pytestmark = pytest.mark.contract


FIXTURES = Path(__file__).parent / "fixtures" / "clob_testnet"
TOKEN_ID = "90210987654321098765432109876543210987654321098765432109876543210987"


def _load(name: str) -> dict:
    with (FIXTURES / name).open() as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Network → config consistency
# ---------------------------------------------------------------------------


def test_for_network_mainnet_defaults():
    c = CLOBConfig.for_network("mainnet")
    assert c.chain_id == POLY_CHAIN_ID_MAINNET
    assert c.base_url == POLY_CLOB_BASE
    assert c.network == "mainnet"
    assert c.resolved_verifying_contract() == POLY_CTF_EXCHANGE_MAINNET


def test_for_network_testnet_defaults():
    c = CLOBConfig.for_network("testnet")
    assert c.chain_id == POLY_CHAIN_ID_AMOY
    assert c.base_url == POLY_CLOB_BASE_TESTNET
    assert c.network == "testnet"
    # Testnet CTF Exchange address is the fail-closed zero default — operator
    # must override to sign anything useful.
    assert c.resolved_verifying_contract() == POLY_CTF_EXCHANGE_AMOY
    assert int(POLY_CTF_EXCHANGE_AMOY, 16) == 0


def test_from_env_reads_poly_network(monkeypatch):
    monkeypatch.setenv("POLY_NETWORK", "testnet")
    c = CLOBConfig.from_env()
    assert c.network == "testnet"
    assert c.chain_id == POLY_CHAIN_ID_AMOY
    assert c.base_url == POLY_CLOB_BASE_TESTNET


def test_from_env_testnet_kwarg_wins_over_env(monkeypatch):
    monkeypatch.setenv("POLY_NETWORK", "mainnet")
    c = CLOBConfig.from_env(testnet=True)
    assert c.network == "testnet"


def test_from_env_base_url_override(monkeypatch):
    monkeypatch.setenv("POLY_NETWORK", "testnet")
    monkeypatch.setenv("POLY_CLOB_BASE", "https://my-own-clob.example.com")
    c = CLOBConfig.from_env()
    assert c.base_url == "https://my-own-clob.example.com"


def test_from_env_verifying_contract_override(monkeypatch):
    monkeypatch.setenv("POLY_NETWORK", "testnet")
    monkeypatch.setenv(
        "POLY_VERIFYING_CONTRACT",
        "0x1234567890123456789012345678901234567890",
    )
    c = CLOBConfig.from_env()
    assert c.resolved_verifying_contract() == "0x1234567890123456789012345678901234567890"


def test_unknown_chain_id_raises():
    c = CLOBConfig(chain_id=1)  # Ethereum mainnet — not supported
    with pytest.raises(ValueError, match="Unsupported chain_id"):
        _ = c.network


# ---------------------------------------------------------------------------
# Signer: chain_id / verifying_contract plumbing
# ---------------------------------------------------------------------------


def _test_order(**overrides) -> PolymarketOrder:
    base = dict(
        salt=42,
        maker="0x0000000000000000000000000000000000000001",
        signer="0x0000000000000000000000000000000000000001",
        taker="0x0000000000000000000000000000000000000000",
        token_id=123,
        maker_amount=1_000_000,
        taker_amount=2_000_000,
        expiration=0,
        nonce=0,
        fee_rate_bps=0,
        side=0,
        signature_type=1,
    )
    base.update(overrides)
    return PolymarketOrder(**base)


def test_signer_message_reflects_chain_id():
    order = _test_order()
    msg_main = order.to_typed_message(
        chain_id=POLY_CHAIN_ID_MAINNET,
        verifying_contract=POLY_CTF_EXCHANGE_MAINNET,
    )
    msg_test = order.to_typed_message(
        chain_id=POLY_CHAIN_ID_AMOY,
        verifying_contract="0x1111111111111111111111111111111111111111",
    )
    assert msg_main["domain"]["chainId"] == POLY_CHAIN_ID_MAINNET
    assert msg_test["domain"]["chainId"] == POLY_CHAIN_ID_AMOY
    assert msg_main["domain"]["verifyingContract"] != msg_test["domain"]["verifyingContract"]


def test_signatures_differ_across_networks():
    """Same order struct, different chain_id+contract, must produce
    different signatures — otherwise a canary on testnet would mint a
    mainnet-valid signature by accident."""
    key = "0x" + "1" * 64
    order = _test_order()
    sig_mainnet = sign_order(
        order, key,
        chain_id=POLY_CHAIN_ID_MAINNET,
        verifying_contract=POLY_CTF_EXCHANGE_MAINNET,
    ).signature
    sig_testnet = sign_order(
        order, key,
        chain_id=POLY_CHAIN_ID_AMOY,
        verifying_contract="0x1111111111111111111111111111111111111111",
    ).signature
    assert sig_mainnet != sig_testnet


# ---------------------------------------------------------------------------
# PyCLOBClient with testnet-shaped fixtures
# ---------------------------------------------------------------------------


def _build_testnet_client() -> PyCLOBClient:
    cfg = CLOBConfig.for_network(
        "testnet",
        private_key="0x" + "2" * 64,
        funder="0x0000000000000000000000000000000000000002",
        api_key="k", api_secret="s", api_passphrase="p",
        verifying_contract="0x1111111111111111111111111111111111111111",
    )
    with patch("py_clob_client.client.ClobClient", MagicMock()):
        client = PyCLOBClient(cfg)
    client._raw = MagicMock()
    client._ensure_l2 = lambda: None
    return client


async def test_testnet_get_book_parses_into_book_snap():
    fixture = _load("get_book.json")
    client = _build_testnet_client()

    def _lvl(d):
        return SimpleNamespace(price=d["price"], size=d["size"])

    client._raw.get_order_book.return_value = SimpleNamespace(
        bids=[_lvl(b) for b in fixture["bids"]],
        asks=[_lvl(a) for a in fixture["asks"]],
    )
    snap = await client.get_book(TOKEN_ID)
    assert isinstance(snap, BookSnap)
    # Same parser, same conventions — bids desc, asks asc.
    assert [lv.price for lv in snap.bids] == [0.42, 0.40]
    assert [lv.price for lv in snap.asks] == [0.58, 0.60]


async def test_testnet_post_order_success_maps_to_open_order():
    fixture = _load("post_order_success.json")
    client = _build_testnet_client()
    client._raw.create_and_post_order.return_value = fixture

    order = await client.place_order(
        token_id=TOKEN_ID, side=OrderSide.BUY, price=0.42, size=5,
        client_id="testnet-canary-1",
    )
    assert order.status is OrderStatus.OPEN
    assert order.venue_id == fixture["orderID"]


async def test_testnet_cancel_confirms_cancellation():
    fixture = _load("delete_order_success.json")
    client = _build_testnet_client()
    client._raw.cancel.return_value = fixture

    venue_id = fixture["canceled"][0]
    assert await client.cancel_order(venue_id) is True
    client._raw.cancel.assert_called_once_with(venue_id)


# ---------------------------------------------------------------------------
# Canary refuses fail-closed default
# ---------------------------------------------------------------------------


def test_canary_refuses_zero_verifying_contract_on_testnet(monkeypatch, caplog):
    """The signing canary must loudly refuse to sign against the zero-
    address fail-closed default on testnet."""
    import importlib.util
    import logging
    spec = importlib.util.spec_from_file_location(
        "blksch_scripts_signing_canary",
        Path(__file__).resolve().parents[2] / "scripts" / "signing_canary.py",
    )
    canary = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(canary)

    # Provide minimum env so we reach the zero-address check.
    monkeypatch.setenv("POLY_PRIVATE_KEY", "0x" + "3" * 64)
    monkeypatch.setenv("POLY_FUNDER_ADDRESS", "0x0000000000000000000000000000000000000003")
    monkeypatch.setenv("POLY_NETWORK", "testnet")
    monkeypatch.delenv("POLY_VERIFYING_CONTRACT", raising=False)

    caplog.set_level(logging.ERROR)
    import asyncio
    # Mock out py-clob-client so the canary doesn't actually init a signer.
    with patch("py_clob_client.client.ClobClient", MagicMock()):
        rc = asyncio.run(canary._run("0xtoken", 0.01, 1.0, testnet=True))
    assert rc == 2
    assert "fail-closed" in caplog.text.lower() or "zero" in caplog.text.lower() or "verifying_contract" in caplog.text.lower()
