"""Bot entrypoint. Day-0 stub: parses --mode and logs 'not wired yet'.

Wiring up the tracks happens in Track B (mm/refresh_loop.py) and Track C
(exec/order_router.py). See HANDOFF.md and ARCHITECTURE.md.
"""

from __future__ import annotations

import argparse
import logging
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="blksch", description="Polymarket MM bot")
    parser.add_argument(
        "--mode",
        choices=("paper", "live"),
        default="paper",
        help="paper trading (simulated matching) or live CLOB orders",
    )
    parser.add_argument(
        "--config",
        default="config/bot.yaml",
        help="path to bot.yaml",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("blksch")

    log.info("blksch starting mode=%s config=%s", args.mode, args.config)
    log.warning("not wired yet — Day-0 scaffold only; see CATALOG.md for status")
    return 0


if __name__ == "__main__":
    sys.exit(main())
