from __future__ import annotations

import os
import time

import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters import FinamRestClient
from core import get_logger


def main() -> None:
    load_dotenv()
    log = get_logger("finam_api_test")

    account_id = os.getenv("FINAM_ACCOUNT_ID")
    if not account_id:
        raise RuntimeError("FINAM_ACCOUNT_ID is required")

    client = FinamRestClient()
    try:
        log.info("JWT token details:")
        details = client.token_details(client._ensure_token())
        log.info(details)

        log.info("Account info:")
        account = client.get_account(account_id)
        log.info(account)

        symbol = os.getenv("FINAM_TEST_SYMBOL", "SBER@MISX")
        log.info("Last quote for %s:", symbol)
        quote = client.get_last_quote(symbol)
        log.info(quote)

        timeframe = os.getenv("FINAM_TEST_TIMEFRAME", "TIME_FRAME_M5")
        days_back = int(os.getenv("FINAM_TEST_DAYS_BACK", "1"))
        end_ts = int(time.time())
        start_ts = end_ts - days_back * 86400
        log.info("Bars for %s (%s, days_back=%s):", symbol, timeframe, days_back)
        bars = client.get_bars(symbol, timeframe, start_ts, end_ts)
        log.info({"bars_count": len(bars.get("bars", []))})
    finally:
        client.close()


if __name__ == "__main__":
    main()
