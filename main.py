import os
import time

from dotenv import load_dotenv

import httpx
import yaml

from adapters import FinamRestBrokerAdapter, FinamRestClient, FinamRestDataAdapter
from core import EventEngine, NoopBrokerAdapter, RiskLimits, SimpleRiskManager, get_logger
from strategies.ma_cross import MACrossStrategy


def main() -> None:
    load_dotenv()
    log = get_logger("main")
    account_id = os.getenv("FINAM_ACCOUNT_ID")
    if not account_id:
        raise RuntimeError("FINAM_ACCOUNT_ID is required")

    client = FinamRestClient()
    try:
        details = client.token_details(client._ensure_token())
        log.info("Token details: %s", details)
        try:
            account = client.get_account(account_id)
            log.info("Account: %s", account)
        except httpx.HTTPStatusError as exc:
            response = exc.response
            log.error("Account request failed: %s %s", response.status_code, response.text)

        with open("configs/main.yaml", "r", encoding="utf-8") as handle:
            main_cfg = yaml.safe_load(handle) or {}
        with open("configs/strategy_ma_cross.yaml", "r", encoding="utf-8") as handle:
            strat_cfg = yaml.safe_load(handle) or {}

        strat_cfg["account_id"] = account_id
        strategy = MACrossStrategy(
            name=strat_cfg["name"],
            account_id=strat_cfg["account_id"],
            symbol=strat_cfg["symbol"],
            sma_short=int(strat_cfg["sma_short"]),
            sma_long=int(strat_cfg["sma_long"]),
            quantity=float(strat_cfg["quantity"]),
        )

        limits_cfg = main_cfg.get("risk_limits", {})
        limits = RiskLimits(
            max_position_per_symbol=float(limits_cfg.get("max_position_per_symbol", 0)),
            max_trades_per_day=int(limits_cfg.get("max_trades_per_day", 0)),
            daily_loss_limit=float(limits_cfg.get("daily_loss_limit", 0)),
        )
        risk_manager = SimpleRiskManager(limits)

        data_adapter = FinamRestDataAdapter(client)
        mode = str(main_cfg.get("mode", "paper")).lower()
        if mode == "paper":
            broker_adapter = NoopBrokerAdapter()
        else:
            live_cfg = main_cfg.get("live_launch", {})
            max_real_position = float(live_cfg.get("max_real_position", 0))
            if max_real_position and (limits.max_position_per_symbol == 0 or max_real_position < limits.max_position_per_symbol):
                limits.max_position_per_symbol = max_real_position
                log.info("Live launch: max_position_per_symbol set to %.2f", max_real_position)
            broker_adapter = FinamRestBrokerAdapter(client)

        engine = EventEngine(
            data_adapter=data_adapter,
            broker_adapter=broker_adapter,
            risk_manager=risk_manager,
            strategies=[strategy],
        )

        timeframe = main_cfg.get("timeframe", "TIME_FRAME_M1")
        days_back = int(main_cfg.get("days_back", 1))
        end_ts = int(time.time())
        start_ts = end_ts - days_back * 86400
        engine.run_bars(strat_cfg["symbol"], timeframe, start_ts, end_ts)
    finally:
        client.close()


if __name__ == "__main__":
    main()
