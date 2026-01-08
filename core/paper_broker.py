from __future__ import annotations

from typing import Any, Dict

from .interfaces import BrokerAdapter, OrderRequest
from .logger import get_logger


class NoopBrokerAdapter(BrokerAdapter):
    def __init__(self) -> None:
        self.log = get_logger("noop_broker")

    def place_order(self, order: OrderRequest) -> Dict[str, Any]:
        self.log.info("Paper order: %s", order)
        return {"status": "paper", "order": order}

    def cancel_order(self, account_id: str, order_id: str) -> Dict[str, Any]:
        self.log.info("Paper cancel: account=%s order=%s", account_id, order_id)
        return {"status": "paper", "order_id": order_id}

    def get_order(self, account_id: str, order_id: str) -> Dict[str, Any]:
        return {"status": "paper", "order_id": order_id}

    def get_account(self, account_id: str) -> Dict[str, Any]:
        return {"status": "paper", "account_id": account_id}
