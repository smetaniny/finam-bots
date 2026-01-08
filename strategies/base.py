from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from core.interfaces import Bar, OrderRequest, Strategy


class BaseStrategy(Strategy, ABC):
    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def on_bar(self, bar: Bar) -> Optional[OrderRequest]:
        raise NotImplementedError
