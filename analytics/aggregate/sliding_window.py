from collections import deque
from decimal import Decimal
from typing import Deque, Tuple


class SlidingWindow:
    """
    Time-based sliding window for tick-based metrics.
    Provides O(1) add and eviction operations to maintain VWAP, volume, and price extremes.
    """
    def __init__(self, window_sec: int):
        self.window_sec = window_sec
        # Each tick stored as (timestamp_ms, price, volume)
        self._ticks: Deque[Tuple[int, Decimal, Decimal]] = deque()
        self._sum_price_volume: Decimal = Decimal(0)
        self._sum_volume: Decimal = Decimal(0)

    def add(self, timestamp_ms: int, price: Decimal, volume: Decimal) -> None:
        """Add a new tick and evict ticks outside the window."""
        # Update aggregates
        self._ticks.append((timestamp_ms, price, volume))
        self._sum_price_volume += price * volume
        self._sum_volume += volume
        self._evict(timestamp_ms)

    def _evict(self, current_ts_ms: int) -> None:
        """Evict ticks older than window_sec (inclusive)."""
        cutoff = current_ts_ms - self.window_sec * 1000
        while self._ticks and self._ticks[0][0] <= cutoff:
            _, old_price, old_volume = self._ticks.popleft()
            self._sum_price_volume -= old_price * old_volume
            self._sum_volume -= old_volume

    def vwap(self) -> Decimal:
        """Compute VWAP over the stored ticks."""
        return self._sum_price_volume / self._sum_volume if self._sum_volume else Decimal(0)

    def volume(self) -> Decimal:
        """Total volume in the window."""
        return self._sum_volume

    def high(self) -> Decimal:
        """Highest tick price in the window."""
        return max((price for _, price, _ in self._ticks), default=Decimal(0))

    def low(self) -> Decimal:
        """Lowest tick price in the window."""
        return min((price for _, price, _ in self._ticks), default=Decimal(0))
