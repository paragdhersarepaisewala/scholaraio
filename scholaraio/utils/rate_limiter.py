from collections import deque
from threading import Lock
from typing import Deque

class RateLimiter:
    """
    A thread-safe rate limiter that allows a certain number of calls
    within a specified time period.

    This utility is crucial for AI agents interacting with external APIs
    (e.g., LLMs, research databases, web scraping) to prevent exceeding
    rate limits, respect API terms of service, and ensure stable operation
    without incurring unnecessary costs or IP bans.

    It can be used explicitly by calling `wait_for_next_call()` before an
    operation, or as a context manager.
    """
    def __init__(self, rate: int, period: float = 1.0):
        """
        Initializes the RateLimiter.

        Args:
            rate: The maximum number of calls allowed within the period.
            period: The time window in seconds for the rate limit. Defaults to 1.0 second.
        """
        if not isinstance(rate, int) or rate <= 0:
            raise ValueError("Rate must be a positive integer.")
        if not isinstance(period, (int, float)) or period <= 0:
            raise ValueError("Period must be a positive number.")

        self._rate = rate
        self._period = period
        self._timestamps: Deque[float] = deque()
        self._lock = Lock()

    def wait_for_next_call(self) -> None:
        """
        Blocks the current thread until the next call is permitted by the rate limit.
        This method is thread-safe.

        The thread will sleep if necessary to ensure that the rate limit
        (N calls within T seconds) is not exceeded.
        """
        with self._lock:
            current_time = time.monotonic()

            # Clean up old timestamps (those outside the current period)
            while self._timestamps and self._timestamps[0] <= current_time - self._period:
                self._timestamps.popleft()

            # If we've reached the rate limit for the current period, calculate wait time
            if len(self._timestamps) >= self._rate:
                # The time when the oldest call expires and a new slot becomes available
                time_oldest_expires = self._timestamps[0] + self._period
                time_to_wait = time_oldest_expires - current_time

                if time_to_wait > 0:
                    time.sleep(time_to_wait)
                    # After waiting, update current_time as time.sleep blocks
                    current_time = time.monotonic()
                    # Clean up again in case more timestamps became old during the wait
                    while self._timestamps and self._timestamps[0] <= current_time - self._period:
                        self._timestamps.popleft()

            # Record the timestamp of the new call
            self._timestamps.append(current_time)

    def __enter__(self):
        """
        Allows the RateLimiter to be used as a context manager.
        It calls `wait_for_next_call()` upon entering the context.
        """
        self.wait_for_next_call()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        No specific action needed upon exiting the context.
        """
        pass