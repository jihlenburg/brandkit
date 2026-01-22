#!/usr/bin/env python3
"""
Parallel Checking Infrastructure
=================================
High-performance parallel checking for domain and trademark validation.

Features:
- Parallel TLD checking within DomainChecker
- Batch processing for multiple names
- Interleaved pipeline (start trademark as domain completes)
- Rate limiting for API protection
- Retry logic with exponential backoff

Usage:
    from brandkit.parallel import ParallelDiscoveryPipeline, ParallelConfig

    config = ParallelConfig(
        domain_workers=10,
        trademark_workers=5,
        trademark_rate_limit=2.0
    )

    pipeline = ParallelDiscoveryPipeline(
        domain_checker, trademark_checker, config
    )

    for result in pipeline.process(names):
        print(f"{result.name}: {result.status}")
"""

import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from dataclasses import dataclass, field
from typing import Optional, Iterator, Callable, Any, List, Dict
from queue import Queue, Empty
import logging

from brandkit.settings import get_setting
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ParallelConfig:
    """Configuration for parallel checking."""
    # Domain checking
    domain_workers: Optional[int] = None          # Concurrent domain name checks
    domain_tld_parallel: Optional[bool] = None    # Check TLDs in parallel within each name

    # Trademark checking
    trademark_workers: Optional[int] = None       # Concurrent trademark checks (lower for rate limits)
    trademark_rate_limit: Optional[float] = None  # Max requests per second

    # Retry settings
    max_retries: Optional[int] = None
    retry_base_delay: Optional[float] = None      # Base delay for exponential backoff
    retry_max_delay: Optional[float] = None       # Max delay between retries
    retry_exponential_base: Optional[float] = None

    # Timeouts
    request_timeout: Optional[float] = None       # Per-request timeout

    def __post_init__(self):
        cfg = get_setting("parallel", {}) or {}
        if self.domain_workers is None:
            self.domain_workers = cfg.get("domain_workers")
        if self.domain_tld_parallel is None:
            self.domain_tld_parallel = cfg.get("domain_tld_parallel")
        if self.trademark_workers is None:
            self.trademark_workers = cfg.get("trademark_workers")
        if self.trademark_rate_limit is None:
            self.trademark_rate_limit = cfg.get("trademark_rate_limit")
        if self.max_retries is None:
            self.max_retries = cfg.get("max_retries")
        if self.retry_base_delay is None:
            self.retry_base_delay = cfg.get("retry_base_delay")
        if self.retry_max_delay is None:
            self.retry_max_delay = cfg.get("retry_max_delay")
        if self.retry_exponential_base is None:
            self.retry_exponential_base = cfg.get("retry_exponential_base")
        if self.request_timeout is None:
            self.request_timeout = cfg.get("request_timeout")

        missing = [
            name for name, value in (
                ("domain_workers", self.domain_workers),
                ("domain_tld_parallel", self.domain_tld_parallel),
                ("trademark_workers", self.trademark_workers),
                ("trademark_rate_limit", self.trademark_rate_limit),
                ("max_retries", self.max_retries),
                ("retry_base_delay", self.retry_base_delay),
                ("retry_max_delay", self.retry_max_delay),
                ("retry_exponential_base", self.retry_exponential_base),
                ("request_timeout", self.request_timeout),
            )
            if value is None
        ]
        if missing:
            raise ValueError(f"parallel settings missing in app.yaml: {', '.join(missing)}")


# =============================================================================
# Rate Limiting
# =============================================================================

class RateLimiter:
    """
    Token bucket rate limiter for API calls.

    Thread-safe implementation that limits requests per second.

    Usage:
        limiter = RateLimiter(requests_per_second=2.0)
        limiter.acquire()  # Blocks until rate limit allows
        make_api_call()
    """

    def __init__(self, requests_per_second: float):
        """
        Initialize rate limiter.

        Args:
            requests_per_second: Maximum requests allowed per second
        """
        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")

        self.min_interval = 1.0 / requests_per_second
        self.last_request = 0.0
        self._lock = threading.Lock()

    def acquire(self) -> float:
        """
        Acquire permission to make a request.

        Blocks until the rate limit allows another request.

        Returns:
            Time waited (seconds)
        """
        with self._lock:
            now = time.perf_counter()
            elapsed = now - self.last_request
            wait_time = max(0, self.min_interval - elapsed)

            if wait_time > 0:
                time.sleep(wait_time)

            self.last_request = time.perf_counter()
            return wait_time

    def try_acquire(self) -> bool:
        """
        Try to acquire without blocking.

        Returns:
            True if acquired, False if rate limited
        """
        with self._lock:
            now = time.perf_counter()
            elapsed = now - self.last_request

            if elapsed >= self.min_interval:
                self.last_request = now
                return True
            return False


# =============================================================================
# Retry Logic
# =============================================================================

class RetryHandler:
    """
    Handles retry logic with exponential backoff.

    Usage:
        retry = RetryHandler(max_retries=3, base_delay=1.0)

        result = retry.execute(
            func=api_call,
            args=(name,),
            retryable_exceptions=(ConnectionError, TimeoutError)
        )
    """

    def __init__(self,
                 max_retries: Optional[int] = None,
                 base_delay: Optional[float] = None,
                 max_delay: Optional[float] = None,
                 exponential_base: Optional[float] = None):
        """
        Initialize retry handler.

        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Initial delay between retries (seconds)
            max_delay: Maximum delay between retries (seconds)
            exponential_base: Base for exponential backoff
        """
        cfg = get_setting("parallel", {}) or {}
        if max_retries is None:
            max_retries = cfg.get("max_retries")
        if base_delay is None:
            base_delay = cfg.get("retry_base_delay")
        if max_delay is None:
            max_delay = cfg.get("retry_max_delay")
        if exponential_base is None:
            exponential_base = cfg.get("retry_exponential_base")
        if max_retries is None or base_delay is None or max_delay is None or exponential_base is None:
            raise ValueError("parallel retry settings must be set in app.yaml")

        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base

    def execute(self,
                func: Callable,
                args: tuple = (),
                kwargs: dict = None,
                retryable_exceptions: tuple = (Exception,)) -> Any:
        """
        Execute function with retry logic.

        Args:
            func: Function to execute
            args: Positional arguments
            kwargs: Keyword arguments
            retryable_exceptions: Exception types that trigger retry

        Returns:
            Function result

        Raises:
            Last exception if all retries fail
        """
        kwargs = kwargs or {}
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except retryable_exceptions as e:
                last_exception = e

                if attempt < self.max_retries:
                    delay = min(
                        self.base_delay * (self.exponential_base ** attempt),
                        self.max_delay
                    )
                    logger.debug(f"Retry {attempt + 1}/{self.max_retries} after {delay:.2f}s: {e}")
                    time.sleep(delay)

        raise last_exception


# =============================================================================
# Parallel Checkers
# =============================================================================

@dataclass
class CheckResult:
    """Result from parallel checking pipeline."""
    name: str
    domain_result: Any = None
    trademark_result: Any = None
    available_domains: List[str] = field(default_factory=list)
    is_available: bool = False
    error: Optional[str] = None

    @property
    def has_com(self) -> bool:
        return 'com' in self.available_domains


class ParallelDomainChecker:
    """
    Wrapper for DomainChecker with parallel batch processing.

    Checks multiple names concurrently while respecting rate limits.
    """

    def __init__(self,
                 base_checker,
                 max_workers: Optional[int] = None,
                 rate_limiter: RateLimiter = None,
                 retry_handler: RetryHandler = None):
        """
        Initialize parallel domain checker.

        Args:
            base_checker: Underlying DomainChecker instance
            max_workers: Max concurrent name checks
            rate_limiter: Optional rate limiter
            retry_handler: Optional retry handler
        """
        self.base_checker = base_checker
        if max_workers is None:
            max_workers = get_setting("parallel.domain_workers")
        if max_workers is None:
            raise ValueError("parallel.domain_workers must be set in app.yaml")
        self.max_workers = max_workers
        self.rate_limiter = rate_limiter
        self.retry_handler = retry_handler or RetryHandler()

    def check(self, name: str, use_cache: bool = True):
        """Check a single name (delegates to base checker)."""
        if self.rate_limiter:
            self.rate_limiter.acquire()

        return self.retry_handler.execute(
            self.base_checker.check,
            args=(name,),
            kwargs={'use_cache': use_cache},
            retryable_exceptions=(ConnectionError, TimeoutError, OSError)
        )

    def check_batch(self,
                    names: List[str],
                    use_cache: bool = True,
                    callback: Callable = None) -> Dict[str, Any]:
        """
        Check multiple names in parallel.

        Args:
            names: List of names to check
            use_cache: Use cached results
            callback: Optional callback(name, result) for each completion

        Returns:
            Dictionary mapping names to results
        """
        results = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_name = {
                executor.submit(self.check, name, use_cache): name
                for name in names
            }

            for future in as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    result = future.result()
                    results[name] = result
                    if callback:
                        callback(name, result)
                except Exception as e:
                    logger.warning(f"Domain check failed for {name}: {e}")
                    results[name] = None

        return results


class ParallelTrademarkChecker:
    """
    Wrapper for TrademarkChecker with rate limiting and parallel batch processing.

    Respects API rate limits while maximizing throughput.
    """

    def __init__(self,
                 base_checker,
                 max_workers: Optional[int] = None,
                 rate_limit: Optional[float] = None,
                 retry_handler: RetryHandler = None):
        """
        Initialize parallel trademark checker.

        Args:
            base_checker: Underlying TrademarkChecker instance
            max_workers: Max concurrent trademark checks
            rate_limit: Max requests per second
            retry_handler: Optional retry handler
        """
        self.base_checker = base_checker
        if max_workers is None:
            max_workers = get_setting("parallel.trademark_workers")
        if rate_limit is None:
            rate_limit = get_setting("parallel.trademark_rate_limit")
        if max_workers is None or rate_limit is None:
            raise ValueError("parallel.trademark settings must be set in app.yaml")
        self.max_workers = max_workers
        self.rate_limiter = RateLimiter(rate_limit)
        self.retry_handler = retry_handler or RetryHandler()

    def check(self, name: str, nice_classes=None):
        """Check a single name with rate limiting."""
        self.rate_limiter.acquire()

        return self.retry_handler.execute(
            self.base_checker.check,
            args=(name,),
            kwargs={'nice_classes': nice_classes},
            retryable_exceptions=(ConnectionError, TimeoutError, OSError)
        )

    def check_batch(self,
                    names: List[str],
                    nice_classes=None,
                    callback: Callable = None) -> Dict[str, Any]:
        """
        Check multiple names in parallel with rate limiting.

        Args:
            names: List of names to check
            nice_classes: Nice classification codes
            callback: Optional callback(name, result) for each completion

        Returns:
            Dictionary mapping names to results
        """
        results = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_name = {
                executor.submit(self.check, name, nice_classes): name
                for name in names
            }

            for future in as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    result = future.result()
                    results[name] = result
                    if callback:
                        callback(name, result)
                except Exception as e:
                    logger.warning(f"Trademark check failed for {name}: {e}")
                    results[name] = {'error': str(e)}

        return results


# =============================================================================
# Interleaved Pipeline
# =============================================================================

class ParallelDiscoveryPipeline:
    """
    Interleaved parallel pipeline for discovery.

    Starts trademark checks as soon as domain results arrive,
    maximizing I/O overlap and minimizing total time.

    Pipeline flow:
        names → [parallel domain checks] → [parallel trademark checks] → results
                        ↓                           ↑
                   (as each completes, immediately submit)

    Usage:
        pipeline = ParallelDiscoveryPipeline(domain_checker, trademark_checker)

        for result in pipeline.process(names, nice_classes=[9, 12]):
            if result.is_available:
                print(f"{result.name} is available!")
    """

    def __init__(self,
                 domain_checker: ParallelDomainChecker,
                 trademark_checker: ParallelTrademarkChecker,
                 config: ParallelConfig = None):
        """
        Initialize pipeline.

        Args:
            domain_checker: ParallelDomainChecker instance
            trademark_checker: ParallelTrademarkChecker instance
            config: Optional configuration
        """
        self.domain_checker = domain_checker
        self.trademark_checker = trademark_checker
        self.config = config or ParallelConfig()

    def process(self,
                names: List[str],
                nice_classes=None,
                min_domains: Optional[int] = None,
                require_com: Optional[bool] = None) -> Iterator[CheckResult]:
        """
        Process names through parallel pipeline.

        Names are checked for domain availability first, then those with
        available domains are checked for trademark conflicts. Results
        are yielded as they complete.

        Args:
            names: List of brand names to check
            nice_classes: Nice classification codes for trademark search
            min_domains: Minimum available domains to proceed to trademark check
            require_com: Require .com to be available

        Yields:
            CheckResult for each name that passes domain check
        """
        pipeline_defaults = get_setting("parallel.pipeline_defaults", {}) or {}
        if min_domains is None:
            min_domains = pipeline_defaults.get("min_domains")
        if require_com is None:
            require_com = pipeline_defaults.get("require_com")
        if min_domains is None or require_com is None:
            raise ValueError("parallel.pipeline_defaults must be set in app.yaml")

        domain_executor = ThreadPoolExecutor(
            max_workers=self.config.domain_workers
        )
        trademark_executor = ThreadPoolExecutor(
            max_workers=self.config.trademark_workers
        )

        try:
            # Submit all domain checks
            domain_futures = {
                domain_executor.submit(self.domain_checker.check, name): name
                for name in names
            }

            # Track pending trademark checks
            trademark_futures = {}

            # Process domain results as they complete
            for future in as_completed(domain_futures):
                name = domain_futures[future]

                try:
                    domain_result = future.result()
                except Exception as e:
                    logger.warning(f"Domain check failed for {name}: {e}")
                    continue

                # Check if enough domains are available
                available = []
                if domain_result and hasattr(domain_result, 'domains'):
                    available = [
                        tld for tld, info in domain_result.domains.items()
                        if info.available
                    ]

                # Skip if doesn't meet requirements
                if len(available) < min_domains:
                    continue
                if require_com and 'com' not in available:
                    continue

                # Submit trademark check (interleaved!)
                tm_future = trademark_executor.submit(
                    self.trademark_checker.check,
                    name,
                    nice_classes
                )
                trademark_futures[tm_future] = (name, domain_result, available)

            # Collect trademark results
            for future in as_completed(trademark_futures):
                name, domain_result, available = trademark_futures[future]

                try:
                    tm_result = future.result()
                    is_available = tm_result.get('is_available', True) if isinstance(tm_result, dict) else True

                    yield CheckResult(
                        name=name,
                        domain_result=domain_result,
                        trademark_result=tm_result,
                        available_domains=available,
                        is_available=is_available
                    )
                except Exception as e:
                    logger.warning(f"Trademark check failed for {name}: {e}")
                    yield CheckResult(
                        name=name,
                        domain_result=domain_result,
                        available_domains=available,
                        is_available=False,
                        error=str(e)
                    )

        finally:
            domain_executor.shutdown(wait=False)
            trademark_executor.shutdown(wait=False)

    def process_batch(self,
                      names: List[str],
                      nice_classes=None,
                      min_domains: Optional[int] = None,
                      require_com: Optional[bool] = None) -> List[CheckResult]:
        """
        Process names and return all results as a list.

        Same as process() but collects all results before returning.

        Args:
            names: List of brand names to check
            nice_classes: Nice classification codes
            min_domains: Minimum available domains required
            require_com: Require .com availability

        Returns:
            List of CheckResult objects
        """
        return list(self.process(names, nice_classes, min_domains, require_com))


# =============================================================================
# Factory Functions
# =============================================================================

def create_parallel_pipeline(domain_checker,
                             trademark_checker,
                             config: ParallelConfig = None) -> ParallelDiscoveryPipeline:
    """
    Create a configured parallel discovery pipeline.

    Args:
        domain_checker: Base DomainChecker instance
        trademark_checker: Base TrademarkChecker instance
        config: Optional configuration

    Returns:
        Configured ParallelDiscoveryPipeline
    """
    config = config or ParallelConfig()

    # Wrap checkers with parallel support
    parallel_domain = ParallelDomainChecker(
        domain_checker,
        max_workers=config.domain_workers,
        retry_handler=RetryHandler(
            max_retries=config.max_retries,
            base_delay=config.retry_base_delay,
            max_delay=config.retry_max_delay
        )
    )

    parallel_trademark = ParallelTrademarkChecker(
        trademark_checker,
        max_workers=config.trademark_workers,
        rate_limit=config.trademark_rate_limit,
        retry_handler=RetryHandler(
            max_retries=config.max_retries,
            base_delay=config.retry_base_delay,
            max_delay=config.retry_max_delay
        )
    )

    return ParallelDiscoveryPipeline(
        parallel_domain,
        parallel_trademark,
        config
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'ParallelConfig',
    'RateLimiter',
    'RetryHandler',
    'CheckResult',
    'ParallelDomainChecker',
    'ParallelTrademarkChecker',
    'ParallelDiscoveryPipeline',
    'create_parallel_pipeline',
]
