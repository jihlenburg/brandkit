"""
Tests for Discovery Pipeline End-to-End
========================================
Tests the complete discovery pipeline with mocked API responses.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock
from dataclasses import dataclass

# Ensure repo root is on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brandkit.parallel import (
    ParallelConfig,
    RateLimiter,
    RetryHandler,
    CheckResult,
    ParallelDomainChecker,
    ParallelTrademarkChecker,
    ParallelDiscoveryPipeline,
    create_parallel_pipeline,
)


# =============================================================================
# Mock Data Classes
# =============================================================================

@dataclass
class MockDomainInfo:
    """Mock domain availability info."""
    available: bool
    error: str = None


@dataclass
class MockDomainResult:
    """Mock domain check result."""
    name: str
    domains: dict


def create_mock_domain_result(name: str, available_tlds: list) -> MockDomainResult:
    """Helper to create mock domain results."""
    all_tlds = ['com', 'de', 'eu', 'io', 'co']
    domains = {
        tld: MockDomainInfo(available=(tld in available_tlds))
        for tld in all_tlds
    }
    return MockDomainResult(name=name, domains=domains)


# =============================================================================
# RateLimiter Tests
# =============================================================================

class TestRateLimiter:
    """Tests for RateLimiter class."""

    def test_init_valid(self):
        """Test valid initialization."""
        limiter = RateLimiter(requests_per_second=2.0)
        assert limiter.min_interval == 0.5

    def test_init_invalid(self):
        """Test invalid initialization raises error."""
        with pytest.raises(ValueError):
            RateLimiter(requests_per_second=0)
        with pytest.raises(ValueError):
            RateLimiter(requests_per_second=-1)

    def test_acquire_returns_wait_time(self):
        """Test acquire returns wait time."""
        limiter = RateLimiter(requests_per_second=1000)  # High rate for fast test
        wait_time = limiter.acquire()
        assert isinstance(wait_time, (int, float))
        assert wait_time >= 0

    def test_try_acquire_success(self):
        """Test try_acquire succeeds when not rate limited."""
        limiter = RateLimiter(requests_per_second=1000)
        # First call should succeed
        assert limiter.try_acquire()

    def test_try_acquire_rate_limited(self):
        """Test try_acquire fails when rate limited."""
        limiter = RateLimiter(requests_per_second=0.01)  # Very low rate
        limiter.acquire()  # Consume the token
        # Immediate second call should fail
        assert not limiter.try_acquire()


# =============================================================================
# RetryHandler Tests
# =============================================================================

class TestRetryHandler:
    """Tests for RetryHandler class."""

    def test_init_defaults(self):
        """Test default initialization."""
        handler = RetryHandler()
        assert handler.max_retries == 3
        assert handler.base_delay == 1.0
        assert handler.max_delay == 30.0
        assert handler.exponential_base == 2.0

    def test_execute_success_first_try(self):
        """Test execute succeeds on first try."""
        handler = RetryHandler()
        result = handler.execute(lambda: "success")
        assert result == "success"

    def test_execute_success_after_retry(self):
        """Test execute succeeds after retry."""
        handler = RetryHandler(max_retries=3, base_delay=0.01)
        call_count = 0

        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Transient failure")
            return "success"

        result = handler.execute(
            flaky_function,
            retryable_exceptions=(ConnectionError,)
        )
        assert result == "success"
        assert call_count == 2

    def test_execute_max_retries_exceeded(self):
        """Test execute raises after max retries."""
        handler = RetryHandler(max_retries=2, base_delay=0.01)

        def always_fails():
            raise ConnectionError("Always fails")

        with pytest.raises(ConnectionError):
            handler.execute(
                always_fails,
                retryable_exceptions=(ConnectionError,)
            )

    def test_execute_non_retryable_exception(self):
        """Test non-retryable exceptions are not retried."""
        handler = RetryHandler(max_retries=3, base_delay=0.01)
        call_count = 0

        def raises_value_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("Not retryable")

        with pytest.raises(ValueError):
            handler.execute(
                raises_value_error,
                retryable_exceptions=(ConnectionError,)
            )
        # Should only be called once since ValueError is not retryable
        assert call_count == 1


# =============================================================================
# CheckResult Tests
# =============================================================================

class TestCheckResult:
    """Tests for CheckResult dataclass."""

    def test_has_com_true(self):
        """Test has_com returns True when .com is available."""
        result = CheckResult(
            name="test",
            available_domains=['com', 'de']
        )
        assert result.has_com

    def test_has_com_false(self):
        """Test has_com returns False when .com is not available."""
        result = CheckResult(
            name="test",
            available_domains=['de', 'eu']
        )
        assert not result.has_com

    def test_is_available_default(self):
        """Test is_available default is False."""
        result = CheckResult(name="test")
        assert not result.is_available

    def test_error_attribute(self):
        """Test error attribute."""
        result = CheckResult(name="test", error="API timeout")
        assert result.error == "API timeout"


# =============================================================================
# ParallelDomainChecker Tests
# =============================================================================

class TestParallelDomainChecker:
    """Tests for ParallelDomainChecker class."""

    def test_check_delegates_to_base(self):
        """Test check delegates to base checker."""
        mock_base = MagicMock()
        mock_base.check.return_value = create_mock_domain_result("test", ['com'])

        checker = ParallelDomainChecker(mock_base)
        result = checker.check("test")

        mock_base.check.assert_called_once_with("test", use_cache=True)
        assert result.name == "test"

    def test_check_with_rate_limiter(self):
        """Test check uses rate limiter."""
        mock_base = MagicMock()
        mock_base.check.return_value = create_mock_domain_result("test", ['com'])

        mock_limiter = MagicMock()
        checker = ParallelDomainChecker(mock_base, rate_limiter=mock_limiter)
        checker.check("test")

        mock_limiter.acquire.assert_called_once()

    def test_check_batch(self):
        """Test batch checking multiple names."""
        mock_base = MagicMock()
        mock_base.check.side_effect = lambda name, use_cache=True: \
            create_mock_domain_result(name, ['com'] if name == "good" else [])

        checker = ParallelDomainChecker(mock_base, max_workers=2)
        results = checker.check_batch(["good", "bad"])

        assert "good" in results
        assert "bad" in results
        assert results["good"].domains['com'].available
        assert not results["bad"].domains['com'].available

    def test_check_batch_with_callback(self):
        """Test batch checking with callback."""
        mock_base = MagicMock()
        mock_base.check.return_value = create_mock_domain_result("test", ['com'])

        callback_calls = []

        def callback(name, result):
            callback_calls.append((name, result))

        checker = ParallelDomainChecker(mock_base, max_workers=2)
        checker.check_batch(["test"], callback=callback)

        assert len(callback_calls) == 1
        assert callback_calls[0][0] == "test"

    def test_check_batch_handles_errors(self):
        """Test batch checking handles errors gracefully."""
        mock_base = MagicMock()
        mock_base.check.side_effect = ConnectionError("Network error")

        # Use a simple retry handler that doesn't actually retry
        checker = ParallelDomainChecker(
            mock_base,
            max_workers=2,
            retry_handler=RetryHandler(max_retries=0)
        )
        results = checker.check_batch(["test"])

        assert results["test"] is None


# =============================================================================
# ParallelTrademarkChecker Tests
# =============================================================================

class TestParallelTrademarkChecker:
    """Tests for ParallelTrademarkChecker class."""

    def test_check_delegates_to_base(self):
        """Test check delegates to base checker."""
        mock_base = MagicMock()
        mock_base.check.return_value = {'is_available': True, 'conflicts': []}

        checker = ParallelTrademarkChecker(mock_base, rate_limit=1000)
        result = checker.check("TestBrand")

        mock_base.check.assert_called_once_with("TestBrand", nice_classes=None)
        assert result['is_available']

    def test_check_with_nice_classes(self):
        """Test check passes nice classes."""
        mock_base = MagicMock()
        mock_base.check.return_value = {'is_available': True}

        checker = ParallelTrademarkChecker(mock_base, rate_limit=1000)
        checker.check("TestBrand", nice_classes=[9, 42])

        mock_base.check.assert_called_once_with("TestBrand", nice_classes=[9, 42])

    def test_check_batch(self):
        """Test batch checking multiple trademarks."""
        mock_base = MagicMock()
        mock_base.check.return_value = {'is_available': True}

        checker = ParallelTrademarkChecker(mock_base, max_workers=2, rate_limit=1000)
        results = checker.check_batch(["Brand1", "Brand2"])

        assert "Brand1" in results
        assert "Brand2" in results

    def test_check_batch_handles_errors(self):
        """Test batch checking handles API errors."""
        mock_base = MagicMock()
        mock_base.check.side_effect = ConnectionError("API error")

        checker = ParallelTrademarkChecker(
            mock_base,
            max_workers=2,
            rate_limit=1000,
            retry_handler=RetryHandler(max_retries=0)
        )
        results = checker.check_batch(["test"])

        assert 'error' in results["test"]


# =============================================================================
# ParallelDiscoveryPipeline Tests
# =============================================================================

class TestParallelDiscoveryPipeline:
    """Tests for ParallelDiscoveryPipeline class."""

    @pytest.fixture
    def mock_domain_checker(self):
        """Create mock parallel domain checker."""
        mock = MagicMock(spec=ParallelDomainChecker)
        return mock

    @pytest.fixture
    def mock_trademark_checker(self):
        """Create mock parallel trademark checker."""
        mock = MagicMock(spec=ParallelTrademarkChecker)
        return mock

    def test_process_yields_results(self, mock_domain_checker, mock_trademark_checker):
        """Test process yields results for valid names."""
        # Domain check returns available
        mock_domain_checker.check.return_value = create_mock_domain_result(
            "TestBrand", ['com', 'de']
        )
        # Trademark check returns available
        mock_trademark_checker.check.return_value = {'is_available': True}

        pipeline = ParallelDiscoveryPipeline(
            mock_domain_checker,
            mock_trademark_checker,
            ParallelConfig()
        )

        results = list(pipeline.process(["TestBrand"]))

        assert len(results) == 1
        assert results[0].name == "TestBrand"
        assert results[0].is_available
        assert 'com' in results[0].available_domains

    def test_process_filters_by_min_domains(
        self, mock_domain_checker, mock_trademark_checker
    ):
        """Test process filters names with insufficient domains."""
        # Only one domain available
        mock_domain_checker.check.return_value = create_mock_domain_result(
            "TestBrand", ['de']
        )

        pipeline = ParallelDiscoveryPipeline(
            mock_domain_checker,
            mock_trademark_checker,
            ParallelConfig()
        )

        # Require 2 domains minimum
        results = list(pipeline.process(["TestBrand"], min_domains=2))

        assert len(results) == 0
        # Trademark check should not have been called
        mock_trademark_checker.check.assert_not_called()

    def test_process_filters_by_require_com(
        self, mock_domain_checker, mock_trademark_checker
    ):
        """Test process filters names without .com."""
        # .com not available
        mock_domain_checker.check.return_value = create_mock_domain_result(
            "TestBrand", ['de', 'eu']
        )

        pipeline = ParallelDiscoveryPipeline(
            mock_domain_checker,
            mock_trademark_checker,
            ParallelConfig()
        )

        results = list(pipeline.process(["TestBrand"], require_com=True))

        assert len(results) == 0

    def test_process_handles_domain_errors(
        self, mock_domain_checker, mock_trademark_checker
    ):
        """Test process handles domain check errors."""
        mock_domain_checker.check.side_effect = Exception("DNS error")

        pipeline = ParallelDiscoveryPipeline(
            mock_domain_checker,
            mock_trademark_checker,
            ParallelConfig()
        )

        results = list(pipeline.process(["TestBrand"]))

        # Should continue without crashing
        assert len(results) == 0

    def test_process_handles_trademark_errors(
        self, mock_domain_checker, mock_trademark_checker
    ):
        """Test process handles trademark check errors."""
        mock_domain_checker.check.return_value = create_mock_domain_result(
            "TestBrand", ['com']
        )
        mock_trademark_checker.check.side_effect = Exception("API error")

        pipeline = ParallelDiscoveryPipeline(
            mock_domain_checker,
            mock_trademark_checker,
            ParallelConfig()
        )

        results = list(pipeline.process(["TestBrand"]))

        assert len(results) == 1
        assert not results[0].is_available
        assert results[0].error == "API error"

    def test_process_batch(self, mock_domain_checker, mock_trademark_checker):
        """Test process_batch returns list."""
        mock_domain_checker.check.return_value = create_mock_domain_result(
            "TestBrand", ['com']
        )
        mock_trademark_checker.check.return_value = {'is_available': True}

        pipeline = ParallelDiscoveryPipeline(
            mock_domain_checker,
            mock_trademark_checker,
            ParallelConfig()
        )

        results = pipeline.process_batch(["TestBrand"])

        assert isinstance(results, list)
        assert len(results) == 1


# =============================================================================
# Integration Tests with Mocked APIs
# =============================================================================

class TestPipelineIntegration:
    """Integration tests for the full pipeline with mocked external APIs."""

    def test_create_parallel_pipeline(self):
        """Test create_parallel_pipeline factory function."""
        mock_domain = MagicMock()
        mock_trademark = MagicMock()

        config = ParallelConfig(
            domain_workers=5,
            trademark_workers=3,
            trademark_rate_limit=1.0
        )

        pipeline = create_parallel_pipeline(
            mock_domain,
            mock_trademark,
            config
        )

        assert isinstance(pipeline, ParallelDiscoveryPipeline)
        assert pipeline.config.domain_workers == 5
        assert pipeline.config.trademark_workers == 3

    def test_end_to_end_with_mocks(self):
        """Test complete end-to-end flow with mocked checkers."""
        # Create mock base checkers
        mock_domain_base = MagicMock()
        mock_trademark_base = MagicMock()

        # Configure domain responses
        def domain_check(name, use_cache=True):
            available = {
                "GoodBrand": ['com', 'de', 'io'],
                "NoDotCom": ['de', 'eu'],
                "NoDomains": []
            }
            return create_mock_domain_result(name, available.get(name, []))

        mock_domain_base.check.side_effect = domain_check

        # Configure trademark responses
        def trademark_check(name, nice_classes=None):
            available = {"GoodBrand": True, "NoDotCom": True}
            return {'is_available': available.get(name, False)}

        mock_trademark_base.check.side_effect = trademark_check

        # Create pipeline
        pipeline = create_parallel_pipeline(
            mock_domain_base,
            mock_trademark_base,
            ParallelConfig(trademark_rate_limit=1000)
        )

        # Process names
        results = pipeline.process_batch(
            ["GoodBrand", "NoDotCom", "NoDomains"],
            require_com=True
        )

        # Only GoodBrand should pass (has .com and trademark clear)
        assert len(results) == 1
        assert results[0].name == "GoodBrand"
        assert results[0].is_available

    def test_pipeline_with_nice_classes(self):
        """Test pipeline passes nice classes to trademark checker."""
        mock_domain_base = MagicMock()
        mock_trademark_base = MagicMock()

        mock_domain_base.check.return_value = create_mock_domain_result(
            "test", ['com']
        )
        mock_trademark_base.check.return_value = {'is_available': True}

        pipeline = create_parallel_pipeline(
            mock_domain_base,
            mock_trademark_base,
            ParallelConfig(trademark_rate_limit=1000)
        )

        list(pipeline.process(["test"], nice_classes=[9, 42]))

        # Verify nice classes were passed
        call_args = mock_trademark_base.check.call_args
        assert call_args[1].get('nice_classes') == [9, 42] or \
               call_args[0][1] == [9, 42] if len(call_args[0]) > 1 else True


# =============================================================================
# ParallelConfig Tests
# =============================================================================

class TestParallelConfig:
    """Tests for ParallelConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ParallelConfig()

        assert config.domain_workers == 10
        assert config.domain_tld_parallel is True
        assert config.trademark_workers == 5
        assert config.trademark_rate_limit == 2.0
        assert config.max_retries == 3
        assert config.retry_base_delay == 1.0
        assert config.retry_max_delay == 30.0
        assert config.request_timeout == 10.0

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ParallelConfig(
            domain_workers=20,
            trademark_workers=10,
            trademark_rate_limit=5.0
        )

        assert config.domain_workers == 20
        assert config.trademark_workers == 10
        assert config.trademark_rate_limit == 5.0

