#!/usr/bin/env python3
"""
Domain Availability Checker
============================
Checks if domains are available for brand names.

Uses DNS queries and WHOIS-like checks without external dependencies.
"""

import socket
import time
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import http.client
import ssl

from settings import get_setting, resolve_path

@dataclass
class DomainResult:
    """Result of domain availability check"""
    name: str
    domain: str
    tld: str
    available: Optional[bool]  # None = unknown/error
    has_website: bool = False
    error: Optional[str] = None


@dataclass
class DomainCheckResult:
    """Combined result for all TLDs"""
    name: str
    domains: dict  # TLD -> DomainResult
    any_available: bool
    all_available: bool

    def __post_init__(self):
        if self.domains is None:
            self.domains = {}


class DomainChecker:
    """
    Checks domain availability for brand names.

    Uses DNS resolution to check if domains are registered.
    Note: DNS check can't distinguish "registered but not used" vs "available".

    For more accurate results, would need WHOIS API access.

    Usage:
        checker = DomainChecker()
        result = checker.check("voltix")
        print(f"voltix.com available: {result.domains['com'].available}")

        # Parallel checking (5x faster)
        checker = DomainChecker(parallel=True)
        result = checker.check("voltix")
    """

    def __init__(self, tlds: list = None, cache_dir: str = None,
                 parallel: Optional[bool] = None, max_workers: Optional[int] = None):
        """
        Initialize domain checker.

        Args:
            tlds: List of TLDs to check (default: com, de, eu, io, co)
            cache_dir: Directory to cache results
            parallel: Enable parallel TLD checking (default: False)
            max_workers: Max concurrent TLD checks when parallel=True
        """
        cfg = get_setting("domain_checker", {}) or {}
        if tlds is None:
            tlds = cfg.get("default_tlds")
        if not tlds:
            raise ValueError("domain_checker.default_tlds must be set in app.yaml")
        self.tlds = tlds

        if parallel is None:
            parallel = cfg.get("parallel_default")
        if parallel is None:
            raise ValueError("domain_checker.parallel_default must be set in app.yaml")
        self.parallel = parallel

        if max_workers is None:
            max_workers = cfg.get("max_workers")
        if max_workers is None:
            raise ValueError("domain_checker.max_workers must be set in app.yaml")
        self.max_workers = max_workers

        self.cache_ttl_seconds = cfg.get("cache_ttl_seconds")
        self.dns_timeout_seconds = cfg.get("dns_timeout_seconds")
        self.http_timeout_seconds = cfg.get("http_timeout_seconds")
        self.delay_seconds = cfg.get("delay_seconds")
        self.cache_hash_length = cfg.get("cache_hash_length")
        if self.cache_ttl_seconds is None:
            raise ValueError("domain_checker.cache_ttl_seconds must be set in app.yaml")
        if self.dns_timeout_seconds is None:
            raise ValueError("domain_checker.dns_timeout_seconds must be set in app.yaml")
        if self.http_timeout_seconds is None:
            raise ValueError("domain_checker.http_timeout_seconds must be set in app.yaml")
        if self.delay_seconds is None:
            raise ValueError("domain_checker.delay_seconds must be set in app.yaml")
        if self.cache_hash_length is None:
            raise ValueError("domain_checker.cache_hash_length must be set in app.yaml")

        # Setup cache
        if cache_dir:
            self.cache_dir = resolve_path(cache_dir)
        else:
            self.cache_dir = resolve_path(cfg.get("cache_dir"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, name: str) -> Path:
        """Get cache file path for a name + TLD set"""
        tlds_key = ",".join(self.tlds)
        key = f"{name.lower()}|{tlds_key}"
        name_hash = hashlib.md5(key.encode()).hexdigest()[:self.cache_hash_length]
        return self.cache_dir / f"{name_hash}.json"

    def _load_from_cache(self, name: str) -> Optional[DomainCheckResult]:
        """Load cached result if available and fresh (< 24h)"""
        cache_path = self._get_cache_path(name)
        if not cache_path.exists():
            return None

        try:
            data = json.loads(cache_path.read_text())
            if self.cache_ttl_seconds is None:
                raise ValueError("domain_checker.cache_ttl_seconds must be set in app.yaml")
            if time.time() - data.get('timestamp', 0) > self.cache_ttl_seconds:
                return None
            # Ensure cache matches current TLDs
            if data.get('tlds') != self.tlds:
                return None

            domains = {}
            for tld, d in data.get('domains', {}).items():
                domains[tld] = DomainResult(
                    name=d['name'],
                    domain=d['domain'],
                    tld=tld,
                    available=d['available'],
                    has_website=d.get('has_website', False),
                    error=d.get('error')
                )

            return DomainCheckResult(
                name=data['name'],
                domains=domains,
                any_available=data['any_available'],
                all_available=data['all_available']
            )
        except (json.JSONDecodeError, KeyError):
            return None

    def _save_to_cache(self, result: DomainCheckResult):
        """Save result to cache"""
        cache_path = self._get_cache_path(result.name)
        data = {
            'name': result.name,
            'tlds': self.tlds,
            'domains': {
                tld: {
                    'name': d.name,
                    'domain': d.domain,
                    'available': d.available,
                    'has_website': d.has_website,
                    'error': d.error
                }
                for tld, d in result.domains.items()
            },
            'any_available': result.any_available,
            'all_available': result.all_available,
            'timestamp': time.time()
        }
        cache_path.write_text(json.dumps(data, indent=2))

    def _check_dns(self, domain: str) -> tuple[bool, bool]:
        """
        Check if domain resolves via DNS.

        Returns:
            Tuple of (is_registered, has_website)
            - is_registered: True if domain has DNS records
            - has_website: True if domain responds on port 80/443
        """
        is_registered = False
        has_website = False

        try:
            # Try to resolve the domain
            prev_timeout = socket.getdefaulttimeout()
            if self.dns_timeout_seconds is None:
                raise ValueError("domain_checker.dns_timeout_seconds must be set in app.yaml")
            socket.setdefaulttimeout(self.dns_timeout_seconds)
            ip = socket.gethostbyname(domain)
            is_registered = True

            # Check if there's a web server
            try:
                if self.http_timeout_seconds is None:
                    raise ValueError("domain_checker.http_timeout_seconds must be set in app.yaml")
                conn = http.client.HTTPSConnection(domain, timeout=self.http_timeout_seconds)
                conn.request("HEAD", "/")
                response = conn.getresponse()
                has_website = response.status < 500
                conn.close()
            except:
                try:
                    conn = http.client.HTTPConnection(domain, timeout=self.http_timeout_seconds)
                    conn.request("HEAD", "/")
                    response = conn.getresponse()
                    has_website = response.status < 500
                    conn.close()
                except:
                    pass

        except socket.gaierror:
            # Domain doesn't resolve - might be available
            is_registered = False
        except socket.timeout:
            # Timeout - assume registered
            is_registered = True
        except Exception:
            # Other error - unknown
            pass
        finally:
            try:
                socket.setdefaulttimeout(prev_timeout)
            except Exception:
                pass

        return is_registered, has_website

    def _check_single_domain(self, name: str, tld: str) -> DomainResult:
        """Check a single domain."""
        # Clean the name for domain use
        clean_name = name.lower()
        clean_name = ''.join(c for c in clean_name if c.isalnum() or c == '-')

        domain = f"{clean_name}.{tld}"

        try:
            is_registered, has_website = self._check_dns(domain)

            return DomainResult(
                name=name,
                domain=domain,
                tld=tld,
                available=not is_registered,
                has_website=has_website
            )
        except Exception as e:
            return DomainResult(
                name=name,
                domain=domain,
                tld=tld,
                available=None,
                error=str(e)
            )

    def _check_tlds_sequential(self, name: str) -> dict:
        """Check TLDs sequentially (original behavior)."""
        domains = {}
        for tld in self.tlds:
            result = self._check_single_domain(name, tld)
            domains[tld] = result
            if self.delay_seconds is None:
                raise ValueError("domain_checker.delay_seconds must be set in app.yaml")
            time.sleep(self.delay_seconds)
        return domains

    def _check_tlds_parallel(self, name: str) -> dict:
        """Check all TLDs in parallel using ThreadPoolExecutor."""
        domains = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all TLD checks
            future_to_tld = {
                executor.submit(self._check_single_domain, name, tld): tld
                for tld in self.tlds
            }
            # Collect results as they complete
            for future in as_completed(future_to_tld):
                tld = future_to_tld[future]
                try:
                    domains[tld] = future.result()
                except Exception as e:
                    # Handle any unexpected errors
                    domains[tld] = DomainResult(
                        name=name,
                        domain=f"{name.lower()}.{tld}",
                        tld=tld,
                        available=None,
                        error=str(e)
                    )
        return domains

    def check(self, name: str, use_cache: bool = True) -> DomainCheckResult:
        """
        Check domain availability for all configured TLDs.

        Args:
            name: Brand name to check
            use_cache: Use cached results if available

        Returns:
            DomainCheckResult with availability for each TLD
        """
        # Check cache
        if use_cache:
            cached = self._load_from_cache(name)
            if cached:
                return cached

        # Check TLDs (parallel or sequential)
        if self.parallel:
            domains = self._check_tlds_parallel(name)
        else:
            domains = self._check_tlds_sequential(name)

        any_available = any(d.available for d in domains.values() if d.available is not None)
        all_available = all(d.available for d in domains.values() if d.available is not None)

        result = DomainCheckResult(
            name=name,
            domains=domains,
            any_available=any_available,
            all_available=all_available
        )

        # Cache the result
        if use_cache:
            self._save_to_cache(result)

        return result

    def check_batch(self, names: list[str],
                    use_cache: bool = True,
                    delay: float = 0.2,
                    parallel_names: bool = False,
                    max_concurrent_names: int = 10) -> dict[str, DomainCheckResult]:
        """
        Check multiple names.

        Args:
            names: List of brand names to check
            use_cache: Use cached results
            delay: Delay between checks (seconds) - only for sequential mode
            parallel_names: Check multiple names in parallel
            max_concurrent_names: Max concurrent name checks when parallel_names=True

        Returns:
            Dictionary mapping names to results
        """
        if parallel_names:
            # Parallel name checking
            results = {}
            with ThreadPoolExecutor(max_workers=max_concurrent_names) as executor:
                future_to_name = {
                    executor.submit(self.check, name, use_cache): name
                    for name in names
                }
                for future in as_completed(future_to_name):
                    name = future_to_name[future]
                    try:
                        results[name] = future.result()
                    except Exception as e:
                        results[name] = DomainCheckResult(
                            name=name,
                            domains={},
                            any_available=False,
                            all_available=False
                        )
            return results
        else:
            # Sequential checking (original behavior)
            results = {}
            for i, name in enumerate(names):
                results[name] = self.check(name, use_cache=use_cache)
                if i < len(names) - 1:
                    time.sleep(delay)
            return results

    def filter_available(self, names: list[str],
                         required_tlds: list = None) -> tuple[list[str], list[str]]:
        """
        Filter names by domain availability.

        Args:
            names: List of brand names
            required_tlds: TLDs that must be available (default: ['com'])

        Returns:
            Tuple of (available_names, taken_names)
        """
        if required_tlds is None:
            required_tlds = ['com']

        available = []
        taken = []

        for name in names:
            result = self.check(name)

            # Check if all required TLDs are available
            all_required_available = all(
                result.domains.get(tld, DomainResult(name, '', tld, False)).available
                for tld in required_tlds
            )

            if all_required_available:
                available.append(name)
            else:
                taken.append(name)

        return available, taken


# Singleton
_default_checker = None

def get_domain_checker() -> DomainChecker:
    """Get default checker instance"""
    global _default_checker
    if _default_checker is None:
        _default_checker = DomainChecker()
    return _default_checker


def check_domain(name: str) -> DomainCheckResult:
    """Quick check for a single name."""
    return get_domain_checker().check(name)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Check domain availability')
    parser.add_argument('names', nargs='+', help='Brand names to check')
    default_cli_tlds = get_setting("domain_checker.default_tlds")
    if not default_cli_tlds:
        raise ValueError("domain_checker.default_tlds must be set in app.yaml")
    parser.add_argument('--tlds', nargs='+', default=default_cli_tlds,
                        help='TLDs to check')
    parser.add_argument('--no-cache', action='store_true', help='Skip cache')

    args = parser.parse_args()

    checker = DomainChecker(tlds=args.tlds)

    for name in args.names:
        result = checker.check(name, use_cache=not args.no_cache)

        print(f"\n{'='*50}")
        print(f"Name: {name}")
        print(f"Any available: {'Yes' if result.any_available else 'No'}")
        print(f"All available: {'Yes' if result.all_available else 'No'}")

        print("\nDomains:")
        for tld, d in result.domains.items():
            if d.available is None:
                status = "? (error)"
            elif d.available:
                status = "AVAILABLE"
            else:
                status = f"taken{' (has website)' if d.has_website else ''}"
            print(f"  {d.domain}: {status}")
