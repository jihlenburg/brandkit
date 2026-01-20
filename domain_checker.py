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
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import http.client
import ssl


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
    """

    DEFAULT_TLDS = ['com', 'de', 'eu', 'io', 'co']

    def __init__(self, tlds: list = None, cache_dir: str = None):
        """
        Initialize domain checker.

        Args:
            tlds: List of TLDs to check (default: com, de, eu, io, co)
            cache_dir: Directory to cache results
        """
        self.tlds = tlds or self.DEFAULT_TLDS

        # Setup cache
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path.home() / ".cache" / "domain_checker"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, name: str) -> Path:
        """Get cache file path for a name"""
        name_hash = hashlib.md5(name.lower().encode()).hexdigest()[:12]
        return self.cache_dir / f"{name_hash}.json"

    def _load_from_cache(self, name: str) -> Optional[DomainCheckResult]:
        """Load cached result if available and fresh (< 24h)"""
        cache_path = self._get_cache_path(name)
        if not cache_path.exists():
            return None

        try:
            data = json.loads(cache_path.read_text())
            # Check if cache is fresh (24 hours)
            if time.time() - data.get('timestamp', 0) > 86400:
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
            socket.setdefaulttimeout(3)
            ip = socket.gethostbyname(domain)
            is_registered = True

            # Check if there's a web server
            try:
                conn = http.client.HTTPSConnection(domain, timeout=3)
                conn.request("HEAD", "/")
                response = conn.getresponse()
                has_website = response.status < 500
                conn.close()
            except:
                try:
                    conn = http.client.HTTPConnection(domain, timeout=3)
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

        domains = {}
        for tld in self.tlds:
            result = self._check_single_domain(name, tld)
            domains[tld] = result
            # Small delay to avoid rate limiting
            time.sleep(0.1)

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
                    delay: float = 0.2) -> dict[str, DomainCheckResult]:
        """
        Check multiple names.

        Args:
            names: List of brand names to check
            use_cache: Use cached results
            delay: Delay between checks (seconds)

        Returns:
            Dictionary mapping names to results
        """
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
    parser.add_argument('--tlds', nargs='+', default=['com', 'de', 'eu'],
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
