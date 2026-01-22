#!/usr/bin/env python3
"""
EUIPO (European Union Intellectual Property Office) Trademark Checker
=====================================================================
Checks if brand names are already registered as EU trademarks.

Two modes:
1. API Mode: Uses official EUIPO Trademark Search API (requires registration)
2. URL Mode: Generates eSearch URLs for manual checking

API Registration: https://dev.euipo.europa.eu/
Sandbox: https://dev-sandbox.euipo.europa.eu/
"""

import os
import json
import time
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from urllib.parse import quote_plus
import http.client
import ssl

from settings import get_setting, resolve_path

@dataclass
class TrademarkResult:
    """Result of a trademark search"""
    query: str
    found: bool
    exact_matches: int = 0
    similar_matches: int = 0
    matches: list = None
    error: Optional[str] = None
    search_url: Optional[str] = None

    def __post_init__(self):
        if self.matches is None:
            self.matches = []


@dataclass
class TrademarkMatch:
    """A single trademark match"""
    application_number: str
    name: str
    status: str
    applicant: Optional[str] = None
    nice_classes: list = None
    registration_date: Optional[str] = None

    def __post_init__(self):
        if self.nice_classes is None:
            self.nice_classes = []


class EUIPOChecker:
    """
    Checks trademarks against EUIPO database.

    Usage with API (requires registration at https://dev.euipo.europa.eu/):
        checker = EUIPOChecker(
            client_id="your-client-id",
            client_secret="your-client-secret"
        )
        result = checker.check("Voltix")

    Usage without API (generates URLs for manual checking):
        checker = EUIPOChecker()
        result = checker.check("Voltix")
        print(result.search_url)  # Open in browser
    """

    def __init__(self,
                 client_id: Optional[str] = None,
                 client_secret: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 sandbox: bool = False):
        """
        Initialize the checker.

        Args:
            client_id: EUIPO API Client ID (X-IBM-Client-Id)
            client_secret: EUIPO API Client Secret
            cache_dir: Directory to cache results (default: ~/.cache/euipo)
            sandbox: Use sandbox environment for testing
        """
        cfg = get_setting("trademark.euipo", {}) or {}
        self.client_id = client_id or os.environ.get('EUIPO_CLIENT_ID')
        self.client_secret = client_secret or os.environ.get('EUIPO_CLIENT_SECRET')
        self.sandbox = sandbox
        self.api_base = cfg.get("api_base_sandbox") if sandbox else cfg.get("api_base")
        self.auth_url = cfg.get("auth_url")
        self.esearch_url = cfg.get("esearch_url")
        self.request_timeout_seconds = cfg.get("request_timeout_seconds")
        self.cache_ttl_seconds = cfg.get("cache_ttl_seconds")
        self.token_expiry_leeway_seconds = cfg.get("token_expiry_leeway_seconds")
        self.token_default_expires_seconds = cfg.get("token_default_expires_seconds")
        self.api_limit = cfg.get("api_limit")
        self.access_token_path = cfg.get("access_token_path")
        self.trademark_search_path = cfg.get("trademark_search_path")
        self.cache_hash_length = cfg.get("cache_hash_length")

        if not self.api_base:
            raise ValueError("trademark.euipo.api_base must be set in app.yaml")
        if not self.auth_url:
            raise ValueError("trademark.euipo.auth_url must be set in app.yaml")
        if not self.esearch_url:
            raise ValueError("trademark.euipo.esearch_url must be set in app.yaml")
        if self.request_timeout_seconds is None:
            raise ValueError("trademark.euipo.request_timeout_seconds must be set in app.yaml")
        if self.cache_ttl_seconds is None:
            raise ValueError("trademark.euipo.cache_ttl_seconds must be set in app.yaml")
        if self.token_expiry_leeway_seconds is None:
            raise ValueError("trademark.euipo.token_expiry_leeway_seconds must be set in app.yaml")
        if self.token_default_expires_seconds is None:
            raise ValueError("trademark.euipo.token_default_expires_seconds must be set in app.yaml")
        if self.api_limit is None:
            raise ValueError("trademark.euipo.api_limit must be set in app.yaml")
        if self.cache_hash_length is None:
            raise ValueError("trademark.euipo.cache_hash_length must be set in app.yaml")
        if not self.access_token_path:
            raise ValueError("trademark.euipo.access_token_path must be set in app.yaml")
        if not self.trademark_search_path:
            raise ValueError("trademark.euipo.trademark_search_path must be set in app.yaml")

        # Setup cache
        if cache_dir:
            self.cache_dir = resolve_path(cache_dir)
        else:
            self.cache_dir = resolve_path(cfg.get("cache_dir"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._access_token = None
        self._token_expiry = 0

    @property
    def has_api_access(self) -> bool:
        """Check if API credentials are available"""
        return bool(self.client_id and self.client_secret)

    def _get_cache_path(self, query: str, nice_classes: list = None) -> Path:
        """Get cache file path for a query and class filter"""
        classes_key = "all"
        if nice_classes:
            classes_key = ",".join(str(c) for c in sorted(nice_classes))
        key = f"{query.lower()}|{classes_key}|{self.api_base}"
        query_hash = hashlib.md5(key.encode()).hexdigest()[:self.cache_hash_length]
        return self.cache_dir / f"{query_hash}.json"

    def _load_from_cache(self, query: str, nice_classes: list = None) -> Optional[TrademarkResult]:
        """Load cached result if available and fresh (< 24h)"""
        cache_path = self._get_cache_path(query, nice_classes)
        if not cache_path.exists():
            return None

        try:
            data = json.loads(cache_path.read_text())
            if time.time() - data.get('timestamp', 0) > self.cache_ttl_seconds:
                return None

            # Ensure class filter matches
            if data.get('nice_classes') is not None:
                cached_classes = data.get('nice_classes')
                if nice_classes is None:
                    if cached_classes != "all":
                        return None
                else:
                    if cached_classes != sorted(nice_classes):
                        return None

            matches = [TrademarkMatch(**m) for m in data.get('matches', [])]
            return TrademarkResult(
                query=data['query'],
                found=data['found'],
                exact_matches=data.get('exact_matches', 0),
                similar_matches=data.get('similar_matches', 0),
                matches=matches,
                search_url=data.get('search_url')
            )
        except (json.JSONDecodeError, KeyError):
            return None

    def _save_to_cache(self, result: TrademarkResult, nice_classes: list = None):
        """Save result to cache"""
        cache_path = self._get_cache_path(result.query, nice_classes)
        data = {
            'query': result.query,
            'found': result.found,
            'exact_matches': result.exact_matches,
            'similar_matches': result.similar_matches,
            'matches': [
                {
                    'application_number': m.application_number,
                    'name': m.name,
                    'status': m.status,
                    'applicant': m.applicant,
                    'nice_classes': m.nice_classes,
                    'registration_date': m.registration_date
                }
                for m in result.matches
            ],
            'search_url': result.search_url,
            'nice_classes': sorted(nice_classes) if nice_classes else "all",
            'timestamp': time.time()
        }
        cache_path.write_text(json.dumps(data, indent=2))

    def _get_access_token(self) -> Optional[str]:
        """Get OAuth2 access token using client credentials flow"""
        if not self.has_api_access:
            return None

        # Return cached token if still valid
        if self._access_token and time.time() < self._token_expiry - self.token_expiry_leeway_seconds:
            return self._access_token

        try:
            conn = http.client.HTTPSConnection(
                self.auth_url,
                context=ssl.create_default_context(),
                timeout=self.request_timeout_seconds
            )

            # Client credentials grant
            payload = (
                f"grant_type=client_credentials"
                f"&client_id={quote_plus(self.client_id)}"
                f"&client_secret={quote_plus(self.client_secret)}"
            )

            headers = {
                'Content-Type': 'application/x-www-form-urlencoded'
            }

            conn.request(
                "POST",
                self.access_token_path,
                payload,
                headers
            )

            response = conn.getresponse()
            if response.status == 200:
                data = json.loads(response.read().decode())
                self._access_token = data.get('access_token')
                expires_in = data.get('expires_in', self.token_default_expires_seconds)
                self._token_expiry = time.time() + expires_in
                return self._access_token

        except Exception as e:
            print(f"Warning: Failed to get access token: {e}")

        return None

    def _search_api(self, query: str, nice_classes: list = None) -> TrademarkResult:
        """Search using the official EUIPO API

        Args:
            query: Brand name to search
            nice_classes: List of Nice class numbers to filter (None = search all classes)
        """
        search_url = self._generate_esearch_url(query)

        token = self._get_access_token()
        if not token:
            return TrademarkResult(
                query=query,
                found=False,
                error="Failed to obtain access token",
                search_url=search_url
            )

        try:
            conn = http.client.HTTPSConnection(
                self.api_base,
                context=ssl.create_default_context(),
                timeout=self.request_timeout_seconds
            )

            # RSQL query for verbal element (case-insensitive search)
            rsql_query = f"wordMarkSpecification.verbalElement==*{query}*"

            # Add Nice class filter if specified
            if nice_classes:
                class_conditions = ','.join(str(c) for c in nice_classes)
                rsql_query += f";niceClasses=in=({class_conditions})"

            encoded_query = quote_plus(rsql_query)

            headers = {
                'X-IBM-Client-Id': self.client_id,
                'Authorization': f'Bearer {token}',
                'Accept': 'application/json'
            }

            conn.request(
                "GET",
                f"{self.trademark_search_path}?query={encoded_query}&limit={self.api_limit}",
                headers=headers
            )

            response = conn.getresponse()

            if response.status == 200:
                data = json.loads(response.read().decode())
                return self._parse_api_response(query, data, search_url)
            elif response.status == 429:
                return TrademarkResult(
                    query=query,
                    found=False,
                    error="Rate limit exceeded. Try again later.",
                    search_url=search_url
                )
            elif response.status == 403:
                return TrademarkResult(
                    query=query,
                    found=False,
                    error="API subscription pending approval. Check manually.",
                    search_url=search_url
                )
            elif response.status == 401:
                return TrademarkResult(
                    query=query,
                    found=False,
                    error="API authentication failed. Check credentials.",
                    search_url=search_url
                )
            else:
                return TrademarkResult(
                    query=query,
                    found=False,
                    error=f"API error: {response.status}",
                    search_url=search_url
                )

        except Exception as e:
            return TrademarkResult(
                query=query,
                found=False,
                error=f"API request failed: {e}",
                search_url=search_url
            )

    def _parse_api_response(self, query: str, data: dict, search_url: str) -> TrademarkResult:
        """Parse EUIPO API response"""
        matches = []
        exact_count = 0
        similar_count = 0
        query_lower = query.lower()

        for item in data.get('content', []):
            # Extract trademark info
            verbal_element = ""
            word_mark_spec = item.get('wordMarkSpecification', {})
            if word_mark_spec:
                verbal_element = word_mark_spec.get('verbalElement', '')

            if not verbal_element:
                continue

            # Check if exact or similar match
            is_exact = verbal_element.lower() == query_lower
            if is_exact:
                exact_count += 1
            else:
                similar_count += 1

            # Extract applicant
            applicant = None
            applicants = item.get('applicants', [])
            if applicants:
                applicant = applicants[0].get('name', '')

            match = TrademarkMatch(
                application_number=item.get('applicationNumber', ''),
                name=verbal_element,
                status=item.get('status', ''),
                applicant=applicant,
                nice_classes=item.get('niceClasses', []),
                registration_date=item.get('registrationDate')
            )
            matches.append(match)

        return TrademarkResult(
            query=query,
            found=len(matches) > 0,
            exact_matches=exact_count,
            similar_matches=similar_count,
            matches=matches,
            search_url=search_url
        )

    def _generate_esearch_url(self, query: str) -> str:
        """Generate eSearch URL for manual checking"""
        encoded = quote_plus(query)
        return self.esearch_url.format(query=encoded)

    def check(self, name: str, nice_classes: list = None, use_cache: bool = True) -> TrademarkResult:
        """
        Check if a name is registered as EU trademark.

        Args:
            name: The brand name to check
            nice_classes: Nice classification codes to filter (None = search all classes)
            use_cache: Use cached results if available

        Returns:
            TrademarkResult with search results
        """
        # Check cache first
        if use_cache:
            cached = self._load_from_cache(name, nice_classes)
            if cached:
                return cached

        # Use API if available
        if self.has_api_access:
            result = self._search_api(name, nice_classes)
        else:
            # Fallback: just provide URL for manual checking
            result = TrademarkResult(
                query=name,
                found=False,  # Unknown without API
                error="No API credentials. Check manually.",
                search_url=self._generate_esearch_url(name)
            )

        # Cache result
        if use_cache and not result.error:
            self._save_to_cache(result, nice_classes)

        return result

    def check_batch(self, names: list[str],
                    delay: float = 0.5,
                    use_cache: bool = True) -> dict[str, TrademarkResult]:
        """
        Check multiple names with rate limiting.

        Args:
            names: List of brand names to check
            delay: Delay between API calls (seconds)
            use_cache: Use cached results if available

        Returns:
            Dictionary mapping names to results
        """
        results = {}

        for i, name in enumerate(names):
            results[name] = self.check(name, use_cache=use_cache)

            # Rate limiting (only for API calls, not cached)
            if self.has_api_access and i < len(names) - 1:
                time.sleep(delay)

        return results

    def filter_available(self, names: list[str]) -> tuple[list[str], list[str]]:
        """
        Filter names into available and potentially taken.

        Returns:
            Tuple of (likely_available, potentially_taken)
        """
        likely_available = []
        potentially_taken = []

        results = self.check_batch(names)

        for name, result in results.items():
            if result.exact_matches > 0:
                potentially_taken.append(name)
            elif result.found and result.similar_matches > 3:
                # Many similar matches might indicate crowded space
                potentially_taken.append(name)
            else:
                likely_available.append(name)

        return likely_available, potentially_taken


def check_single_name(name: str) -> TrademarkResult:
    """Convenience function to check a single name"""
    checker = EUIPOChecker()
    return checker.check(name)


def generate_check_urls(names: list[str]) -> list[tuple[str, str]]:
    """Generate eSearch URLs for a list of names"""
    checker = EUIPOChecker()
    return [(name, checker._generate_esearch_url(name)) for name in names]


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Check EUIPO trademark database')
    parser.add_argument('names', nargs='+', help='Brand names to check')
    parser.add_argument('--api-key', help='EUIPO API Client ID')
    parser.add_argument('--api-secret', help='EUIPO API Client Secret')
    parser.add_argument('--no-cache', action='store_true', help='Skip cache')
    parser.add_argument('--urls-only', action='store_true',
                        help='Only generate eSearch URLs')

    args = parser.parse_args()

    if args.urls_only:
        for name, url in generate_check_urls(args.names):
            print(f"{name}: {url}")
    else:
        checker = EUIPOChecker(
            client_id=args.api_key,
            client_secret=args.api_secret
        )

        for name in args.names:
            result = checker.check(name, use_cache=not args.no_cache)

            print(f"\n{'='*50}")
            print(f"Name: {name}")
            print(f"Found: {result.found}")

            if result.exact_matches:
                print(f"⚠️  EXACT MATCHES: {result.exact_matches}")
            if result.similar_matches:
                print(f"Similar matches: {result.similar_matches}")

            if result.error:
                print(f"Note: {result.error}")

            print(f"Manual check: {result.search_url}")

            if result.matches:
                top_matches = get_setting("trademark.cli_top_matches")
                if top_matches is None:
                    raise ValueError("trademark.cli_top_matches must be set in app.yaml")
                print("\nTop matches:")
                for m in result.matches[:top_matches]:
                    status_emoji = "✓" if m.status == "REGISTERED" else "○"
                    print(f"  {status_emoji} {m.name} ({m.status})")
                    if m.applicant:
                        print(f"    Applicant: {m.applicant}")
