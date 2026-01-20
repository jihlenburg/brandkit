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

    API_BASE = "api.euipo.europa.eu"
    AUTH_URL = "euipo.europa.eu"
    ESEARCH_URL = "https://euipo.europa.eu/eSearch/#basic/{query}"

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
        self.client_id = client_id or os.environ.get('EUIPO_CLIENT_ID')
        self.client_secret = client_secret or os.environ.get('EUIPO_CLIENT_SECRET')
        self.sandbox = sandbox

        if sandbox:
            self.API_BASE = "api-sandbox.euipo.europa.eu"

        # Setup cache
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path.home() / ".cache" / "euipo"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._access_token = None
        self._token_expiry = 0

    @property
    def has_api_access(self) -> bool:
        """Check if API credentials are available"""
        return bool(self.client_id and self.client_secret)

    def _get_cache_path(self, query: str) -> Path:
        """Get cache file path for a query"""
        query_hash = hashlib.md5(query.lower().encode()).hexdigest()[:12]
        return self.cache_dir / f"{query_hash}.json"

    def _load_from_cache(self, query: str) -> Optional[TrademarkResult]:
        """Load cached result if available and fresh (< 24h)"""
        cache_path = self._get_cache_path(query)
        if not cache_path.exists():
            return None

        try:
            data = json.loads(cache_path.read_text())
            # Check if cache is fresh (24 hours)
            if time.time() - data.get('timestamp', 0) > 86400:
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

    def _save_to_cache(self, result: TrademarkResult):
        """Save result to cache"""
        cache_path = self._get_cache_path(result.query)
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
            'timestamp': time.time()
        }
        cache_path.write_text(json.dumps(data, indent=2))

    def _get_access_token(self) -> Optional[str]:
        """Get OAuth2 access token using client credentials flow"""
        if not self.has_api_access:
            return None

        # Return cached token if still valid
        if self._access_token and time.time() < self._token_expiry - 60:
            return self._access_token

        try:
            conn = http.client.HTTPSConnection(
                self.AUTH_URL,
                context=ssl.create_default_context()
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
                "/cas-server-webapp/oidc/accessToken",
                payload,
                headers
            )

            response = conn.getresponse()
            if response.status == 200:
                data = json.loads(response.read().decode())
                self._access_token = data.get('access_token')
                expires_in = data.get('expires_in', 3600)
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
                self.API_BASE,
                context=ssl.create_default_context()
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
                f"/trademark-search/trademarks?query={encoded_query}&limit=20",
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
        return self.ESEARCH_URL.format(query=encoded)

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
        # Check cache first (note: cache doesn't distinguish by class filter)
        if use_cache:
            cached = self._load_from_cache(name)
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
            self._save_to_cache(result)

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
                print("\nTop matches:")
                for m in result.matches[:5]:
                    status_emoji = "✓" if m.status == "REGISTERED" else "○"
                    print(f"  {status_emoji} {m.name} ({m.status})")
                    if m.applicant:
                        print(f"    Applicant: {m.applicant}")
