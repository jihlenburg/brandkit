#!/usr/bin/env python3
"""
RapidAPI Trademark Lookup Checker
==================================
Searches USPTO and WIPO trademark databases via RapidAPI.

Free tier: 1000 requests/month
Sign up: https://rapidapi.com/Creativesdev/api/trademark-lookup-api

Usage:
    checker = RapidAPIChecker(api_key="your-key")
    result = checker.check("Voltix")
"""

import os
import json
import time
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import http.client
import ssl

from settings import get_setting, resolve_path

@dataclass
class TrademarkMatch:
    """A single trademark match"""
    name: str
    serial_number: str
    registration_number: Optional[str]
    status: str
    owner: Optional[str] = None
    filing_date: Optional[str] = None
    source: str = "USPTO"  # or "WIPO"
    nice_classes: list = None  # Nice classification codes if available

    def __post_init__(self):
        if self.nice_classes is None:
            self.nice_classes = []


@dataclass
class TrademarkResult:
    """Result of a trademark search"""
    query: str
    found: bool
    exact_matches: int = 0
    similar_matches: int = 0
    matches: list = None
    error: Optional[str] = None

    def __post_init__(self):
        if self.matches is None:
            self.matches = []

    def filter_by_classes(self, nice_classes: list) -> 'TrademarkResult':
        """
        Filter matches to only those in specified Nice classes.

        Args:
            nice_classes: List of Nice class numbers to keep

        Returns:
            New TrademarkResult with filtered matches
        """
        if not nice_classes:
            return self

        filtered = []
        exact = 0
        similar = 0
        query_lower = self.query.lower()

        for match in self.matches:
            # If match has no class info, keep it (conservative approach)
            if not match.nice_classes:
                filtered.append(match)
                if match.name.lower() == query_lower:
                    exact += 1
                else:
                    similar += 1
            # If match has class info, check overlap
            elif any(c in nice_classes for c in match.nice_classes):
                filtered.append(match)
                if match.name.lower() == query_lower:
                    exact += 1
                else:
                    similar += 1

        return TrademarkResult(
            query=self.query,
            found=len(filtered) > 0,
            exact_matches=exact,
            similar_matches=similar,
            matches=filtered,
            error=self.error
        )


class RapidAPIChecker:
    """
    Checks trademarks via RapidAPI Trademark Lookup.

    Searches USPTO (US) and WIPO (international) databases.
    Free tier: 1000 requests/month.

    Usage:
        checker = RapidAPIChecker(api_key="your-rapidapi-key")
        result = checker.check("Voltix")
    """

    def __init__(self,
                 api_key: Optional[str] = None,
                 cache_dir: Optional[str] = None):
        """
        Initialize the checker.

        Args:
            api_key: RapidAPI key (or set RAPIDAPI_KEY env var)
            cache_dir: Directory to cache results
        """
        cfg = get_setting("trademark.rapidapi", {}) or {}
        self.api_key = api_key or os.environ.get('RAPIDAPI_KEY')
        self.api_host = cfg.get("api_host")
        self.request_timeout_seconds = cfg.get("request_timeout_seconds")
        self.cache_ttl_seconds = cfg.get("cache_ttl_seconds")
        self.max_results = cfg.get("max_results")
        self.search_page = cfg.get("search", {}).get("page")
        self.search_count = cfg.get("search", {}).get("count")
        self.search_path_template = cfg.get("search", {}).get("path_template")
        self.cache_hash_length = cfg.get("cache_hash_length")

        if not self.api_host:
            raise ValueError("trademark.rapidapi.api_host must be set in app.yaml")
        if self.request_timeout_seconds is None:
            raise ValueError("trademark.rapidapi.request_timeout_seconds must be set in app.yaml")
        if self.cache_ttl_seconds is None:
            raise ValueError("trademark.rapidapi.cache_ttl_seconds must be set in app.yaml")
        if self.max_results is None:
            raise ValueError("trademark.rapidapi.max_results must be set in app.yaml")
        if self.cache_hash_length is None:
            raise ValueError("trademark.rapidapi.cache_hash_length must be set in app.yaml")
        if self.search_page is None or self.search_count is None or not self.search_path_template:
            raise ValueError("trademark.rapidapi.search settings must be set in app.yaml")

        # Setup cache
        if cache_dir:
            self.cache_dir = resolve_path(cache_dir)
        else:
            self.cache_dir = resolve_path(cfg.get("cache_dir"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def has_api_access(self) -> bool:
        """Check if API key is available"""
        return bool(self.api_key)

    def _get_cache_path(self, query: str) -> Path:
        """Get cache file path for a query"""
        query_hash = hashlib.md5(query.lower().encode()).hexdigest()[:self.cache_hash_length]
        return self.cache_dir / f"{query_hash}.json"

    def _load_from_cache(self, query: str) -> Optional[TrademarkResult]:
        """Load cached result if available and fresh (< 7 days)"""
        cache_path = self._get_cache_path(query)
        if not cache_path.exists():
            return None

        try:
            data = json.loads(cache_path.read_text())
            if time.time() - data.get('timestamp', 0) > self.cache_ttl_seconds:
                return None

            matches = []
            for m in data.get('matches', []):
                if 'nice_classes' not in m:
                    m['nice_classes'] = []
                matches.append(TrademarkMatch(**m))
            return TrademarkResult(
                query=data['query'],
                found=data['found'],
                exact_matches=data.get('exact_matches', 0),
                similar_matches=data.get('similar_matches', 0),
                matches=matches
            )
        except (json.JSONDecodeError, KeyError, TypeError):
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
                    'name': m.name,
                    'serial_number': m.serial_number,
                    'registration_number': m.registration_number,
                    'status': m.status,
                    'owner': m.owner,
                    'filing_date': m.filing_date,
                    'source': m.source,
                    'nice_classes': m.nice_classes,
                }
                for m in result.matches
            ],
            'timestamp': time.time()
        }
        cache_path.write_text(json.dumps(data, indent=2))

    def _search_api(self, query: str) -> TrademarkResult:
        """Search using RapidAPI"""
        if not self.has_api_access:
            return TrademarkResult(
                query=query,
                found=False,
                error="No RapidAPI key. Set RAPIDAPI_KEY env var."
            )

        try:
            conn = http.client.HTTPSConnection(
                self.api_host,
                context=ssl.create_default_context(),
                timeout=self.request_timeout_seconds
            )

            # URL encode the query
            encoded_query = query.replace(' ', '%20')

            headers = {
                'X-RapidAPI-Key': self.api_key,
                'X-RapidAPI-Host': self.api_host,
                'Accept': 'application/json'
            }

            # Search endpoint: /{name}/namesearch/{page}/{count}
            conn.request(
                "GET",
                self.search_path_template.format(
                    query=encoded_query,
                    page=self.search_page,
                    count=self.search_count,
                ),
                headers=headers
            )

            response = conn.getresponse()

            if response.status == 200:
                data = json.loads(response.read().decode())
                return self._parse_response(query, data)
            elif response.status == 429:
                return TrademarkResult(
                    query=query,
                    found=False,
                    error="Rate limit exceeded. Free tier: 1000/month."
                )
            elif response.status == 401 or response.status == 403:
                return TrademarkResult(
                    query=query,
                    found=False,
                    error="Invalid RapidAPI key."
                )
            else:
                body = response.read().decode()[:200]
                return TrademarkResult(
                    query=query,
                    found=False,
                    error=f"API error {response.status}: {body}"
                )

        except Exception as e:
            return TrademarkResult(
                query=query,
                found=False,
                error=f"Request failed: {e}"
            )

    def _parse_response(self, query: str, data: dict) -> TrademarkResult:
        """Parse RapidAPI response"""
        matches = []
        exact_count = 0
        similar_count = 0
        query_lower = query.lower()

        # Handle the actual API response format
        items = []
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            # Primary format: {"list": [...], "totalResults": N}
            items = data.get('list', [])
            if not items:
                items = data.get('items', data.get('results', data.get('trademarks', [])))
            if not items and 'markIdentification' in data:
                # Single result
                items = [data]

        for item in items[:int(self.max_results)]:
            # Extract name - API uses markIdentification
            name = (item.get('markIdentification') or
                   item.get('wordmark') or
                   item.get('name') or
                   item.get('mark_identification') or '')

            if not name:
                continue

            # Check if exact or similar (case-insensitive)
            is_exact = name.lower() == query_lower
            if is_exact:
                exact_count += 1
            else:
                similar_count += 1

            # Extract status from events if available
            status = "Unknown"
            events = item.get('events', [])
            if events:
                # Get most recent event
                status = events[0].get('descriptionText', 'Unknown')

            # Extract owner
            owner = None
            owners = item.get('owners', [])
            if owners:
                owner = owners[0].get('partyName')

            # Try to extract Nice classes (various possible field names)
            nice_classes = []
            class_data = (item.get('niceClasses') or
                         item.get('nice_classes') or
                         item.get('internationalClasses') or
                         item.get('classifications') or [])
            if class_data:
                for c in class_data:
                    if isinstance(c, int):
                        nice_classes.append(c)
                    elif isinstance(c, str) and c.isdigit():
                        nice_classes.append(int(c))
                    elif isinstance(c, dict):
                        code = c.get('code') or c.get('class') or c.get('classNumber')
                        if code and str(code).isdigit():
                            nice_classes.append(int(code))

            match = TrademarkMatch(
                name=name,
                serial_number=str(item.get('serialNumber', item.get('serial_number', ''))),
                registration_number=str(item.get('registrationNumber', item.get('registration_number', ''))) if item.get('registrationNumber') else None,
                status=status,
                owner=owner,
                filing_date=item.get('filingDate', item.get('filing_date')),
                source="USPTO",
                nice_classes=nice_classes
            )
            matches.append(match)

        return TrademarkResult(
            query=query,
            found=len(matches) > 0,
            exact_matches=exact_count,
            similar_matches=similar_count,
            matches=matches
        )

    def check(self, name: str, nice_classes: list = None, use_cache: bool = True) -> TrademarkResult:
        """
        Check if a name is registered as trademark.

        Args:
            name: The brand name to check
            nice_classes: Nice classification codes to filter (None = return all matches)
                         Note: Filtering is post-search since API doesn't support class queries
            use_cache: Use cached results if available

        Returns:
            TrademarkResult with search results
        """
        # Check cache first
        if use_cache:
            cached = self._load_from_cache(name)
            if cached:
                # Apply class filter to cached results if specified
                if nice_classes:
                    return cached.filter_by_classes(nice_classes)
                return cached

        # Search API
        result = self._search_api(name)

        # Cache successful results (cache unfiltered for reuse)
        if use_cache and not result.error:
            self._save_to_cache(result)

        # Apply class filter if specified
        if nice_classes:
            return result.filter_by_classes(nice_classes)

        return result

    def check_batch(self, names: list[str],
                    delay: float = 0.2,
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

            # Rate limiting
            if self.has_api_access and i < len(names) - 1:
                time.sleep(delay)

        return results


# Singleton
_default_checker = None

def get_rapidapi_checker() -> RapidAPIChecker:
    """Get default checker instance"""
    global _default_checker
    if _default_checker is None:
        _default_checker = RapidAPIChecker()
    return _default_checker


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Check trademarks via RapidAPI')
    parser.add_argument('names', nargs='+', help='Brand names to check')
    parser.add_argument('--api-key', help='RapidAPI key')
    parser.add_argument('--no-cache', action='store_true', help='Skip cache')

    args = parser.parse_args()

    checker = RapidAPIChecker(api_key=args.api_key)

    if not checker.has_api_access:
        print("No API key. Set RAPIDAPI_KEY environment variable.")
        print("Get free key at: https://rapidapi.com/Creativesdev/api/trademark-lookup-api")
        exit(1)

    for name in args.names:
        result = checker.check(name, use_cache=not args.no_cache)

        print(f"\n{'='*50}")
        print(f"Name: {name}")
        print(f"Found: {result.found}")

        if result.exact_matches:
            print(f"  EXACT MATCHES: {result.exact_matches}")
        if result.similar_matches:
            print(f"  Similar matches: {result.similar_matches}")

        if result.error:
            print(f"Error: {result.error}")

        if result.matches:
            print("\nTop matches:")
            top_matches = get_setting("trademark.cli_top_matches")
            if top_matches is None:
                raise ValueError("trademark.cli_top_matches must be set in app.yaml")
            for m in result.matches[:top_matches]:
                status_icon = "R" if "REGISTERED" in m.status.upper() else "P"
                print(f"  [{status_icon}] {m.name} - {m.status}")
                if m.owner:
                    print(f"      Owner: {m.owner}")
