from types import SimpleNamespace

import euipo_checker
import rapidapi_checker


def test_euipo_cache_separates_nice_classes(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake_search(self, query, nice_classes=None):
        calls["n"] += 1
        return euipo_checker.TrademarkResult(
            query=query,
            found=False,
            exact_matches=0,
            similar_matches=0,
            matches=[],
            error=None,
            search_url="https://example.test",
        )

    monkeypatch.setattr(euipo_checker.EUIPOChecker, "_search_api", fake_search)

    checker = euipo_checker.EUIPOChecker(
        client_id="x",
        client_secret="y",
        cache_dir=tmp_path,
    )

    checker.check("Voltix", nice_classes=[9], use_cache=True)
    checker.check("Voltix", nice_classes=[12], use_cache=True)
    checker.check("Voltix", nice_classes=[9], use_cache=True)

    # Different nice classes should not hit the same cache entry
    assert calls["n"] == 2


def test_rapidapi_cache_preserves_nice_classes_for_filtering(tmp_path, monkeypatch):
    def fake_search(self, query):
        match = rapidapi_checker.TrademarkMatch(
            name="Voltix",
            serial_number="123",
            registration_number=None,
            status="LIVE",
            owner=None,
            filing_date=None,
            source="USPTO",
            nice_classes=[9],
        )
        return rapidapi_checker.TrademarkResult(
            query=query,
            found=True,
            exact_matches=1,
            similar_matches=0,
            matches=[match],
            error=None,
        )

    monkeypatch.setattr(rapidapi_checker.RapidAPIChecker, "_search_api", fake_search)

    checker = rapidapi_checker.RapidAPIChecker(api_key="key", cache_dir=tmp_path)
    checker.check("Voltix", nice_classes=[9], use_cache=True)

    # New checker reads from cache; class filter should exclude non-overlapping classes
    checker2 = rapidapi_checker.RapidAPIChecker(api_key="key", cache_dir=tmp_path)
    result = checker2.check("Voltix", nice_classes=[12], use_cache=True)
    assert result.found is False
    assert result.matches == []

    result_ok = checker2.check("Voltix", nice_classes=[9], use_cache=True)
    assert result_ok.found is True
    assert len(result_ok.matches) == 1
