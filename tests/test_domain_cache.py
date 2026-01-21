import domain_checker


def test_domain_cache_varies_by_tld_set(tmp_path, monkeypatch):
    calls_a = {"n": 0}

    def fake_check(self, name, tld):
        calls_a["n"] += 1
        return domain_checker.DomainResult(
            name=name,
            domain=f"{name.lower()}.{tld}",
            tld=tld,
            available=True,
            has_website=False,
            error=None,
        )

    checker_a = domain_checker.DomainChecker(
        tlds=["com"],
        cache_dir=tmp_path,
        parallel=False,
    )
    monkeypatch.setattr(domain_checker.DomainChecker, "_check_single_domain", fake_check)
    checker_a.check("Voltix", use_cache=True)
    assert calls_a["n"] == 1

    calls_b = {"n": 0}

    def fake_check_b(self, name, tld):
        calls_b["n"] += 1
        return domain_checker.DomainResult(
            name=name,
            domain=f"{name.lower()}.{tld}",
            tld=tld,
            available=True,
            has_website=False,
            error=None,
        )

    checker_b = domain_checker.DomainChecker(
        tlds=["net"],
        cache_dir=tmp_path,
        parallel=False,
    )
    monkeypatch.setattr(domain_checker.DomainChecker, "_check_single_domain", fake_check_b)
    checker_b.check("Voltix", use_cache=True)
    assert calls_b["n"] == 1


def test_domain_cache_hit_same_tlds(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake_check(self, name, tld):
        calls["n"] += 1
        return domain_checker.DomainResult(
            name=name,
            domain=f"{name.lower()}.{tld}",
            tld=tld,
            available=True,
            has_website=False,
            error=None,
        )

    checker = domain_checker.DomainChecker(
        tlds=["com"],
        cache_dir=tmp_path,
        parallel=False,
    )
    monkeypatch.setattr(domain_checker.DomainChecker, "_check_single_domain", fake_check)
    checker.check("Voltix", use_cache=True)
    assert calls["n"] == 1

    # New checker should hit cache and not call _check_single_domain
    checker2 = domain_checker.DomainChecker(
        tlds=["com"],
        cache_dir=tmp_path,
        parallel=False,
    )
    monkeypatch.setattr(domain_checker.DomainChecker, "_check_single_domain", fake_check)
    checker2.check("Voltix", use_cache=True)
    assert calls["n"] == 1
