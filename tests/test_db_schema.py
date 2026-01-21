import sqlite3

from namedb import BrandNameDB


def _get_columns(conn, table):
    cursor = conn.execute(f"PRAGMA table_info({table})")
    return {row[1] for row in cursor.fetchall()}


def test_trademark_matches_schema_includes_risk_fields(tmp_path):
    db_path = tmp_path / "test.db"
    BrandNameDB(db_path=str(db_path))

    with sqlite3.connect(db_path) as conn:
        cols = _get_columns(conn, "trademark_matches")

    assert "is_exact" in cols
    assert "risk_level" in cols
    assert "phonetic_similarity" in cols


def test_names_schema_includes_extended_scores(tmp_path):
    db_path = tmp_path / "test.db"
    BrandNameDB(db_path=str(db_path))

    with sqlite3.connect(db_path) as conn:
        cols = _get_columns(conn, "names")

    for col in ("score_cluster_quality", "score_ending_quality", "score_memorability"):
        assert col in cols


def test_save_and_read_trademark_match_risk_fields(tmp_path):
    db_path = tmp_path / "test.db"
    db = BrandNameDB(db_path=str(db_path))
    db.add("Voltix")

    ok = db.save_trademark_match(
        name="Voltix",
        region="US",
        match_name="Voltixx",
        match_serial="123",
        match_classes=[9],
        match_status="LIVE",
        similarity_score=0.8,
        is_exact=False,
        risk_level="HIGH",
        phonetic_similarity=0.92,
    )
    assert ok is True

    matches = db.get_trademark_matches("Voltix", region="US")
    assert matches
    match = matches[0]
    assert match["risk_level"] == "HIGH"
    assert match["phonetic_similarity"] == 0.92
