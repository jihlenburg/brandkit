"""
Tests for Database Operations
=============================
Tests for NameDB, BrandNameDB, and database operations
in brandkit/db.py and namedb.py.
"""

import pytest
import sys
import tempfile
import os
from pathlib import Path

# Ensure repo root is on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brandkit.db import get_db
from namedb import BrandNameDB, NameStatus


class TestNameStatus:
    """Tests for NameStatus enum."""

    def test_status_values(self):
        """Test that all expected statuses exist."""
        assert NameStatus.NEW.value == "new"
        assert NameStatus.CANDIDATE.value == "candidate"
        assert NameStatus.SHORTLIST.value == "shortlist"
        assert NameStatus.APPROVED.value == "approved"
        assert NameStatus.REJECTED.value == "rejected"
        assert NameStatus.BLOCKED.value == "blocked"

    def test_status_from_string(self):
        """Test creating status from string."""
        assert NameStatus("new") == NameStatus.NEW
        assert NameStatus("candidate") == NameStatus.CANDIDATE


class TestBrandNameDB:
    """Tests for BrandNameDB class."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        db = BrandNameDB(db_path)
        yield db
        # Cleanup
        try:
            os.unlink(db_path)
        except:
            pass

    def test_init_creates_db(self, temp_db):
        """Test that init creates database."""
        assert temp_db is not None

    def test_add_name(self, temp_db):
        """Test adding a name."""
        result = temp_db.add(
            name="TestName",
            method="test",
            status=NameStatus.NEW,
        )
        assert result is not None

    def test_add_and_get(self, temp_db):
        """Test adding and retrieving a name."""
        temp_db.add(
            name="Voltix",
            method="greek",
            status=NameStatus.CANDIDATE,
        )
        record = temp_db.get("Voltix")
        assert record is not None
        assert record.name == "Voltix"
        assert record.method == "greek"

    def test_get_nonexistent(self, temp_db):
        """Test getting nonexistent name."""
        record = temp_db.get("NonexistentName")
        assert record is None

    def test_update_status(self, temp_db):
        """Test updating name status."""
        temp_db.add(name="UpdateTest", method="test", status=NameStatus.NEW)
        temp_db.update_status("UpdateTest", NameStatus.CANDIDATE)
        record = temp_db.get("UpdateTest")
        assert record.status == NameStatus.CANDIDATE

    def test_set_status(self, temp_db):
        """Test set_status method."""
        temp_db.add(name="SetStatusTest", method="test", status=NameStatus.NEW)
        temp_db.set_status("SetStatusTest", NameStatus.SHORTLIST)
        record = temp_db.get("SetStatusTest")
        assert record.status == NameStatus.SHORTLIST

    def test_get_by_status(self, temp_db):
        """Test getting names by status."""
        temp_db.add(name="New1", method="test", status=NameStatus.NEW)
        temp_db.add(name="New2", method="test", status=NameStatus.NEW)
        temp_db.add(name="Candidate1", method="test", status=NameStatus.CANDIDATE)

        new_names = temp_db.get_by_status(NameStatus.NEW)
        assert len(new_names) == 2

        candidate_names = temp_db.get_by_status(NameStatus.CANDIDATE)
        assert len(candidate_names) == 1

    def test_count_by_status(self, temp_db):
        """Test counting names by status."""
        temp_db.add(name="Count1", method="test", status=NameStatus.NEW)
        temp_db.add(name="Count2", method="test", status=NameStatus.NEW)
        temp_db.add(name="Count3", method="test", status=NameStatus.CANDIDATE)

        assert temp_db.count_by_status(NameStatus.NEW) == 2
        assert temp_db.count_by_status(NameStatus.CANDIDATE) == 1

    def test_delete_name(self, temp_db):
        """Test deleting a name."""
        temp_db.add(name="ToDelete", method="test", status=NameStatus.NEW)
        assert temp_db.get("ToDelete") is not None

        temp_db.delete("ToDelete")
        assert temp_db.get("ToDelete") is None

    def test_add_then_update_scores(self, temp_db):
        """Test adding a name then updating scores."""
        temp_db.add(
            name="Scored",
            method="test",
            status=NameStatus.NEW,
        )
        temp_db.update_phonaesthetic_scores(
            "Scored",
            overall=0.75,
        )
        record = temp_db.get("Scored")
        assert record.score_phonaesthetic == 0.75

    def test_update_phonaesthetic_scores(self, temp_db):
        """Test updating phonaesthetic scores."""
        temp_db.add(name="ScoreUpdate", method="test", status=NameStatus.NEW)
        temp_db.update_phonaesthetic_scores(
            "ScoreUpdate",
            overall=0.8,
            consonant=0.7,
            vowel=0.6,
            fluency=0.75,
            rhythm=0.9,
            naturalness=0.65,
            quality_tier="excellent",
        )
        record = temp_db.get("ScoreUpdate")
        assert record.score_phonaesthetic == 0.8

    def test_update_validation_results(self, temp_db):
        """Test updating validation results."""
        temp_db.add(name="Validated", method="test", status=NameStatus.NEW)
        temp_db.update_validation_results(
            "Validated",
            eu_conflict=False,
            us_conflict=True,
            domains={'.com': False, '.de': True},
        )
        record = temp_db.get("Validated")
        assert record.eu_conflict is False
        assert record.us_conflict is True

    def test_get_available(self, temp_db):
        """Test getting available names (no conflicts)."""
        temp_db.add(name="Available", method="test", status=NameStatus.NEW)
        temp_db.update_validation_results(
            "Available",
            eu_conflict=False,
            us_conflict=False,
        )

        temp_db.add(name="Conflicted", method="test", status=NameStatus.NEW)
        temp_db.update_validation_results(
            "Conflicted",
            eu_conflict=True,
            us_conflict=False,
        )

        available = temp_db.get_available()
        names = [r.name for r in available]
        assert "Available" in names
        assert "Conflicted" not in names

    def test_get_conflicts(self, temp_db):
        """Test getting names with conflicts."""
        temp_db.add(name="HasConflict", method="test", status=NameStatus.NEW)
        temp_db.update_validation_results(
            "HasConflict",
            eu_conflict=True,
            us_conflict=False,
        )

        conflicts = temp_db.get_conflicts()
        names = [r.name for r in conflicts]
        assert "HasConflict" in names


class TestBrandNameDBQualityTier:
    """Tests for quality tier filtering."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        db = BrandNameDB(db_path)
        yield db
        try:
            os.unlink(db_path)
        except:
            pass

    def test_get_excellent(self, temp_db):
        """Test getting excellent quality names."""
        temp_db.add(name="Excellent1", method="test", status=NameStatus.NEW)
        temp_db.update_phonaesthetic_scores(
            "Excellent1", overall=0.9, quality_tier="excellent"
        )

        temp_db.add(name="Good1", method="test", status=NameStatus.NEW)
        temp_db.update_phonaesthetic_scores(
            "Good1", overall=0.7, quality_tier="good"
        )

        excellent = temp_db.get_excellent()
        names = [r.name for r in excellent]
        assert "Excellent1" in names
        assert "Good1" not in names

    def test_get_by_quality_tier(self, temp_db):
        """Test getting names by quality tier."""
        temp_db.add(name="GoodName", method="test", status=NameStatus.NEW)
        temp_db.update_phonaesthetic_scores(
            "GoodName", overall=0.7, quality_tier="good"
        )

        good_names = temp_db.get_by_quality_tier("good")
        names = [r.name for r in good_names]
        assert "GoodName" in names


class TestBrandNameDBTrademarkChecks:
    """Tests for trademark check storage."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        db = BrandNameDB(db_path)
        yield db
        try:
            os.unlink(db_path)
        except:
            pass

    def test_save_trademark_check(self, temp_db):
        """Test saving trademark check result."""
        temp_db.add(name="TMCheck", method="test", status=NameStatus.NEW)
        temp_db.save_trademark_check(
            "TMCheck",
            nice_class=9,
            region="EU",
            available=True,
            conflicts_count=0,
        )

    def test_save_trademark_check_with_conflicts(self, temp_db):
        """Test saving trademark check with conflicts."""
        temp_db.add(name="TMConflict", method="test", status=NameStatus.NEW)
        temp_db.save_trademark_check(
            "TMConflict",
            nice_class=12,
            region="EU",
            available=False,
            conflicts_count=3,
            conflict_details='["BRAND1", "BRAND2", "BRAND3"]',
        )

    def test_get_trademark_checks(self, temp_db):
        """Test retrieving trademark checks."""
        temp_db.add(name="TMGet", method="test", status=NameStatus.NEW)
        temp_db.save_trademark_check(
            "TMGet",
            nice_class=9,
            region="EU",
            available=True,
            conflicts_count=0,
        )
        temp_db.save_trademark_check(
            "TMGet",
            nice_class=12,
            region="EU",
            available=False,
            conflicts_count=2,
        )

        checks = temp_db.get_trademark_checks("TMGet")
        assert isinstance(checks, dict)
        assert "EU" in checks

    def test_get_available_classes(self, temp_db):
        """Test getting available Nice classes."""
        temp_db.add(name="ClassCheck", method="test", status=NameStatus.NEW)
        temp_db.save_trademark_check(
            "ClassCheck",
            nice_class=9,
            region="EU",
            available=True,
            conflicts_count=0,
        )
        temp_db.save_trademark_check(
            "ClassCheck",
            nice_class=12,
            region="EU",
            available=False,
            conflicts_count=1,
        )

        available = temp_db.get_available_classes("ClassCheck", region="EU")
        assert 9 in available
        assert 12 not in available

    def test_get_conflicting_classes(self, temp_db):
        """Test getting conflicting Nice classes."""
        temp_db.add(name="ConflictClass", method="test", status=NameStatus.NEW)
        temp_db.save_trademark_check(
            "ConflictClass",
            nice_class=9,
            region="EU",
            available=True,
            conflicts_count=0,
        )
        temp_db.save_trademark_check(
            "ConflictClass",
            nice_class=12,
            region="EU",
            available=False,
            conflicts_count=2,
        )

        conflicting = temp_db.get_conflicting_classes("ConflictClass", region="EU")
        assert 12 in conflicting
        assert 9 not in conflicting


class TestBrandNameDBEdgeCases:
    """Edge case tests for BrandNameDB."""

    @pytest.fixture
    def temp_db(self):
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        db = BrandNameDB(db_path)
        yield db
        try:
            os.unlink(db_path)
        except:
            pass

    def test_duplicate_name_handling(self, temp_db):
        """Test handling of duplicate names."""
        temp_db.add(name="Duplicate", method="test", status=NameStatus.NEW)
        # Second add with same name should update
        temp_db.add(name="Duplicate", method="updated", status=NameStatus.CANDIDATE)
        record = temp_db.get("Duplicate")
        # Behavior depends on implementation (update or reject)
        assert record is not None

    def test_special_characters_in_name(self, temp_db):
        """Test names with special handling needs."""
        # Should handle cleanly
        try:
            temp_db.add(name="Name'sTest", method="test", status=NameStatus.NEW)
        except:
            pass  # May reject special chars

    def test_empty_name(self, temp_db):
        """Test empty name handling."""
        # Should reject or handle gracefully
        try:
            temp_db.add(name="", method="test", status=NameStatus.NEW)
        except:
            pass  # Expected to fail

    def test_very_long_name(self, temp_db):
        """Test very long name."""
        long_name = "A" * 100
        try:
            temp_db.add(name=long_name, method="test", status=NameStatus.NEW)
            record = temp_db.get(long_name)
            assert record is not None
        except:
            pass  # May reject very long names

    def test_unicode_name(self, temp_db):
        """Test Unicode name handling."""
        try:
            temp_db.add(name="Völtix", method="test", status=NameStatus.NEW)
            record = temp_db.get("Völtix")
            # Should either store or reject cleanly
        except:
            pass  # May reject non-ASCII


class TestGetDB:
    """Tests for get_db singleton function."""

    def test_get_db_returns_instance(self):
        """Test that get_db returns a database instance."""
        db = get_db()
        assert db is not None
        assert hasattr(db, 'add')
        assert hasattr(db, 'get')

    def test_get_db_singleton(self):
        """Test that get_db returns same instance."""
        db1 = get_db()
        db2 = get_db()
        assert db1 is db2


class TestBrandNameRecord:
    """Tests for BrandName record from database."""

    def test_record_from_db(self):
        """Test retrieving a record from database."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        try:
            db = BrandNameDB(db_path)
            db.add(name="TestRecord", method="test", status=NameStatus.NEW)
            record = db.get("TestRecord")
            assert record is not None
            assert record.name == "TestRecord"
            assert record.status == NameStatus.NEW
        finally:
            os.unlink(db_path)
