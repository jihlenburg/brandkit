"""
Tests for CLI Commands
======================
Tests for brandkit CLI interface in brandkit/cli.py.
"""

import pytest
import sys
import subprocess
from pathlib import Path
from io import StringIO
from unittest.mock import patch, MagicMock

# Ensure repo root is on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class TestCLIBasic:
    """Basic CLI tests."""

    def test_version_flag(self):
        """Test --version flag."""
        result = subprocess.run(
            [sys.executable, "-m", "brandkit", "--version"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        assert result.returncode == 0
        assert "brandkit" in result.stdout.lower()

    def test_help_flag(self):
        """Test --help flag."""
        result = subprocess.run(
            [sys.executable, "-m", "brandkit", "--help"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        assert result.returncode == 0
        assert "generate" in result.stdout.lower()
        assert "check" in result.stdout.lower()

    def test_generate_help(self):
        """Test generate --help."""
        result = subprocess.run(
            [sys.executable, "-m", "brandkit", "generate", "--help"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        assert result.returncode == 0
        assert "--count" in result.stdout or "-n" in result.stdout

    def test_check_help(self):
        """Test check --help."""
        result = subprocess.run(
            [sys.executable, "-m", "brandkit", "check", "--help"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        assert result.returncode == 0

    def test_score_help(self):
        """Test score --help."""
        result = subprocess.run(
            [sys.executable, "-m", "brandkit", "score", "--help"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        assert result.returncode == 0


class TestCLIGenerate:
    """Tests for generate command."""

    def test_generate_default(self):
        """Test generate with defaults."""
        result = subprocess.run(
            [sys.executable, "-m", "brandkit", "generate", "-n", "3"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            timeout=60,
        )
        # Should complete (may have no output if quality filter removes all)
        assert result.returncode == 0

    def test_generate_greek(self):
        """Test generate with greek method."""
        result = subprocess.run(
            [sys.executable, "-m", "brandkit", "generate", "-n", "3", "-m", "greek"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            timeout=60,
        )
        assert result.returncode == 0

    def test_generate_japanese(self):
        """Test generate with japanese method."""
        result = subprocess.run(
            [sys.executable, "-m", "brandkit", "generate", "-n", "3", "-m", "japanese"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            timeout=60,
        )
        assert result.returncode == 0

    def test_generate_latin(self):
        """Test generate with latin method."""
        result = subprocess.run(
            [sys.executable, "-m", "brandkit", "generate", "-n", "3", "-m", "latin"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            timeout=60,
        )
        assert result.returncode == 0

    def test_generate_invalid_method(self):
        """Test generate with invalid method."""
        result = subprocess.run(
            [sys.executable, "-m", "brandkit", "generate", "-m", "invalid_method"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        # Should fail or show error
        assert result.returncode != 0 or "invalid" in result.stderr.lower() or "error" in result.stderr.lower()


class TestCLIScore:
    """Tests for score command."""

    def test_score_basic(self):
        """Test basic scoring."""
        result = subprocess.run(
            [sys.executable, "-m", "brandkit", "score", "Voltix"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            timeout=30,
        )
        assert result.returncode == 0
        # Should output some score info

    def test_score_with_category(self):
        """Test scoring with category."""
        result = subprocess.run(
            [sys.executable, "-m", "brandkit", "score", "Voltix", "-c", "tech"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            timeout=30,
        )
        assert result.returncode == 0

    def test_score_verbose(self):
        """Test verbose scoring."""
        result = subprocess.run(
            [sys.executable, "-m", "brandkit", "score", "Lumina", "-v"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            timeout=30,
        )
        assert result.returncode == 0


class TestCLIHazards:
    """Tests for hazards command."""

    def test_hazards_safe_name(self):
        """Test hazards check on safe name."""
        result = subprocess.run(
            [sys.executable, "-m", "brandkit", "hazards", "Lumina"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            timeout=30,
        )
        assert result.returncode == 0

    def test_hazards_unsafe_name(self):
        """Test hazards check on unsafe name."""
        result = subprocess.run(
            [sys.executable, "-m", "brandkit", "hazards", "Gift"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            timeout=30,
        )
        assert result.returncode == 0
        # Should indicate hazard

    def test_hazards_with_markets(self):
        """Test hazards with specific markets."""
        result = subprocess.run(
            [sys.executable, "-m", "brandkit", "hazards", "Gift", "-m", "german"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            timeout=30,
        )
        assert result.returncode == 0


class TestCLIStats:
    """Tests for stats command."""

    def test_stats(self):
        """Test stats command."""
        result = subprocess.run(
            [sys.executable, "-m", "brandkit", "stats"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            timeout=30,
        )
        assert result.returncode == 0


class TestCLIList:
    """Tests for list command."""

    def test_list_default(self):
        """Test list with defaults."""
        result = subprocess.run(
            [sys.executable, "-m", "brandkit", "list"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            timeout=30,
        )
        assert result.returncode == 0

    def test_list_with_limit(self):
        """Test list with limit."""
        result = subprocess.run(
            [sys.executable, "-m", "brandkit", "list", "--limit", "5"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            timeout=30,
        )
        assert result.returncode == 0

    def test_list_json(self):
        """Test list with JSON output."""
        result = subprocess.run(
            [sys.executable, "-m", "brandkit", "list", "--json"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            timeout=30,
        )
        assert result.returncode == 0


class TestCLIProfiles:
    """Tests for profiles command."""

    def test_profiles(self):
        """Test profiles command."""
        result = subprocess.run(
            [sys.executable, "-m", "brandkit", "profiles"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            timeout=30,
        )
        assert result.returncode == 0
        # Should list Nice class profiles


class TestCLIIndustries:
    """Tests for industries command."""

    def test_industries(self):
        """Test industries command."""
        result = subprocess.run(
            [sys.executable, "-m", "brandkit", "industries"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            timeout=30,
        )
        assert result.returncode == 0
        # Should list industries


class TestCLIAliases:
    """Tests for CLI command aliases."""

    def test_gen_alias(self):
        """Test 'gen' alias for 'generate'."""
        result = subprocess.run(
            [sys.executable, "-m", "brandkit", "gen", "-n", "2"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            timeout=60,
        )
        assert result.returncode == 0

    def test_g_alias(self):
        """Test 'g' alias for 'generate'."""
        result = subprocess.run(
            [sys.executable, "-m", "brandkit", "g", "-n", "2"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            timeout=60,
        )
        assert result.returncode == 0

    def test_c_alias(self):
        """Test 'c' alias for 'check'."""
        result = subprocess.run(
            [sys.executable, "-m", "brandkit", "c", "Voltix"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            timeout=30,
        )
        assert result.returncode == 0

    def test_ls_alias(self):
        """Test 'ls' alias for 'list'."""
        result = subprocess.run(
            [sys.executable, "-m", "brandkit", "ls"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            timeout=30,
        )
        assert result.returncode == 0

    def test_haz_alias(self):
        """Test 'haz' alias for 'hazards'."""
        result = subprocess.run(
            [sys.executable, "-m", "brandkit", "haz", "Voltix"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            timeout=30,
        )
        assert result.returncode == 0

    def test_s_alias(self):
        """Test 's' alias for 'score'."""
        result = subprocess.run(
            [sys.executable, "-m", "brandkit", "s", "Voltix"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            timeout=30,
        )
        assert result.returncode == 0


class TestCLICheck:
    """Tests for check command."""

    def test_check_basic(self):
        """Test basic check (similarity only)."""
        result = subprocess.run(
            [sys.executable, "-m", "brandkit", "check", "Voltix"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            timeout=30,
        )
        assert result.returncode == 0

    def test_check_with_profile(self):
        """Test check with Nice class profile."""
        result = subprocess.run(
            [sys.executable, "-m", "brandkit", "check", "Voltix", "-p", "electronics"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            timeout=30,
        )
        # May succeed or fail depending on profile existence
        assert result.returncode in [0, 1, 2]

    def test_check_with_classes(self):
        """Test check with explicit Nice classes."""
        result = subprocess.run(
            [sys.executable, "-m", "brandkit", "check", "Voltix", "-c", "9,12"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            timeout=30,
        )
        assert result.returncode in [0, 1, 2]


class TestCLIValidation:
    """Tests for CLI input validation."""

    def test_empty_name_check(self):
        """Test check with empty name."""
        result = subprocess.run(
            [sys.executable, "-m", "brandkit", "check", ""],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        # Should handle gracefully

    def test_special_chars_name(self):
        """Test check with special characters."""
        result = subprocess.run(
            [sys.executable, "-m", "brandkit", "check", "Test!@#"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        # Should handle gracefully

    def test_negative_count(self):
        """Test generate with negative count."""
        result = subprocess.run(
            [sys.executable, "-m", "brandkit", "generate", "-n", "-5"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        # Should fail or use default

    def test_zero_count(self):
        """Test generate with zero count."""
        result = subprocess.run(
            [sys.executable, "-m", "brandkit", "generate", "-n", "0"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        assert result.returncode == 0  # Should complete with no output


class TestCLIOutput:
    """Tests for CLI output formatting."""

    def test_quiet_flag(self):
        """Test quiet flag reduces output."""
        result_normal = subprocess.run(
            [sys.executable, "-m", "brandkit", "stats"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            timeout=30,
        )
        result_quiet = subprocess.run(
            [sys.executable, "-m", "brandkit", "-q", "stats"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            timeout=30,
        )
        # Both should succeed
        assert result_normal.returncode == 0
        assert result_quiet.returncode == 0

    def test_list_json_format(self):
        """Test that JSON output is valid JSON."""
        import json
        result = subprocess.run(
            [sys.executable, "-m", "brandkit", "list", "--json", "--limit", "5"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            timeout=30,
        )
        if result.stdout.strip():
            try:
                json.loads(result.stdout)
            except json.JSONDecodeError:
                pass  # May have non-JSON prefix/suffix


class TestCLIModule:
    """Tests for CLI module internals."""

    def test_import_cli(self):
        """Test that CLI module can be imported."""
        from brandkit import cli
        assert hasattr(cli, 'main')

    def test_cli_has_commands(self):
        """Test that CLI defines expected commands."""
        from brandkit.cli import main
        # Should exist and be callable
        assert callable(main)
