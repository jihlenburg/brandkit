#!/usr/bin/env python3
"""
Brand Name Project Database
============================
Comprehensive database for the entire brand naming process:
- Track all generated candidates
- Manage status workflow (new → candidate → approved/rejected/blocked)
- Store comments, notes, and thoughts
- Record EUIPO check results
- Maintain history and audit trail

Storage: SQLite database (brandnames.db)
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Union
from enum import Enum

from settings import get_setting, resolve_path


class NameStatus(Enum):
    """Status of a brand name in the workflow"""
    NEW = "new"                    # Just generated, not reviewed
    CANDIDATE = "candidate"        # Under consideration
    SHORTLIST = "shortlist"        # Made it to shortlist
    APPROVED = "approved"          # Final approval
    REJECTED = "rejected"          # Rejected (but not blocked)
    BLOCKED = "blocked"            # Blocked from future generation


class BlockReason(Enum):
    """Reasons for blocking a name"""
    EUIPO_CONFLICT = "euipo_conflict"
    EUIPO_SIMILAR = "euipo_similar"
    USPTO_CONFLICT = "uspto_conflict"
    PHONETIC_ISSUE_DE = "phonetic_issue_de"
    PHONETIC_ISSUE_EN = "phonetic_issue_en"
    NEGATIVE_CONNOTATION = "negative_connotation"
    CLIENT_REJECTED = "client_rejected"
    LEGAL_ISSUE = "legal_issue"
    UNPRONOUNCEABLE = "unpronounceable"
    OTHER = "other"


class QualityTier(Enum):
    """Quality tier based on phonaesthetic scoring"""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"


@dataclass
class Comment:
    """A comment on a brand name"""
    id: int
    text: str
    author: Optional[str]
    created_at: str

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'text': self.text,
            'author': self.author,
            'created_at': self.created_at,
        }


@dataclass
class BrandName:
    """A brand name entry with full metadata"""
    id: Optional[int]
    name: str
    status: NameStatus = NameStatus.NEW

    # Phonaesthetic scores (research-backed)
    score_phonaesthetic: Optional[float] = None  # Overall phonaesthetic score
    score_consonant: Optional[float] = None      # Consonant quality
    score_vowel: Optional[float] = None          # Vowel quality
    score_fluency: Optional[float] = None        # Processing fluency
    score_rhythm: Optional[float] = None         # Rhythm score
    score_naturalness: Optional[float] = None    # Phonotactic naturalness
    score_cluster_quality: Optional[float] = None  # Cluster quality
    score_ending_quality: Optional[float] = None   # Ending quality
    score_memorability: Optional[float] = None     # Memorability
    quality_tier: Optional[QualityTier] = None   # excellent/good/acceptable/poor

    # Generation info
    method: Optional[str] = None          # 'rules', cultural methods
    semantic_associations: Optional[str] = None
    generation_details: Optional[str] = None
    semantic_meaning: Optional[str] = None  # Brand story/meaning explanation

    # Validation results
    validated_at: Optional[str] = None    # When validation was last done
    eu_conflict: Optional[bool] = None    # True if EU trademark conflict
    us_conflict: Optional[bool] = None    # True if US trademark conflict
    domains_available: Optional[str] = None  # JSON dict of domain availability

    # Blocking (reason is a string - can be BlockReason.value or custom like "pronounceability:...")
    block_reason: Optional[str] = None
    block_notes: Optional[str] = None

    # Timestamps
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    # Comments (loaded separately)
    comments: List[Comment] = field(default_factory=list)

    # Backward compatibility property
    @property
    def score(self) -> Optional[float]:
        """Alias for score_phonaesthetic for backward compatibility."""
        return self.score_phonaesthetic

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'name': self.name,
            'status': self.status.value,
            'score': self.score_phonaesthetic,
            'phonaesthetic': {
                'overall': self.score_phonaesthetic,
                'consonant': self.score_consonant,
                'vowel': self.score_vowel,
                'fluency': self.score_fluency,
                'rhythm': self.score_rhythm,
                'naturalness': self.score_naturalness,
                'cluster_quality': self.score_cluster_quality,
                'ending_quality': self.score_ending_quality,
                'memorability': self.score_memorability,
                'quality_tier': self.quality_tier.value if self.quality_tier else None,
            },
            'validation': {
                'validated_at': self.validated_at,
                'eu_conflict': self.eu_conflict,
                'us_conflict': self.us_conflict,
                'domains_available': json.loads(self.domains_available) if self.domains_available else None,
            },
            'method': self.method,
            'semantic_associations': self.semantic_associations,
            'semantic_meaning': self.semantic_meaning,
            'block': {
                'reason': self.block_reason,
                'notes': self.block_notes,
            } if self.block_reason else None,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'comments': [c.to_dict() for c in self.comments],
        }


class BrandNameDB:
    """
    SQLite database for brand name project management.

    Usage:
        db = BrandNameDB()

        # Add a new name
        db.add("Fluxon", score=0.95, method="rules")

        # Update status
        db.set_status("Fluxon", NameStatus.CANDIDATE)

        # Add comment
        db.add_comment("Fluxon", "Klingt technisch aber warm", author="Max")

        # Block a name
        db.block("Voltex", BlockReason.EUIPO_CONFLICT, notes="3 Treffer")

        # Get candidates
        candidates = db.get_by_status(NameStatus.CANDIDATE)

        # Search
        results = db.search("flux")
    """

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = get_setting("paths.data_db")
            if not db_path:
                raise ValueError("paths.data_db must be set in app.yaml")
            db_path = resolve_path(db_path)

        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self):
        """Create database tables and run migrations"""
        with sqlite3.connect(self.db_path) as conn:
            # Main names table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS names (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    name_lower TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL DEFAULT 'new',

                    -- Phonaesthetic scores
                    score_phonaesthetic REAL,
                    score_consonant REAL,
                    score_vowel REAL,
                    score_fluency REAL,
                    score_rhythm REAL,
                    score_naturalness REAL,
                    score_cluster_quality REAL,
                    score_ending_quality REAL,
                    score_memorability REAL,
                    quality_tier TEXT,

                    -- Generation
                    method TEXT,
                    semantic_associations TEXT,
                    generation_details TEXT,
                    semantic_meaning TEXT,

                    -- Validation results
                    validated_at TEXT,
                    eu_conflict INTEGER,
                    us_conflict INTEGER,
                    domains_available TEXT,

                    -- Blocking
                    block_reason TEXT,
                    block_notes TEXT,

                    -- Timestamps
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            # Run migrations for existing databases
            self._migrate(conn)

            # Comments table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS comments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name_id INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    author TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (name_id) REFERENCES names(id) ON DELETE CASCADE
                )
            """)

            # History/audit table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name_id INTEGER NOT NULL,
                    action TEXT NOT NULL,
                    old_value TEXT,
                    new_value TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (name_id) REFERENCES names(id) ON DELETE CASCADE
                )
            """)

            # Tags table (for categorization)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tags (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name_id INTEGER NOT NULL,
                    tag TEXT NOT NULL,
                    FOREIGN KEY (name_id) REFERENCES names(id) ON DELETE CASCADE,
                    UNIQUE(name_id, tag)
                )
            """)

            # Trademark checks table (class-specific results)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trademark_checks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name_id INTEGER NOT NULL,
                    nice_class INTEGER NOT NULL,
                    region TEXT NOT NULL,
                    checked_at TEXT NOT NULL,
                    available INTEGER,
                    conflicts_count INTEGER DEFAULT 0,
                    conflict_details TEXT,
                    FOREIGN KEY (name_id) REFERENCES names(id) ON DELETE CASCADE,
                    UNIQUE(name_id, nice_class, region)
                )
            """)

            # Trademark matches table (individual conflicting marks)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trademark_matches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name_id INTEGER NOT NULL,
                    region TEXT NOT NULL,
                    match_name TEXT NOT NULL,
                    match_serial TEXT,
                    match_classes TEXT,
                    match_status TEXT,
                    similarity_score REAL,
                    is_exact INTEGER,
                    found_at TEXT NOT NULL,
                    risk_level TEXT,
                    phonetic_similarity REAL,
                    FOREIGN KEY (name_id) REFERENCES names(id) ON DELETE CASCADE
                )
            """)

            # Indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_name_lower ON names(name_lower)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON names(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_score ON names(score_phonaesthetic DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tm_name_class ON trademark_checks(name_id, nice_class)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tm_matches_name ON trademark_matches(name_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tm_matches_region ON trademark_matches(region)")

            conn.commit()

    def _now(self) -> str:
        return datetime.now().isoformat()

    def _migrate(self, conn):
        """Add new columns to existing databases (v0.4.0 migration)"""
        # Get existing columns
        cursor = conn.execute("PRAGMA table_info(names)")
        existing_cols = {row[1] for row in cursor.fetchall()}

        # New columns to add (v0.4.0+)
        new_columns = [
            ("score_phonaesthetic", "REAL"),
            ("score_consonant", "REAL"),
            ("score_vowel", "REAL"),
            ("score_fluency", "REAL"),
            ("score_rhythm", "REAL"),
            ("score_naturalness", "REAL"),
            ("score_cluster_quality", "REAL"),
            ("score_ending_quality", "REAL"),
            ("score_memorability", "REAL"),
            ("quality_tier", "TEXT"),
            ("validated_at", "TEXT"),
            ("eu_conflict", "INTEGER"),
            ("us_conflict", "INTEGER"),
            ("domains_available", "TEXT"),
            # v0.5.0 - semantic meaning
            ("semantic_meaning", "TEXT"),
        ]

        for col_name, col_type in new_columns:
            if col_name not in existing_cols:
                try:
                    conn.execute(f"ALTER TABLE names ADD COLUMN {col_name} {col_type}")
                except sqlite3.OperationalError:
                    pass  # Column already exists

        # Trademark matches table migrations
        cursor = conn.execute("PRAGMA table_info(trademark_matches)")
        tm_cols = {row[1] for row in cursor.fetchall()}
        tm_new_columns = [
            ("is_exact", "INTEGER"),
            ("risk_level", "TEXT"),
            ("phonetic_similarity", "REAL"),
        ]
        for col_name, col_type in tm_new_columns:
            if col_name not in tm_cols:
                try:
                    conn.execute(f"ALTER TABLE trademark_matches ADD COLUMN {col_name} {col_type}")
                except sqlite3.OperationalError:
                    pass

        conn.commit()

    # =========================================================================
    # CRUD Operations
    # =========================================================================

    def add(self,
            name: str,
            method: str = None,
            semantic_associations: str = None,
            generation_details: str = None,
            semantic_meaning: str = None,
            status: NameStatus = NameStatus.NEW) -> Optional[int]:
        """
        Add a new name to the database.

        Returns:
            ID of the new entry, or None if name already exists
        """
        now = self._now()
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    INSERT INTO names (
                        name, name_lower, status,
                        method, semantic_associations, generation_details,
                        semantic_meaning,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    name, name.lower(), status.value,
                    method, semantic_associations, generation_details,
                    semantic_meaning,
                    now, now
                ))
                conn.commit()
                return cursor.lastrowid
        except sqlite3.IntegrityError:
            return None

    def get(self, name: str) -> Optional[BrandName]:
        """Get a name by its text"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM names WHERE name_lower = ?
            """, (name.lower(),))
            row = cursor.fetchone()

            if not row:
                return None

            brand = self._row_to_brand(row)

            # Load comments
            cursor = conn.execute("""
                SELECT id, text, author, created_at
                FROM comments WHERE name_id = ?
                ORDER BY created_at ASC
            """, (row['id'],))
            brand.comments = [
                Comment(id=r['id'], text=r['text'], author=r['author'], created_at=r['created_at'])
                for r in cursor.fetchall()
            ]

            return brand

    def _row_to_brand(self, row) -> BrandName:
        """Convert database row to BrandName object"""
        # Handle columns that may not exist in older databases
        def safe_get(key, default=None):
            try:
                return row[key]
            except (IndexError, KeyError):
                return default

        return BrandName(
            id=row['id'],
            name=row['name'],
            status=NameStatus(row['status']),
            # Phonaesthetic scores
            score_phonaesthetic=safe_get('score_phonaesthetic'),
            score_consonant=safe_get('score_consonant'),
            score_vowel=safe_get('score_vowel'),
            score_fluency=safe_get('score_fluency'),
            score_rhythm=safe_get('score_rhythm'),
            score_naturalness=safe_get('score_naturalness'),
            score_cluster_quality=safe_get('score_cluster_quality'),
            score_ending_quality=safe_get('score_ending_quality'),
            score_memorability=safe_get('score_memorability'),
            quality_tier=QualityTier(safe_get('quality_tier')) if safe_get('quality_tier') else None,
            # Generation
            method=safe_get('method'),
            semantic_associations=safe_get('semantic_associations'),
            generation_details=safe_get('generation_details'),
            semantic_meaning=safe_get('semantic_meaning'),
            # Validation results
            validated_at=safe_get('validated_at'),
            eu_conflict=bool(safe_get('eu_conflict')) if safe_get('eu_conflict') is not None else None,
            us_conflict=bool(safe_get('us_conflict')) if safe_get('us_conflict') is not None else None,
            domains_available=safe_get('domains_available'),
            # Blocking (stored as string, not enum)
            block_reason=safe_get('block_reason'),
            block_notes=safe_get('block_notes'),
            # Timestamps
            created_at=row['created_at'],
            updated_at=row['updated_at'],
        )

    def update(self, name: str, **kwargs) -> bool:
        """
        Update name attributes.

        Example:
            db.update("Fluxon", score=0.96, method="hybrid")
            db.update("Fluxon", score_phonaesthetic=0.62, quality_tier="excellent")
        """
        if not kwargs:
            return False

        # Build SET clause
        allowed_fields = {
            # Phonaesthetic scores
            'score_phonaesthetic', 'score_consonant', 'score_vowel',
            'score_fluency', 'score_rhythm', 'score_naturalness',
            'score_cluster_quality', 'score_ending_quality', 'score_memorability',
            'quality_tier',
            # Generation
            'method', 'semantic_associations', 'generation_details', 'semantic_meaning',
            # Validation
            'validated_at', 'eu_conflict', 'us_conflict', 'domains_available',
            # Blocking
            'block_reason', 'block_notes'
        }

        updates = []
        values = []
        for key, value in kwargs.items():
            if key in allowed_fields:
                updates.append(f"{key} = ?")
                # Handle enum types (convert to string values)
                if key == 'block_reason':
                    # Accept both BlockReason enum and plain strings
                    if isinstance(value, BlockReason):
                        values.append(value.value)
                    else:
                        values.append(str(value) if value else None)
                elif key == 'quality_tier' and isinstance(value, QualityTier):
                    values.append(value.value)
                elif key == 'quality_tier' and isinstance(value, str):
                    values.append(value)  # Allow string for convenience
                elif key in ('eu_conflict', 'us_conflict') and isinstance(value, bool):
                    values.append(1 if value else 0)
                else:
                    values.append(value)

        if not updates:
            return False

        updates.append("updated_at = ?")
        values.append(self._now())
        values.append(name.lower())

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(f"""
                UPDATE names SET {', '.join(updates)}
                WHERE name_lower = ?
            """, values)
            conn.commit()
            return cursor.rowcount > 0

    def delete(self, name: str) -> bool:
        """Delete a name completely"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                DELETE FROM names WHERE name_lower = ?
            """, (name.lower(),))
            conn.commit()
            return cursor.rowcount > 0

    # =========================================================================
    # Status Management
    # =========================================================================

    def set_status(self, name: str, status: NameStatus) -> bool:
        """Change the status of a name"""
        brand = self.get(name)
        if not brand:
            return False

        old_status = brand.status.value

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE names SET status = ?, updated_at = ?
                WHERE name_lower = ?
            """, (status.value, self._now(), name.lower()))

            # Record in history
            conn.execute("""
                INSERT INTO history (name_id, action, old_value, new_value, created_at)
                VALUES (?, 'status_change', ?, ?, ?)
            """, (brand.id, old_status, status.value, self._now()))

            conn.commit()
            return True

    def block(self, name: str, reason: Union[BlockReason, str], notes: str = None) -> bool:
        """Block a name from future generation.

        Args:
            name: The name to block
            reason: BlockReason enum or custom string (e.g., "pronounceability:awkward_start:sv")
            notes: Optional additional notes
        """
        # Convert enum to string value if needed
        reason_str = reason.value if isinstance(reason, BlockReason) else str(reason)

        brand = self.get(name)

        if brand:
            # Update existing
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE names
                    SET status = ?, block_reason = ?, block_notes = ?, updated_at = ?
                    WHERE name_lower = ?
                """, (NameStatus.BLOCKED.value, reason_str, notes, self._now(), name.lower()))
                conn.commit()
            return True
        else:
            # Add new as blocked
            self.add(name, status=NameStatus.BLOCKED)
            return self.update(name, block_reason=reason_str, block_notes=notes)

    def unblock(self, name: str) -> bool:
        """Remove block from a name"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                UPDATE names
                SET status = ?, block_reason = NULL, block_notes = NULL, updated_at = ?
                WHERE name_lower = ? AND status = ?
            """, (NameStatus.REJECTED.value, self._now(), name.lower(), NameStatus.BLOCKED.value))
            conn.commit()
            return cursor.rowcount > 0

    def is_blocked(self, name: str) -> bool:
        """Check if a name is blocked"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT 1 FROM names WHERE name_lower = ? AND status = ?
            """, (name.lower(), NameStatus.BLOCKED.value))
            return cursor.fetchone() is not None

    # =========================================================================
    # Comments
    # =========================================================================

    def add_comment(self, name: str, text: str, author: str = None) -> Optional[int]:
        """Add a comment to a name"""
        brand = self.get(name)
        if not brand:
            return None

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO comments (name_id, text, author, created_at)
                VALUES (?, ?, ?, ?)
            """, (brand.id, text, author, self._now()))

            conn.execute("""
                UPDATE names SET updated_at = ? WHERE id = ?
            """, (self._now(), brand.id))

            conn.commit()
            return cursor.lastrowid

    def get_comments(self, name: str) -> List[Comment]:
        """Get all comments for a name"""
        brand = self.get(name)
        return brand.comments if brand else []

    # =========================================================================
    # Tags
    # =========================================================================

    def add_tag(self, name: str, tag: str) -> bool:
        """Add a tag to a name"""
        brand = self.get(name)
        if not brand:
            return False

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO tags (name_id, tag) VALUES (?, ?)
                """, (brand.id, tag.lower()))
                conn.commit()
                return True
        except sqlite3.IntegrityError:
            return False

    def remove_tag(self, name: str, tag: str) -> bool:
        """Remove a tag from a name"""
        brand = self.get(name)
        if not brand:
            return False

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                DELETE FROM tags WHERE name_id = ? AND tag = ?
            """, (brand.id, tag.lower()))
            conn.commit()
            return cursor.rowcount > 0

    def get_tags(self, name: str) -> List[str]:
        """Get all tags for a name"""
        brand = self.get(name)
        if not brand:
            return []

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT tag FROM tags WHERE name_id = ?
            """, (brand.id,))
            return [row[0] for row in cursor.fetchall()]

    # =========================================================================
    # Trademark Checks (Class-Specific)
    # =========================================================================

    def save_trademark_check(self,
                            name: str,
                            nice_class: int,
                            region: str,
                            available: bool,
                            conflicts_count: int = 0,
                            conflict_details: str = None) -> bool:
        """
        Save a trademark check result for a specific Nice class.

        Args:
            name: Brand name
            nice_class: Nice classification code (1-45)
            region: Region code ('EU', 'US', 'WIPO')
            available: True if no conflicts, False if conflicts found
            conflicts_count: Number of conflicting trademarks
            conflict_details: JSON string with match details

        Returns:
            True if saved successfully
        """
        brand = self.get(name)
        if not brand:
            return False

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO trademark_checks
                    (name_id, nice_class, region, checked_at, available, conflicts_count, conflict_details)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(name_id, nice_class, region)
                DO UPDATE SET
                    checked_at = excluded.checked_at,
                    available = excluded.available,
                    conflicts_count = excluded.conflicts_count,
                    conflict_details = excluded.conflict_details
            """, (brand.id, nice_class, region.upper(), self._now(),
                  1 if available else 0, conflicts_count, conflict_details))
            conn.commit()
            return True

    def save_trademark_checks_batch(self,
                                   name: str,
                                   results: dict) -> bool:
        """
        Save multiple trademark check results at once.

        Args:
            name: Brand name
            results: Dict with structure:
                {
                    'EU': {9: {'available': True, 'conflicts': 0}, 12: {...}},
                    'US': {9: {'available': False, 'conflicts': 3, 'details': '...'}}
                }

        Returns:
            True if saved successfully
        """
        brand = self.get(name)
        if not brand:
            return False

        with sqlite3.connect(self.db_path) as conn:
            for region, classes in results.items():
                for nice_class, data in classes.items():
                    conn.execute("""
                        INSERT INTO trademark_checks
                            (name_id, nice_class, region, checked_at, available, conflicts_count, conflict_details)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(name_id, nice_class, region)
                        DO UPDATE SET
                            checked_at = excluded.checked_at,
                            available = excluded.available,
                            conflicts_count = excluded.conflicts_count,
                            conflict_details = excluded.conflict_details
                    """, (brand.id, int(nice_class), region.upper(), self._now(),
                          1 if data.get('available') else 0,
                          data.get('conflicts', 0),
                          data.get('details')))
            conn.commit()
            return True

    def get_trademark_checks(self, name: str) -> dict:
        """
        Get all trademark check results for a name.

        Returns:
            Dict with structure:
            {
                'EU': {9: {'available': True, 'conflicts': 0, 'checked_at': '...'}, ...},
                'US': {...}
            }
        """
        brand = self.get(name)
        if not brand:
            return {}

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT nice_class, region, available, conflicts_count, conflict_details, checked_at
                FROM trademark_checks
                WHERE name_id = ?
                ORDER BY region, nice_class
            """, (brand.id,))

            results = {}
            for row in cursor.fetchall():
                nice_class, region, available, conflicts, details, checked_at = row
                if region not in results:
                    results[region] = {}
                results[region][nice_class] = {
                    'available': bool(available),
                    'conflicts': conflicts,
                    'details': details,
                    'checked_at': checked_at,
                }
            return results

    def save_trademark_match(self,
                            name: str,
                            region: str,
                            match_name: str,
                            match_serial: str = None,
                            match_classes: List[int] = None,
                            match_status: str = None,
                            similarity_score: float = None,
                            is_exact: bool = False,
                            risk_level: str = None,
                            phonetic_similarity: float = None) -> bool:
        """
        Save a conflicting trademark match.

        Args:
            name: The brand name being checked
            region: 'US' or 'EU'
            match_name: Name of the conflicting trademark
            match_serial: USPTO serial number or EUIPO number
            match_classes: List of Nice classes for this match
            match_status: Status (LIVE, DEAD, etc.)
            similarity_score: Optional similarity score
            is_exact: True if exact match, False if similar/phonetic match
            risk_level: Risk assessment (CRITICAL, HIGH, MEDIUM, LOW, UNKNOWN)
            phonetic_similarity: Phonetic similarity score (0.0-1.0)

        Returns:
            True if saved successfully
        """
        brand = self.get(name)
        if not brand:
            return False

        classes_json = json.dumps(match_classes) if match_classes else None
        now = self._now()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO trademark_matches (
                    name_id, region, match_name, match_serial,
                    match_classes, match_status, similarity_score, is_exact, found_at,
                    risk_level, phonetic_similarity
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (brand.id, region, match_name, match_serial,
                  classes_json, match_status, similarity_score, 1 if is_exact else 0, now,
                  risk_level, phonetic_similarity))
            conn.commit()
            return True

    def save_trademark_matches_batch(self,
                                     name: str,
                                     region: str,
                                     matches: List[dict]) -> int:
        """
        Save multiple trademark matches for a name.

        Args:
            name: The brand name being checked
            region: 'US' or 'EU'
            matches: List of dicts with keys: match_name, match_serial, match_classes,
                     match_status, similarity_score, is_exact, risk_level, phonetic_similarity

        Returns:
            Number of matches saved
        """
        brand = self.get(name)
        if not brand:
            return 0

        now = self._now()
        saved = 0

        with sqlite3.connect(self.db_path) as conn:
            for m in matches:
                classes_json = json.dumps(m.get('match_classes')) if m.get('match_classes') else None
                is_exact = 1 if m.get('is_exact') else 0
                conn.execute("""
                    INSERT INTO trademark_matches (
                        name_id, region, match_name, match_serial,
                        match_classes, match_status, similarity_score, is_exact, found_at,
                        risk_level, phonetic_similarity
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (brand.id, region, m.get('match_name'), m.get('match_serial'),
                      classes_json, m.get('match_status'), m.get('similarity_score'), is_exact, now,
                      m.get('risk_level'), m.get('phonetic_similarity')))
                saved += 1
            conn.commit()
            return saved

    def get_trademark_matches(self, name: str, region: str = None) -> List[dict]:
        """
        Get all trademark matches for a name.

        Args:
            name: Brand name
            region: Filter by region ('US', 'EU') or None for all

        Returns:
            List of match dicts with keys: region, match_name, match_serial, match_classes,
            match_status, similarity_score, is_exact, found_at, risk_level, phonetic_similarity
        """
        brand = self.get(name)
        if not brand:
            return []

        with sqlite3.connect(self.db_path) as conn:
            if region:
                cursor = conn.execute("""
                    SELECT region, match_name, match_serial, match_classes, match_status,
                           similarity_score, is_exact, found_at, risk_level, phonetic_similarity
                    FROM trademark_matches
                    WHERE name_id = ? AND region = ?
                    ORDER BY
                        CASE risk_level
                            WHEN 'CRITICAL' THEN 0
                            WHEN 'HIGH' THEN 1
                            WHEN 'MEDIUM' THEN 2
                            WHEN 'LOW' THEN 3
                            ELSE 4
                        END,
                        is_exact DESC, phonetic_similarity DESC
                """, (brand.id, region))
            else:
                cursor = conn.execute("""
                    SELECT region, match_name, match_serial, match_classes, match_status,
                           similarity_score, is_exact, found_at, risk_level, phonetic_similarity
                    FROM trademark_matches
                    WHERE name_id = ?
                    ORDER BY
                        region,
                        CASE risk_level
                            WHEN 'CRITICAL' THEN 0
                            WHEN 'HIGH' THEN 1
                            WHEN 'MEDIUM' THEN 2
                            WHEN 'LOW' THEN 3
                            ELSE 4
                        END,
                        is_exact DESC, phonetic_similarity DESC
                """, (brand.id,))

            matches = []
            for row in cursor.fetchall():
                matches.append({
                    'region': row[0],
                    'match_name': row[1],
                    'match_serial': row[2],
                    'match_classes': json.loads(row[3]) if row[3] else None,
                    'match_status': row[4],
                    'similarity_score': row[5],
                    'is_exact': bool(row[6]),
                    'found_at': row[7],
                    'risk_level': row[8],
                    'phonetic_similarity': row[9],
                })
            return matches

    def clear_trademark_matches(self, name: str, region: str = None) -> bool:
        """
        Clear trademark matches for a name (before re-checking).

        Args:
            name: Brand name
            region: Clear only for specific region, or None for all

        Returns:
            True if cleared successfully
        """
        brand = self.get(name)
        if not brand:
            return False

        with sqlite3.connect(self.db_path) as conn:
            if region:
                conn.execute("DELETE FROM trademark_matches WHERE name_id = ? AND region = ?", (brand.id, region))
            else:
                conn.execute("DELETE FROM trademark_matches WHERE name_id = ?", (brand.id,))
            conn.commit()
            return True

    def update_trademark_match_risk(self,
                                    match_id: int,
                                    risk_level: str = None,
                                    phonetic_similarity: float = None) -> bool:
        """
        Update risk assessment fields for a trademark match.

        Args:
            match_id: ID of the trademark match
            risk_level: Risk level (CRITICAL, HIGH, MEDIUM, LOW, UNKNOWN)
            phonetic_similarity: Phonetic similarity score (0.0-1.0)

        Returns:
            True if updated successfully
        """
        updates = []
        values = []

        if risk_level is not None:
            updates.append("risk_level = ?")
            values.append(risk_level)
        if phonetic_similarity is not None:
            updates.append("phonetic_similarity = ?")
            values.append(phonetic_similarity)

        if not updates:
            return False

        values.append(match_id)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(f"""
                UPDATE trademark_matches
                SET {', '.join(updates)}
                WHERE id = ?
            """, values)
            conn.commit()
            return cursor.rowcount > 0

    def recalculate_trademark_risks(self, name: str = None) -> int:
        """
        Recalculate risk_level and phonetic_similarity for all trademark matches.

        Uses the phonetic_similarity module to compute similarity scores
        and calculate risk levels based on trademark status.

        Args:
            name: Optional - only recalculate for this name. If None, recalculates all.

        Returns:
            Number of matches updated
        """
        from brandkit.phonetic_similarity import (
            compute_phonetic_similarity,
            calculate_risk_level
        )

        updated = 0

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Get all matches that need updating
            if name:
                brand = self.get(name)
                if not brand:
                    return 0
                cursor = conn.execute("""
                    SELECT tm.id, tm.match_name, tm.match_status, tm.is_exact, n.name
                    FROM trademark_matches tm
                    JOIN names n ON tm.name_id = n.id
                    WHERE n.name_lower = ?
                """, (name.lower(),))
            else:
                cursor = conn.execute("""
                    SELECT tm.id, tm.match_name, tm.match_status, tm.is_exact, n.name
                    FROM trademark_matches tm
                    JOIN names n ON tm.name_id = n.id
                """)

            matches = cursor.fetchall()

            for row in matches:
                match_id = row['id']
                match_name = row['match_name']
                match_status = row['match_status']
                is_exact = bool(row['is_exact'])
                brand_name = row['name']

                # Calculate phonetic similarity
                phon_sim = compute_phonetic_similarity(brand_name, match_name)

                # Calculate risk level
                risk = calculate_risk_level(
                    match_status=match_status,
                    is_exact=is_exact,
                    phonetic_similarity=phon_sim
                )

                # Update the record
                conn.execute("""
                    UPDATE trademark_matches
                    SET risk_level = ?, phonetic_similarity = ?
                    WHERE id = ?
                """, (risk, phon_sim, match_id))
                updated += 1

            conn.commit()

        return updated

    def get_high_risk_matches(self, limit: int = None) -> List[dict]:
        """
        Get trademark matches with CRITICAL or HIGH risk level.

        Returns:
            List of dicts with name, match details, and risk info
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            query = """
                SELECT n.name, tm.region, tm.match_name, tm.match_serial,
                       tm.match_classes, tm.match_status, tm.is_exact,
                       tm.risk_level, tm.phonetic_similarity
                FROM trademark_matches tm
                JOIN names n ON tm.name_id = n.id
                WHERE tm.risk_level IN ('CRITICAL', 'HIGH')
                ORDER BY
                    CASE tm.risk_level WHEN 'CRITICAL' THEN 0 ELSE 1 END,
                    tm.phonetic_similarity DESC
            """
            if limit:
                query += f" LIMIT {limit}"

            cursor = conn.execute(query)

            results = []
            for row in cursor.fetchall():
                results.append({
                    'name': row['name'],
                    'region': row['region'],
                    'match_name': row['match_name'],
                    'match_serial': row['match_serial'],
                    'match_classes': json.loads(row['match_classes']) if row['match_classes'] else None,
                    'match_status': row['match_status'],
                    'is_exact': bool(row['is_exact']),
                    'risk_level': row['risk_level'],
                    'phonetic_similarity': row['phonetic_similarity'],
                })
            return results

    def get_available_classes(self, name: str, region: str = None) -> List[int]:
        """
        Get Nice classes where the name is available (no conflicts).

        Args:
            name: Brand name
            region: Filter by region ('EU', 'US') or None for all

        Returns:
            List of Nice class numbers that are available
        """
        brand = self.get(name)
        if not brand:
            return []

        with sqlite3.connect(self.db_path) as conn:
            if region:
                cursor = conn.execute("""
                    SELECT DISTINCT nice_class FROM trademark_checks
                    WHERE name_id = ? AND region = ? AND available = 1
                    ORDER BY nice_class
                """, (brand.id, region.upper()))
            else:
                cursor = conn.execute("""
                    SELECT DISTINCT nice_class FROM trademark_checks
                    WHERE name_id = ? AND available = 1
                    ORDER BY nice_class
                """, (brand.id,))
            return [row[0] for row in cursor.fetchall()]

    def get_conflicting_classes(self, name: str, region: str = None) -> List[int]:
        """
        Get Nice classes where the name has conflicts.

        Args:
            name: Brand name
            region: Filter by region ('EU', 'US') or None for all

        Returns:
            List of Nice class numbers with conflicts
        """
        brand = self.get(name)
        if not brand:
            return []

        with sqlite3.connect(self.db_path) as conn:
            if region:
                cursor = conn.execute("""
                    SELECT DISTINCT nice_class FROM trademark_checks
                    WHERE name_id = ? AND region = ? AND available = 0
                    ORDER BY nice_class
                """, (brand.id, region.upper()))
            else:
                cursor = conn.execute("""
                    SELECT DISTINCT nice_class FROM trademark_checks
                    WHERE name_id = ? AND available = 0
                    ORDER BY nice_class
                """, (brand.id,))
            return [row[0] for row in cursor.fetchall()]

    # =========================================================================
    # Queries
    # =========================================================================

    def get_by_status(self, status: NameStatus, limit: int = None) -> List[BrandName]:
        """Get all names with a specific status"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            query = "SELECT * FROM names WHERE status = ? ORDER BY score_phonaesthetic DESC"
            if limit:
                query += f" LIMIT {limit}"
            cursor = conn.execute(query, (status.value,))
            return [self._row_to_brand(row) for row in cursor.fetchall()]

    def get_candidates(self, limit: int = None) -> List[BrandName]:
        """Get names that are candidates or on shortlist"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            query = """
                SELECT * FROM names
                WHERE status IN (?, ?)
                ORDER BY score_phonaesthetic DESC
            """
            if limit:
                query += f" LIMIT {limit}"
            cursor = conn.execute(query, (NameStatus.CANDIDATE.value, NameStatus.SHORTLIST.value))
            return [self._row_to_brand(row) for row in cursor.fetchall()]

    def get_blocked(self) -> List[BrandName]:
        """Get all blocked names"""
        return self.get_by_status(NameStatus.BLOCKED)

    def search(self, query: str) -> List[BrandName]:
        """Search names by text (partial match)"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM names
                WHERE name_lower LIKE ?
                ORDER BY score_phonaesthetic DESC
            """, (f"%{query.lower()}%",))
            return [self._row_to_brand(row) for row in cursor.fetchall()]

    def get_by_tag(self, tag: str) -> List[BrandName]:
        """Get all names with a specific tag"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT n.* FROM names n
                JOIN tags t ON n.id = t.name_id
                WHERE t.tag = ?
                ORDER BY n.score_phonaesthetic DESC
            """, (tag.lower(),))
            return [self._row_to_brand(row) for row in cursor.fetchall()]

    def filter_blocked(self, names: List[str]) -> List[str]:
        """Filter out blocked names from a list"""
        if not names:
            return names

        with sqlite3.connect(self.db_path) as conn:
            placeholders = ','.join('?' * len(names))
            cursor = conn.execute(f"""
                SELECT name_lower FROM names
                WHERE name_lower IN ({placeholders}) AND status = ?
            """, [*[n.lower() for n in names], NameStatus.BLOCKED.value])
            blocked = {row[0] for row in cursor.fetchall()}

        return [n for n in names if n.lower() not in blocked]

    # =========================================================================
    # Phonaesthetic & Validation (v0.4.0)
    # =========================================================================

    def update_status(self, name: str, status: NameStatus) -> bool:
        """
        Alias for set_status() for convenience.

        Args:
            name: Brand name
            status: New status (can be NameStatus enum or string)
        """
        if isinstance(status, str):
            status = NameStatus(status)
        return self.set_status(name, status)

    def count_by_status(self, status: NameStatus = None) -> int:
        """
        Count names by status.

        Args:
            status: Status to count, or None for total count

        Returns:
            Number of names with that status
        """
        with sqlite3.connect(self.db_path) as conn:
            if status:
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM names WHERE status = ?",
                    (status.value,)
                )
            else:
                cursor = conn.execute("SELECT COUNT(*) FROM names")
            return cursor.fetchone()[0]

    def update_phonaesthetic_scores(self,
                                    name: str,
                                    overall: float = None,
                                    consonant: float = None,
                                    vowel: float = None,
                                    fluency: float = None,
                                    rhythm: float = None,
                                    naturalness: float = None,
                                    memorability: float = None,
                                    cluster_quality: float = None,
                                    ending_quality: float = None,
                                    quality_tier: str = None) -> bool:
        """
        Update phonaesthetic scores for a name.

        Args:
            name: Brand name
            overall: Overall phonaesthetic score (0.0-1.0)
            consonant: Consonant quality score
            vowel: Vowel quality score
            fluency: Processing fluency score
            rhythm: Rhythm score
            naturalness: Phonotactic naturalness score
            memorability: Memorability score
            cluster_quality: Consonant cluster quality score
            ending_quality: Ending quality score
            quality_tier: Quality tier (excellent/good/acceptable/poor)

        Returns:
            True if updated successfully
        """
        kwargs = {}
        if overall is not None:
            kwargs['score_phonaesthetic'] = overall
        if consonant is not None:
            kwargs['score_consonant'] = consonant
        if vowel is not None:
            kwargs['score_vowel'] = vowel
        if fluency is not None:
            kwargs['score_fluency'] = fluency
        if rhythm is not None:
            kwargs['score_rhythm'] = rhythm
        if naturalness is not None:
            kwargs['score_naturalness'] = naturalness
        if memorability is not None:
            kwargs['score_memorability'] = memorability
        if cluster_quality is not None:
            kwargs['score_cluster_quality'] = cluster_quality
        if ending_quality is not None:
            kwargs['score_ending_quality'] = ending_quality
        if quality_tier is not None:
            kwargs['quality_tier'] = quality_tier

        return self.update(name, **kwargs) if kwargs else False

    def update_validation_results(self,
                                  name: str,
                                  eu_conflict: bool = None,
                                  us_conflict: bool = None,
                                  domains: dict = None) -> bool:
        """
        Update validation results for a name.

        Args:
            name: Brand name
            eu_conflict: True if EU trademark conflict found
            us_conflict: True if US trademark conflict found
            domains: Dict of domain availability {'.com': True, '.de': False, ...}

        Returns:
            True if updated successfully
        """
        kwargs = {'validated_at': self._now()}
        if eu_conflict is not None:
            kwargs['eu_conflict'] = eu_conflict
        if us_conflict is not None:
            kwargs['us_conflict'] = us_conflict
        if domains is not None:
            kwargs['domains_available'] = json.dumps(domains)

        return self.update(name, **kwargs)

    def get_by_quality_tier(self, tier: QualityTier, limit: int = None) -> List[BrandName]:
        """
        Get all names with a specific quality tier.

        Args:
            tier: Quality tier (QualityTier enum or string)
            limit: Maximum number of results

        Returns:
            List of BrandName objects
        """
        if isinstance(tier, QualityTier):
            tier = tier.value

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            query = """
                SELECT * FROM names
                WHERE quality_tier = ?
                ORDER BY score_phonaesthetic DESC
            """
            if limit:
                query += f" LIMIT {limit}"
            cursor = conn.execute(query, (tier,))
            return [self._row_to_brand(row) for row in cursor.fetchall()]

    def get_excellent(self, limit: int = None) -> List[BrandName]:
        """Get all names with excellent quality tier."""
        return self.get_by_quality_tier(QualityTier.EXCELLENT, limit)

    def get_available(self, limit: int = None) -> List[BrandName]:
        """
        Get names that passed validation (no EU or US conflicts).

        Returns:
            List of BrandName objects with no trademark conflicts
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            query = """
                SELECT * FROM names
                WHERE validated_at IS NOT NULL
                  AND (eu_conflict = 0 OR eu_conflict IS NULL)
                  AND (us_conflict = 0 OR us_conflict IS NULL)
                ORDER BY score_phonaesthetic DESC
            """
            if limit:
                query += f" LIMIT {limit}"
            cursor = conn.execute(query)
            return [self._row_to_brand(row) for row in cursor.fetchall()]

    def get_conflicts(self, limit: int = None) -> List[BrandName]:
        """
        Get names that have trademark conflicts.

        Returns:
            List of BrandName objects with EU or US conflicts
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            query = """
                SELECT * FROM names
                WHERE eu_conflict = 1 OR us_conflict = 1
                ORDER BY score_phonaesthetic DESC
            """
            if limit:
                query += f" LIMIT {limit}"
            cursor = conn.execute(query)
            return [self._row_to_brand(row) for row in cursor.fetchall()]

    # =========================================================================
    # Statistics
    # =========================================================================

    def stats(self) -> dict:
        """Get database statistics"""
        with sqlite3.connect(self.db_path) as conn:
            stats = {'total': 0, 'by_status': {}, 'by_block_reason': {}}

            # Total
            cursor = conn.execute("SELECT COUNT(*) FROM names")
            stats['total'] = cursor.fetchone()[0]

            # By status
            cursor = conn.execute("""
                SELECT status, COUNT(*) FROM names GROUP BY status
            """)
            for row in cursor.fetchall():
                stats['by_status'][row[0]] = row[1]

            # By block reason
            cursor = conn.execute("""
                SELECT block_reason, COUNT(*) FROM names
                WHERE block_reason IS NOT NULL
                GROUP BY block_reason
            """)
            for row in cursor.fetchall():
                stats['by_block_reason'][row[0]] = row[1]

            # Average score
            cursor = conn.execute("SELECT AVG(score_phonaesthetic) FROM names WHERE score_phonaesthetic IS NOT NULL")
            stats['avg_score'] = cursor.fetchone()[0]

            # Top score
            cursor = conn.execute("SELECT MAX(score_phonaesthetic) FROM names")
            stats['top_score'] = cursor.fetchone()[0]

            return stats

    # =========================================================================
    # Export/Import
    # =========================================================================

    def export_json(self, filepath: str = None) -> str:
        """Export entire database to JSON"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM names ORDER BY score_phonaesthetic DESC")
            names = []
            for row in cursor.fetchall():
                brand = self._row_to_brand(row)

                # Load comments
                c = conn.execute("""
                    SELECT text, author, created_at FROM comments
                    WHERE name_id = ? ORDER BY created_at
                """, (row['id'],))
                brand.comments = [
                    Comment(id=0, text=r['text'], author=r['author'], created_at=r['created_at'])
                    for r in c.fetchall()
                ]

                # Load tags
                c = conn.execute("SELECT tag FROM tags WHERE name_id = ?", (row['id'],))
                tags = [r[0] for r in c.fetchall()]

                data = brand.to_dict()
                data['tags'] = tags
                names.append(data)

        json_str = json.dumps(names, indent=2, ensure_ascii=False)
        if filepath:
            Path(filepath).write_text(json_str, encoding='utf-8')
        return json_str

    def count(self) -> int:
        """Count total names"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM names")
            return cursor.fetchone()[0]

    def count_blocked(self) -> int:
        """Count blocked names"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM names WHERE status = ?",
                (NameStatus.BLOCKED.value,)
            )
            return cursor.fetchone()[0]


# Singleton
_default_db = None

def get_namedb() -> BrandNameDB:
    """Get default database instance"""
    global _default_db
    if _default_db is None:
        _default_db = BrandNameDB()
    return _default_db


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Brand Name Project Database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s add Fluxon --score 0.95 --method rules
  %(prog)s status Fluxon candidate
  %(prog)s comment Fluxon "Klingt technisch aber warm" --author Max
  %(prog)s block Voltex --reason euipo_conflict --notes "3 Treffer"
  %(prog)s list --status candidate
  %(prog)s search flux
  %(prog)s tag Fluxon energie technik
  %(prog)s show Fluxon
  %(prog)s stats
  %(prog)s export projekt.json
        """
    )

    subparsers = parser.add_subparsers(dest='command')
    cli_defaults = get_setting("namedb_cli", {}) or {}
    list_limit_default = cli_defaults.get("list_limit")
    status_limit_default = cli_defaults.get("status_limit")
    block_reason_default = cli_defaults.get("block_reason_default")
    if list_limit_default is None or status_limit_default is None or block_reason_default is None:
        raise ValueError("namedb_cli defaults must be set in app.yaml")

    # Add
    p = subparsers.add_parser('add', help='Add a new name')
    p.add_argument('name', help='Brand name')
    p.add_argument('--score', type=float, help='Overall score')
    p.add_argument('--method', help='Generation method')
    p.add_argument('--associations', help='Semantic associations')

    # Status
    p = subparsers.add_parser('status', help='Set status')
    p.add_argument('name', help='Brand name')
    p.add_argument('new_status', choices=[s.value for s in NameStatus], help='New status')

    # Comment
    p = subparsers.add_parser('comment', help='Add comment')
    p.add_argument('name', help='Brand name')
    p.add_argument('text', help='Comment text')
    p.add_argument('--author', '-a', help='Author name')

    # Block
    p = subparsers.add_parser('block', help='Block a name')
    p.add_argument('name', help='Brand name')
    p.add_argument('--reason', '-r', choices=[r.value for r in BlockReason],
                   default=block_reason_default)
    p.add_argument('--notes', '-n', help='Notes')

    # Unblock
    p = subparsers.add_parser('unblock', help='Unblock a name')
    p.add_argument('name', help='Brand name')

    # List
    p = subparsers.add_parser('list', help='List names')
    p.add_argument('--status', '-s', choices=[s.value for s in NameStatus], help='Filter by status')
    p.add_argument('--limit', '-l', type=int, default=list_limit_default, help='Limit results')

    # Search
    p = subparsers.add_parser('search', help='Search names')
    p.add_argument('query', help='Search query')

    # Tag
    p = subparsers.add_parser('tag', help='Add tags')
    p.add_argument('name', help='Brand name')
    p.add_argument('tags', nargs='+', help='Tags to add')

    # Show
    p = subparsers.add_parser('show', help='Show name details')
    p.add_argument('name', help='Brand name')

    # Stats
    subparsers.add_parser('stats', help='Show statistics')

    # Export
    p = subparsers.add_parser('export', help='Export to JSON')
    p.add_argument('filepath', help='Output file')

    # Candidates
    p = subparsers.add_parser('candidates', help='Show candidates and shortlist')
    p.add_argument('--limit', '-l', type=int, default=status_limit_default)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    db = BrandNameDB()

    if args.command == 'add':
        if db.add(args.name, score=args.score, method=args.method,
                  semantic_associations=args.associations):
            print(f"✓ Added: {args.name}")
        else:
            print(f"⚠ Already exists: {args.name}")

    elif args.command == 'status':
        if db.set_status(args.name, NameStatus(args.new_status)):
            print(f"✓ {args.name} → {args.new_status}")
        else:
            print(f"✗ Not found: {args.name}")

    elif args.command == 'comment':
        if db.add_comment(args.name, args.text, args.author):
            print(f"✓ Comment added to {args.name}")
        else:
            print(f"✗ Not found: {args.name}")

    elif args.command == 'block':
        if db.block(args.name, BlockReason(args.reason), args.notes):
            print(f"✓ Blocked: {args.name} ({args.reason})")

    elif args.command == 'unblock':
        if db.unblock(args.name):
            print(f"✓ Unblocked: {args.name}")
        else:
            print(f"✗ Not found or not blocked: {args.name}")

    elif args.command == 'list':
        if args.status:
            names = db.get_by_status(NameStatus(args.status), args.limit)
        else:
            with sqlite3.connect(db.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT * FROM names ORDER BY score_phonaesthetic DESC LIMIT ?",
                    (args.limit,)
                )
                names = [db._row_to_brand(row) for row in cursor.fetchall()]

        if not names:
            print("No names found.")
        else:
            print(f"\n{'Name':<15} {'Status':<12} {'Score':<6} {'Method':<8}")
            print("=" * 50)
            for n in names:
                score = f"{n.score:.2f}" if n.score else "-"
                print(f"{n.name:<15} {n.status.value:<12} {score:<6} {n.method or '-':<8}")

    elif args.command == 'search':
        names = db.search(args.query)
        if not names:
            print(f"No matches for '{args.query}'")
        else:
            for n in names:
                score = f"{n.score:.2f}" if n.score else "-"
                print(f"{n.name:<15} {n.status.value:<12} {score}")

    elif args.command == 'tag':
        for tag in args.tags:
            if db.add_tag(args.name, tag):
                print(f"✓ Tag added: {tag}")
            else:
                print(f"⚠ Tag exists or name not found: {tag}")

    elif args.command == 'show':
        brand = db.get(args.name)
        if not brand:
            print(f"Not found: {args.name}")
            return

        print(f"\n{'='*50}")
        print(f"Name: {brand.name}")
        print(f"Status: {brand.status.value}")
        print(f"Score: {brand.score:.2f}" if brand.score else "Score: -")
        print(f"Method: {brand.method or '-'}")
        if brand.semantic_associations:
            print(f"Associations: {brand.semantic_associations}")
        if brand.euipo_checked:
            print(f"EUIPO: {brand.euipo_matches} matches")
            print(f"EUIPO URL: {brand.euipo_url}")
        if brand.block_reason:
            print(f"Block reason: {brand.block_reason}")
            if brand.block_notes:
                print(f"Block notes: {brand.block_notes}")

        tags = db.get_tags(args.name)
        if tags:
            print(f"Tags: {', '.join(tags)}")

        if brand.comments:
            print(f"\nComments:")
            for c in brand.comments:
                author = f" ({c.author})" if c.author else ""
                print(f"  • {c.text}{author}")

        print(f"\nCreated: {brand.created_at[:10]}")
        print(f"Updated: {brand.updated_at[:10]}")

    elif args.command == 'stats':
        s = db.stats()
        print(f"\nDatabase Statistics")
        print("=" * 40)
        print(f"Total names: {s['total']}")
        if s['avg_score']:
            print(f"Average score: {s['avg_score']:.2f}")
        if s['top_score']:
            print(f"Top score: {s['top_score']:.2f}")
        print(f"\nBy status:")
        for status, count in s['by_status'].items():
            print(f"  {status}: {count}")
        if s['by_block_reason']:
            print(f"\nBy block reason:")
            for reason, count in s['by_block_reason'].items():
                print(f"  {reason}: {count}")

    elif args.command == 'export':
        db.export_json(args.filepath)
        print(f"✓ Exported to {args.filepath}")

    elif args.command == 'candidates':
        names = db.get_candidates(args.limit)
        if not names:
            print("No candidates yet.")
        else:
            print(f"\nCandidates & Shortlist ({len(names)} total)")
            print("=" * 50)
            for n in names:
                score = f"{n.score:.2f}" if n.score else "-"
                marker = "★" if n.status == NameStatus.SHORTLIST else " "
                print(f"{marker} {n.name:<15} {score}")


if __name__ == '__main__':
    main()
