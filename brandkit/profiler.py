#!/usr/bin/env python3
"""
Discovery Profiler
==================
Lightweight profiling for the discovery pipeline.

Usage:
    brandkit discover --target 10 --profile
    brandkit discover --target 10 --profile --profile-output profile.json
"""

import json
import statistics
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class StageStats:
    """Statistics for a single profiled stage."""
    times: list = field(default_factory=list)
    items: int = 0
    sub_stages: dict = field(default_factory=dict)

    @property
    def total(self) -> float:
        return sum(self.times)

    @property
    def count(self) -> int:
        return len(self.times)

    @property
    def mean(self) -> float:
        return statistics.mean(self.times) if self.times else 0

    @property
    def median(self) -> float:
        return statistics.median(self.times) if self.times else 0

    @property
    def stdev(self) -> float:
        return statistics.stdev(self.times) if len(self.times) > 1 else 0

    @property
    def per_item(self) -> float:
        return self.total / self.items if self.items else 0

    @property
    def min(self) -> float:
        return min(self.times) if self.times else 0

    @property
    def max(self) -> float:
        return max(self.times) if self.times else 0

    def to_dict(self) -> dict:
        return {
            'total_seconds': self.total,
            'count': self.count,
            'items': self.items,
            'mean_seconds': self.mean,
            'median_seconds': self.median,
            'stdev_seconds': self.stdev,
            'min_seconds': self.min,
            'max_seconds': self.max,
            'per_item_ms': self.per_item * 1000,
            'sub_stages': {k: v.to_dict() for k, v in self.sub_stages.items()},
        }


class DiscoveryProfiler:
    """
    Profiler for the discovery pipeline.

    Example:
        profiler = DiscoveryProfiler(enabled=True)
        profiler.start()

        with profiler.stage("generation", items=100):
            names = generate_names(100)

        for name in names:
            with profiler.stage("scoring"):
                score = calculate_score(name)

        print(profiler.report())
    """

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.stages: dict[str, StageStats] = defaultdict(StageStats)
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self._stage_stack: list[str] = []

    def start(self):
        """Start the profiling session."""
        if self.enabled:
            self.start_time = time.perf_counter()

    def stop(self):
        """Stop the profiling session."""
        if self.enabled:
            self.end_time = time.perf_counter()

    @property
    def total_time(self) -> float:
        """Total elapsed time."""
        if not self.start_time:
            return 0
        end = self.end_time or time.perf_counter()
        return end - self.start_time

    @contextmanager
    def stage(self, name: str, items: int = 1):
        """
        Context manager to time a stage.

        Args:
            name: Stage name (e.g., "generation", "domain_check")
            items: Number of items processed in this call (for per-item stats)
        """
        if not self.enabled:
            yield
            return

        # Handle nested stages
        if self._stage_stack:
            parent = self._stage_stack[-1]
            full_name = f"{parent}/{name}"
        else:
            full_name = name

        self._stage_stack.append(name)
        start = time.perf_counter()

        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self._stage_stack.pop()

            # Record in appropriate location
            if '/' in full_name:
                # Sub-stage: record in parent's sub_stages
                parts = full_name.split('/')
                parent_name = parts[0]
                sub_name = '/'.join(parts[1:])
                if sub_name not in self.stages[parent_name].sub_stages:
                    self.stages[parent_name].sub_stages[sub_name] = StageStats()
                self.stages[parent_name].sub_stages[sub_name].times.append(elapsed)
                self.stages[parent_name].sub_stages[sub_name].items += items
            else:
                # Top-level stage
                self.stages[name].times.append(elapsed)
                self.stages[name].items += items

    def record(self, name: str, elapsed: float, items: int = 1):
        """
        Manually record a timing (for cases where context manager isn't convenient).

        Args:
            name: Stage name
            elapsed: Elapsed time in seconds
            items: Number of items processed
        """
        if not self.enabled:
            return
        self.stages[name].times.append(elapsed)
        self.stages[name].items += items

    def report(self, detailed: bool = True) -> str:
        """
        Generate a profiling report.

        Args:
            detailed: Include per-item and sub-stage breakdown
        """
        if not self.enabled or not self.stages:
            return ""

        self.stop()
        total = self.total_time

        lines = [
            "",
            "=" * 70,
            "PROFILING REPORT",
            "=" * 70,
            f"Total time: {total:.2f}s",
            "",
        ]

        # Summary table
        header = f"{'Stage':<22} {'Total':>8} {'%':>6} {'Calls':>6} {'Items':>7} {'Mean':>8} {'Per-item':>10}"
        lines.append(header)
        lines.append("-" * len(header))

        sorted_stages = sorted(self.stages.items(), key=lambda x: -x[1].total)

        for name, stats in sorted_stages:
            pct = (stats.total / total) * 100 if total > 0 else 0
            per_item_str = f"{stats.per_item * 1000:.1f}ms" if stats.items > 0 else "-"
            lines.append(
                f"{name:<22} {stats.total:>7.2f}s {pct:>5.1f}% "
                f"{stats.count:>6} {stats.items:>7} {stats.mean:>7.3f}s {per_item_str:>10}"
            )

            # Sub-stages
            if detailed and stats.sub_stages:
                for sub_name, sub_stats in sorted(stats.sub_stages.items(), key=lambda x: -x[1].total):
                    sub_pct = (sub_stats.total / stats.total) * 100 if stats.total > 0 else 0
                    sub_per_item = f"{sub_stats.per_item * 1000:.1f}ms" if sub_stats.items > 0 else "-"
                    lines.append(
                        f"  └─{sub_name:<18} {sub_stats.total:>7.2f}s {sub_pct:>5.1f}% "
                        f"{sub_stats.count:>6} {sub_stats.items:>7} {sub_stats.mean:>7.3f}s {sub_per_item:>10}"
                    )

        # Time accounting
        accounted = sum(s.total for s in self.stages.values())
        overhead = total - accounted
        if overhead > 0.01:  # Only show if > 10ms
            lines.append(f"{'(overhead)':<22} {overhead:>7.2f}s {(overhead/total)*100:>5.1f}%")

        lines.append("")

        # Bottleneck analysis
        if sorted_stages:
            lines.append("Bottleneck Analysis:")
            lines.append("-" * 40)

            # Top time consumer
            top_stage, top_stats = sorted_stages[0]
            lines.append(f"  Slowest stage: {top_stage} ({top_stats.total:.2f}s, {(top_stats.total/total)*100:.1f}%)")

            # Slowest per-item
            per_item_sorted = sorted(
                [(n, s) for n, s in self.stages.items() if s.items > 0],
                key=lambda x: -x[1].per_item
            )
            if per_item_sorted:
                slow_name, slow_stats = per_item_sorted[0]
                lines.append(f"  Slowest per-item: {slow_name} ({slow_stats.per_item*1000:.1f}ms/item)")

            # High variance stages (potential optimization targets)
            high_var = [(n, s) for n, s in self.stages.items() if s.count > 1 and s.stdev > s.mean * 0.5]
            if high_var:
                lines.append(f"  High variance: {', '.join(n for n, _ in high_var)}")

        lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Export profiling data as a dictionary (for JSON export)."""
        self.stop()
        return {
            'total_seconds': self.total_time,
            'stages': {name: stats.to_dict() for name, stats in self.stages.items()},
        }

    def save_json(self, path: str):
        """Save profiling data to a JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# Global profiler instance (can be replaced per-run)
_profiler: Optional[DiscoveryProfiler] = None


def get_profiler() -> Optional[DiscoveryProfiler]:
    """Get the current global profiler."""
    return _profiler


def set_profiler(profiler: DiscoveryProfiler):
    """Set the global profiler."""
    global _profiler
    _profiler = profiler


@contextmanager
def profile_stage(name: str, items: int = 1):
    """
    Convenience function to profile a stage using the global profiler.

    Example:
        with profile_stage("generation", items=100):
            names = generate(100)
    """
    profiler = get_profiler()
    if profiler:
        with profiler.stage(name, items):
            yield
    else:
        yield
